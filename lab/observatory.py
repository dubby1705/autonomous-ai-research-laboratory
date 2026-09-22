"""
AARL Observatory — read-only instrumentation of the research loop.
==================================================================
The interface must never invent research. Every number, status, event, node and
state this module returns is derived from records the engines already stored in
the research memory (``research_memory/``) plus the live state of a running
exploration. Nothing here runs a model, fabricates a metric or guesses a result:
when a record does not exist the view says so explicitly (``"no recorded ... yet"``)
and carries a ``note`` describing exactly what is missing.

Views
-----
``runs``          slim run list (the heavy possibility-space payload stays in
                  ``/api/explorations/<run_id>/space``)
``pipeline``      the research phases the command centre draws, each mapped onto
                  the *real* backend stage labels of a run
``agents``        specialised engines as nodes: the stage each one owns, what it
                  consumes, what it produced, how recently and how much
``events``        research activity stream (append-only history ledger + live
                  stage transitions, rendered as human sentences)
``hypotheses``    Hypothesis Laboratory records with a state derived from stored
                  simulation / failure / refinement records
``experiments``   Experiment Monitor records: parameters, measured metrics,
                  compute budget, runtime, curves — measurements only
``thinking``      computational research field: question -> concepts -> evidence
                  -> relationships -> hypotheses -> experiments -> validation,
                  wired only by stored links
``graph``         typed record graph (fast variant of ``/api/graph``)
``report``        research report assembled from stored records
``architecture``  system map with live per-component status

Honesty rules encoded here
--------------------------
* A state is only ever derived from stored records, never from prose.
* Links declare their provenance: ``recorded_id`` (an id the engine wrote on the
  record) or ``referenced_by_text`` (an id that occurs inside a stored text
  field, e.g. a hypothesis naming ``DIR-012``).
* Every list view reports what it truncated.
* Every derived state carries the rule id and the reason it was chosen, so the
  GUI can show *why* it says what it says.
* Nothing here writes to the memory; the module is strictly read-only.
"""

from __future__ import annotations

import datetime
import os
import re
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

#: The phases the command centre draws. ``stages`` names the backend stage keys
#: (lab.web_api.STAGE_LABELS) whose recorded state feeds this phase, so the GUI
#: can always show which real pipeline step a phase reflects.
PIPELINE_PHASES: Tuple[Dict[str, Any], ...] = (
    {"key": "question", "label": "RESEARCH QUESTION", "stages": (),
     "meaning": "The problem exactly as it was submitted to AARL."},
    {"key": "analysis", "label": "PROBLEM ANALYSIS", "stages": ("understanding", "domains"),
     "meaning": "The problem is parsed, its subject and research domain(s) are "
                "detected from the engine's own domain index."},
    {"key": "knowledge", "label": "KNOWLEDGE SEARCH", "stages": ("literature", "graph"),
     "meaning": "AARL's knowledge base is searched for overlap and the possibility "
                "space (facets x transferable mechanisms) is built."},
    {"key": "hypotheses", "label": "HYPOTHESIS GENERATION", "stages": ("exploring", "hypotheses"),
     "meaning": "Candidate directions become stored hypotheses with parameters, "
                "constraints and an expected outcome."},
    {"key": "design", "label": "EXPERIMENT DESIGN", "stages": ("mechanisms",),
     "meaning": "Each hypothesis is mapped onto an executable model in the "
                "simulation registry; conceptual mechanisms stay labelled as such."},
    {"key": "simulation", "label": "SIMULATION", "stages": ("simulations",),
     "meaning": "Seeded baseline-vs-proposal runs are executed and their "
                "measurements written to memory as Simulation JSON."},
    {"key": "validation", "label": "VALIDATION", "stages": ("evaluating",),
     "meaning": "Recorded outcomes, contradictions and failures decide each "
                "direction's status; failed ideas are kept as lessons."},
    {"key": "survivors", "label": "SURVIVING HYPOTHESES", "stages": (),
     "meaning": "What remains after validation: directions and hypotheses with no "
                "recorded contradiction, and the evidence behind them."},
)

#: Ids that AARL engines write into records. Used to rebuild links honestly.
_ID_PATTERN = re.compile(r"\b(?:PAPER|HYP|EXP|FAIL|LESSON|RWE|RWR|DIR|RUN)-\d+\b")
_DIR_PATTERN = re.compile(r"\bDIR-\d+\b")

_COLLECTIONS = ("papers", "hypotheses", "simulations", "failures", "lessons",
                "real_world_experiments", "directions", "explorations")


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _stamp(value: Any) -> str:
    return str(value or "")


def _age_seconds(value: Any) -> Optional[float]:
    """Seconds since a stored ISO timestamp, or ``None`` when unparseable."""
    text = _stamp(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            then = datetime.datetime.strptime(text[:26], fmt)
            return max(0.0, (datetime.datetime.now() - then).total_seconds())
        except ValueError:
            continue
    return None



# --------------------------------------------------------------------------- cache
def _memory_signature(memory: Any, collections: Sequence[str]) -> Tuple[Any, ...]:
    """Cheap change-signature of the store parts a view depends on.

    Uses file mtime/size and directory entry counts only, so building a signature
    never re-reads a record.
    """
    parts: List[Any] = []
    for path in (memory.knowledge_path, memory.history_path):
        try:
            stat = os.stat(path)
            parts.append((os.path.basename(path), int(stat.st_mtime), stat.st_size))
        except OSError:
            parts.append((os.path.basename(path), 0, 0))
    for collection in collections:
        try:
            directory = memory.collection_dir(collection)
            parts.append((collection, int(os.stat(directory).st_mtime),
                          len(os.listdir(directory))))
        except (OSError, ValueError):
            parts.append((collection, 0, 0))
    return tuple(parts)


class _SignatureCache:
    """Tiny thread-safe cache invalidated by store changes (plus a short TTL).

    The GUI polls; several views share the same record loads. Keys include the
    store signature, so a newly published record is never hidden behind stale
    data for more than ``ttl`` seconds.
    """

    def __init__(self, ttl: float = 3.0) -> None:
        self.ttl = float(ttl)
        self._lock = threading.Lock()
        self._entries: Dict[str, Tuple[Any, float, Any]] = {}

    def get(self, key: str, signature: Any, builder: Callable[[], Any]) -> Any:
        now = time.time()
        with self._lock:
            hit = self._entries.get(key)
            if hit and hit[0] == signature and (now - hit[1]) < self.ttl:
                return hit[2]
        value = builder()
        with self._lock:
            self._entries[key] = (signature, time.time(), value)
        return value


_CACHE = _SignatureCache()



# --------------------------------------------------------------------------- helpers
def _records(memory: Any, collections: Sequence[str] = _COLLECTIONS,
             cache_key: str = "records") -> Dict[str, List[Dict[str, Any]]]:
    """Load the collections once (cached) and hand them to a view."""
    signature = _memory_signature(memory, collections)

    def build() -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for collection in collections:
            out[collection] = list(memory.load_collection(collection))
        return out

    return _CACHE.get("%s:records" % cache_key, signature, build)


def _knowledge_items(knowledge: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in knowledge.items():
        try:
            items.append(item.to_dict())
        except Exception:  # noqa: BLE001 - one malformed item must not break a view
            continue
    return items


def _newest(records: Iterable[Dict[str, Any]], field: str = "stored_at") -> str:
    newest = ""
    for record in records:
        stamp = _stamp(record.get(field) or record.get("timestamp"))
        if stamp > newest:
            newest = stamp
    return newest


def _sorted_recent(records: Iterable[Dict[str, Any]], limit: Optional[int] = None,
                   numeric: bool = False) -> List[Dict[str, Any]]:
    """Newest first; ``numeric`` orders by the trailing number of the record id."""
    rows = list(records)

    def numeric_key(record: Dict[str, Any]) -> int:
        digits = re.findall(r"\d+", _stamp(record.get("record_id")))
        return int(digits[-1]) if digits else 0

    if numeric:
        rows.sort(key=numeric_key, reverse=True)
    else:
        rows.sort(key=lambda r: _stamp(r.get("stored_at") or r.get("timestamp")),
                  reverse=True)
    return rows[:limit] if limit else rows


def _ids_in_text(payload: Any) -> List[str]:
    """Ids that occur inside stored text (declared as text references)."""
    found: List[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            found.extend(_ID_PATTERN.findall(value))
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    walk(payload)
    return sorted(set(found))


def _direction_id_of(hypothesis: Dict[str, Any]) -> Tuple[str, str]:
    """(direction_id, link provenance) for a hypothesis; ("", "") when none."""
    for field in ("direction_id", "direction", "source_direction"):
        value = _stamp(hypothesis.get(field))
        if _DIR_PATTERN.fullmatch(value or ""):
            return value, "recorded_id"
    text = "%s %s" % (_stamp(hypothesis.get("research_question")),
                      _stamp(hypothesis.get("hypothesis")))
    match = _DIR_PATTERN.search(text)
    if match:
        return match.group(0), "referenced_by_text"
    return "", ""


def _measured_improvement(metrics: Dict[str, Any]) -> Optional[float]:
    """Best engine-computed improvement in a simulation's metrics, if recorded."""
    values: List[float] = []
    for value in (metrics or {}).values():
        if isinstance(value, dict) and isinstance(value.get("improvement_pct"), (int, float)):
            values.append(float(value["improvement_pct"]))
    return max(values) if values else None


def _downsample(series: Sequence[Any], points: int = 44) -> List[float]:
    """Reduce a stored curve to at most ``points`` samples, keeping its shape."""
    values = [float(v) for v in series if isinstance(v, (int, float))]
    if len(values) <= points:
        return values
    step = len(values) / float(points)
    out: List[float] = []
    for i in range(points):
        start = int(i * step)
        end = max(start + 1, int((i + 1) * step))
        chunk = values[start:end]
        out.append(sum(chunk) / len(chunk))
    return out


def _truncation(total: int, shown: int, what: str) -> List[str]:
    if total <= shown:
        return []
    return ["%d of %d %s shown (newest first)" % (shown, total, what)]


def _run_stages(run: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(stage.get("key")): stage for stage in ((run or {}).get("stages") or [])}


#: A run whose last recorded stage update is older than this is reported as
#: *stalled*, never as live. The server can be stopped mid-run, and a stuck record
#: must not make the interface claim research is happening when it is not.
STALL_AFTER_SECONDS = 900


def live_status(run: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Honest status of a run: live, stalled, complete, failed, queued or idle."""
    recorded = _stamp((run or {}).get("status")) or "idle"
    stages = (run or {}).get("stages") or []
    stamps = [stamp for stamp in (
        [_stamp((run or {}).get("updated_at")), _stamp((run or {}).get("completed_at"))] +
        [_stamp(stage.get("at")) for stage in stages]) if stamp]
    last = max(stamps) if stamps else ""
    age = _age_seconds(last)
    current = ""
    for stage in stages:
        if _stamp(stage.get("state")) == "running":
            current = _stamp(stage.get("label")) or _stamp(stage.get("key"))
            break
    out = {"status": recorded, "recorded_status": recorded, "last_update": last,
           "age_seconds": age, "current_stage": current, "reason": ""}
    if recorded == "running" and age is not None and age > STALL_AFTER_SECONDS:
        out["status"] = "stalled"
        out["reason"] = (
            "the last stage update is %.0f minutes old, so this run was interrupted "
            "before it finished; the records stop where the engine stopped"
            % (age / 60.0))
    return out



# ------------------------------------------------------------------- revisions
def _revision_index(memory: Any, collection: str) -> Dict[str, int]:
    """Highest ``.revN`` written for each record id (one directory listing)."""

    def build() -> Dict[str, int]:
        out: Dict[str, int] = {}
        try:
            names = os.listdir(memory.collection_dir(collection))
        except (OSError, ValueError):
            return out
        for name in names:
            match = re.fullmatch(r"(.+)\.rev(\d+)\.json", name)
            if match:
                base, number = match.group(1), int(match.group(2))
                out[base] = max(out.get(base, 0), number)
        return out

    signature = _memory_signature(memory, (collection,))
    return _CACHE.get("revindex:%s" % collection, signature, build)


def latest_stored(memory: Any, collection: str, record_id: str) -> Optional[Dict[str, Any]]:
    """Newest stored version of one record: the base file or the highest ``.revN``.

    Memory is append-only, so an evaluated record may live in a revision. Views
    that report a record's *current* state must use this, never the base file.
    """
    record_id = _stamp(record_id)
    if not record_id:
        return None
    best = memory.get(collection, record_id)
    number = _revision_index(memory, collection).get(record_id, 0)
    for candidate in range(number, 0, -1):
        body = memory.get(collection, "%s.rev%d" % (record_id, candidate))
        if isinstance(body, dict):
            return body
    return best if isinstance(best, dict) else None


# ---------------------------------------------------------------------- run scope
def run_scope(memory: Any, run: Optional[Dict[str, Any]],
              records: Optional[Dict[str, List[Dict[str, Any]]]] = None
              ) -> Dict[str, Any]:
    """Everything that can honestly be attributed to one exploration run.

    Attribution order:
      1. ids the engines wrote on the records (a direction's ``hypothesis_ids``,
         ``experiment_ids``, ``failure_ids``),
      2. ids referenced inside stored text fields (a hypothesis naming ``DIR-012``),
      3. nothing else — the views never invent an attribution.
    """
    records = records if records is not None else _records(memory)
    summaries = [s for s in ((run or {}).get("directions") or []) if isinstance(s, dict)]
    scope: Dict[str, Any] = {
        "run_id": _stamp((run or {}).get("run_id")),
        "question": _stamp((run or {}).get("problem")),
        "status": _stamp((run or {}).get("status")),
        "window": [_stamp((run or {}).get("created_at")),
                   _stamp((run or {}).get("completed_at") or (run or {}).get("updated_at"))],
        "directions": summaries,
        "direction_ids": sorted({_stamp(s.get("direction_id")) for s in summaries
                                 if _stamp(s.get("direction_id"))}),
    }
    by_hypothesis = {_stamp(h.get("hypothesis_id")): h
                     for h in records.get("hypotheses", [])}
    by_experiment = {_stamp(s.get("experiment_id")): s
                     for s in records.get("simulations", [])}
    hypotheses: List[Dict[str, Any]] = []
    simulations: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    lessons: List[Dict[str, Any]] = []
    hypothesis_links: Dict[str, str] = {}

    for direction_id in scope["direction_ids"]:
        direction = latest_stored(memory, "directions", direction_id) or {}
        for hid in (direction.get("hypothesis_ids") or []):
            hypothesis_links[_stamp(hid)] = "recorded_id"
        for eid in (direction.get("experiment_ids") or []):
            simulation = by_experiment.get(_stamp(eid))
            if simulation:
                simulations.append(simulation)
        for fid in (direction.get("failure_ids") or []):
            wanted = _stamp(fid)
            for failure in records.get("failures", []):
                if _stamp(failure.get("failure_id")) == wanted:
                    failures.append(failure)

    # Text-referenced hypotheses: a stored hypothesis names the direction it came
    # from ("Direction DIR-012: ..."). The link is labelled as a text reference.
    for hypothesis in records.get("hypotheses", []):
        direction_id, provenance = _direction_id_of(hypothesis)
        if direction_id and direction_id in scope["direction_ids"]:
            hypothesis_links.setdefault(_stamp(hypothesis.get("hypothesis_id")), provenance)

    for hid, provenance in hypothesis_links.items():
        hypothesis = by_hypothesis.get(hid)
        if hypothesis:
            hypotheses.append(hypothesis)
        for simulation in records.get("simulations", []):
            if _stamp(simulation.get("hypothesis_id")) == hid:
                simulations.append(simulation)
        for failure in records.get("failures", []):
            if _stamp(failure.get("hypothesis_id")) == hid:
                failures.append(failure)
        for lesson in records.get("lessons", []):
            if _stamp(lesson.get("hypothesis_id")) == hid:
                lessons.append(lesson)

    def dedupe(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            seen[_stamp(row.get(key)) or _stamp(row.get("record_id"))] = row
        return list(seen.values())

    scope["hypotheses"] = _sorted_recent(dedupe(hypotheses, "hypothesis_id"))
    scope["simulations"] = _sorted_recent(dedupe(simulations, "experiment_id"))
    scope["failures"] = _sorted_recent(dedupe(failures, "failure_id"))
    scope["lessons"] = _sorted_recent(dedupe(lessons, "lesson_id"))
    scope["hypothesis_ids"] = [_stamp(h.get("hypothesis_id")) for h in scope["hypotheses"]]
    scope["experiment_ids"] = [_stamp(s.get("experiment_id")) for s in scope["simulations"]]
    scope["failure_ids"] = [_stamp(f.get("failure_id")) for f in scope["failures"]]
    scope["lesson_ids"] = [_stamp(l.get("lesson_id")) for l in scope["lessons"]]
    scope["counts"] = {
        "directions": len(scope["directions"]),
        "hypotheses": len(scope["hypotheses"]),
        "experiments": len(scope["simulations"]),
        "contradictions": len(scope["failures"]),
        "lessons": len(scope["lessons"]),
    }
    scope["link_provenance"] = [
        "Directions: ids recorded on the run record.",
        "Hypotheses: %s." % ("; ".join(sorted(set(hypothesis_links.values())))
                             or "no link recorded for this run"),
        "Experiments and contradictions: %s." % (
            "recorded ids (a direction's hypothesis/experiment/failure lists)"
            if scope["experiment_ids"] or scope["failure_ids"]
            else "no link recorded for this run"),
        "Only ids AARL actually stored are used; nothing is attributed by guesswork.",
    ]
    return scope


# ------------------------------------------------------------------------ pipeline
def _phase_observations(run: Optional[Dict[str, Any]], scope: Dict[str, Any],
                        stages: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """One real observation per phase: recorded numbers, never prose filler."""
    out: Dict[str, List[str]] = {spec["key"]: [] for spec in PIPELINE_PHASES}
    out["question"].append(_stamp(scope.get("question")) or "No research problem recorded.")
    if _stamp(stages.get("domains", {}).get("detail")):
        out["analysis"].append(_stamp(stages["domains"].get("detail")))

    stats = (run or {}).get("stats") or {}
    if stats:
        out["analysis"].append("Subject: %s" % (stats.get("subject_label") or "not recorded"))
        out["knowledge"].append("%s facet(s) x %s concept(s) considered" % (
            stats.get("facets_explored", 0), stats.get("concepts_considered", 0)))
        out["knowledge"].append(
            "%s direction(s) built, %s candidate(s) considered and not taken" % (
                stats.get("directions", 0), stats.get("candidates_rejected", 0)))
    overlap = (run or {}).get("literature_overlap") or []
    if overlap:
        out["knowledge"].append("%d stored knowledge item(s) overlapped the problem" % len(overlap))

    out["hypotheses"].append("%d hypothesis record(s) attributed to this run" % (
        scope.get("counts", {}).get("hypotheses", 0)))
    adapters = sorted({"%s v%s" % (((s.get("adapter") or {}).get("name") or "unknown adapter"),
                                   ((s.get("adapter") or {}).get("version") or "?"))
                       for s in scope.get("simulations", [])})
    out["design"].append("Executable model(s) used: %s" % (
        "; ".join(adapters) if adapters else "none recorded for this run"))
    if _stamp(stages.get("mechanisms", {}).get("detail")):
        out["design"].append(_stamp(stages["mechanisms"].get("detail")))

    simulations = scope.get("simulations", [])
    out["simulation"].append("%d simulation record(s) stored (status: %s)" % (
        len(simulations),
        ", ".join(sorted({_stamp(s.get("status")) for s in simulations})) or "none"))
    measured = [(value, simulation) for value, simulation in
                ((_measured_improvement(s.get("metrics") or {}), s) for s in simulations)
                if value is not None]
    if measured:
        value, simulation = max(measured, key=lambda pair: pair[0])
        out["simulation"].append("Largest measured improvement: %+.2f%% (%s for %s)" % (
            value, simulation.get("experiment_id"), simulation.get("hypothesis_id")))

    out["validation"].append("Recorded contradictions: %d  ·  lessons stored: %d" % (
        scope.get("counts", {}).get("contradictions", 0),
        scope.get("counts", {}).get("lessons", 0)))
    status_counts: Dict[str, int] = {}
    for summary in scope.get("directions", []):
        key = _stamp(summary.get("status")) or "UNKNOWN"
        status_counts[key] = status_counts.get(key, 0) + 1
    if status_counts:
        out["validation"].append("Direction statuses: %s" % ", ".join(
            "%s=%d" % item for item in sorted(status_counts.items())))

    survivors = [s for s in scope.get("directions", [])
                 if _stamp(s.get("status")) in ("EVIDENCE_BACKED", "PROMISING")]
    out["survivors"].append(
        "%d direction(s) in EVIDENCE-BACKED / PROMISING after validation" % len(survivors))
    for summary in survivors[:4]:
        out["survivors"].append("%s · %s" % (summary.get("direction_id"), summary.get("title")))
    return out


def pipeline(run: Optional[Dict[str, Any]], scope: Dict[str, Any]) -> Dict[str, Any]:
    """The command-centre phases, each derived from the run's real stages."""
    stages = _run_stages(run)
    run_state = live_status(run)
    run_status = run_state["status"]
    observations = _phase_observations(run, scope, stages)
    phases: List[Dict[str, Any]] = []
    for spec in PIPELINE_PHASES:
        owned = [stages[key] for key in spec["stages"] if key in stages]
        states = [_stamp(stage.get("state")) for stage in owned]
        if any(state == "running" for state in states):
            state = "RUNNING"
        elif owned and all(state == "done" for state in states):
            state = "COMPLETE"
        elif states:
            state = "PARTIAL"
        elif spec["key"] == "question" and run:
            state = "COMPLETE"
        else:
            state = "PENDING"
        if run_status == "failed" and state in ("RUNNING", "PARTIAL"):
            state = "FAILED"
        if run_status == "stalled" and state == "RUNNING":
            state = "STALLED"
        detail = " · ".join(_stamp(stage.get("detail")) for stage in owned
                            if _stamp(stage.get("detail")) and
                            _stamp(stage.get("detail")) != _stamp(stage.get("label")))
        phases.append({
            "key": spec["key"], "label": spec["label"], "state": state,
            "meaning": spec["meaning"],
            "backend_stages": [{"key": _stamp(s.get("key")), "label": _stamp(s.get("label")),
                                "state": _stamp(s.get("state")), "detail": _stamp(s.get("detail")),
                                "at": _stamp(s.get("at"))} for s in owned],
            "detail": detail or "No backend stage state recorded for this phase yet.",
            "observations": observations.get(spec["key"], []),
        })
    status_counts: Dict[str, int] = {}
    for summary in scope.get("directions", []):
        key = _stamp(summary.get("status")) or "UNKNOWN"
        status_counts[key] = status_counts.get(key, 0) + 1
    measured = [(_measured_improvement(s.get("metrics") or {}), s)
                for s in scope.get("simulations", [])]
    measured = [(value, simulation) for value, simulation in measured if value is not None]
    best = max(measured, key=lambda pair: pair[0]) if measured else None
    return {
        "run_id": scope.get("run_id"), "question": scope.get("question"),
        "status": run_status or "idle", "run_state": run_state, "phases": phases,
        "stage_template": (run or {}).get("stage_template") or [],
        "counts": scope.get("counts", {}),
        "direction_status_counts": status_counts,
        "domains": (run or {}).get("domains") or [],
        "primary_domain": _stamp((run or {}).get("primary_domain")),
        "best_measured_improvement": ({"value": best[0],
                                       "experiment_id": best[1].get("experiment_id"),
                                       "hypothesis_id": best[1].get("hypothesis_id")}
                                      if best else None),
        "link_provenance": scope.get("link_provenance", []),
        "generated_at": _now(),
        "note": ("Phase states are the backend stages of this run mapped one to one. "
                 "Numbers come from stored records; a phase with no recorded state stays PENDING."),
    }


# -------------------------------------------------------------------------- agents
#: The specialised engines of AARL. Each entry names the real module it wraps, the
#: backend stages it owns and the memory collections it writes — which is exactly
#: how the observatory decides whether it is currently working.
AGENT_SPECS: Tuple[Dict[str, Any], ...] = (
    {"key": "problem_analyst", "label": "PROBLEM ANALYST",
     "module": "lab.domains · lab.config.LabConfig",
     "role": "Parses the submitted problem, detects its subject and research "
             "domain(s), and fixes the configuration a cycle runs under.",
     "capabilities": ["problem parsing", "subject detection", "domain detection",
                      "reproducible configuration"],
     "inputs": ["research problem text", "engine domain index (lab/domains.py)",
                "caller configuration"],
     "outputs": ["primary domain", "relevant domains", "run configuration snapshot"],
     "stages": ("understanding", "domains"),
     "collections": ("explorations",)},
    {"key": "literature_researcher", "label": "LITERATURE RESEARCHER",
     "module": "lab.literature · lab.papers.PaperIngestor",
     "role": "Fetches public metadata (arXiv / Crossref), ingests documents the "
             "researcher supplies, and extracts provenance-carrying claims.",
     "capabilities": ["public API search", "PDF / text ingestion", "claim extraction",
                      "source attribution"],
     "inputs": ["search queries", "uploaded documents", "researcher-pasted text"],
     "outputs": ["paper records (PAPER-###)", "knowledge items with provenance"],
     "stages": ("literature",),
     "collections": ("papers",)},
    {"key": "knowledge_engine", "label": "KNOWLEDGE ENGINE",
     "module": "lab.knowledge.KnowledgeManager",
     "role": "Maintains the living knowledge base: sections, claim types, "
             "deduplication and the honest sourced-vs-inferred labels.",
     "capabilities": ["sectioned store", "dedupe index", "claim typing",
                      "overlap search", "withdrawal by source paper"],
     "inputs": ["paper extractions", "hypothesis records", "researcher data"],
     "outputs": ["knowledge items (K-####)", "summary counts", "overlap hits"],
     "stages": ("graph",),
     "collections": ()},
    {"key": "direction_discovery", "label": "RESEARCH DIRECTOR",
     "module": "lab.discovery.directions.DirectionDiscovery",
     "role": "Splits the problem into structural facets, mates them with "
             "transferable mechanisms and builds a possibility space of candidate "
             "directions — including the branches it decided not to take.",
     "capabilities": ["facet analysis", "concept library", "possibility space",
                      "rejected-branch record"],
     "inputs": ["problem + detected domains", "concept library (lab/discovery/concepts.py)",
                "recorded failures from earlier runs"],
     "outputs": ["direction records (DIR-###)", "possibility-space tree", "run statistics"],
     "stages": ("exploring",),
     "collections": ("directions",)},
    {"key": "hypothesis_engine", "label": "HYPOTHESIS ENGINE",
     "module": "lab.hypotheses.HypothesisEngine",
     "role": "Turns a direction into a testable record: text, parameter block, "
             "constraints, expected outcome and the lessons it was built from.",
     "capabilities": ["hypothesis construction", "constraint definition",
                      "lesson reuse", "refinement chain (parent_hypothesis)"],
     "inputs": ["research question", "direction records", "stored lessons"],
     "outputs": ["hypothesis records (HYP-###)"],
     "stages": ("hypotheses",),
     "collections": ("hypotheses",)},
    {"key": "critic_agent", "label": "CRITIC AGENT",
     "module": "lab.discovery.dimensions · lab.discovery.gaps",
     "role": "Challenges every direction descriptively: novelty relative to AARL's "
             "own memory, mechanistic plausibility, evidence strength, unknowns and "
             "the failure conditions that would kill the idea.",
     "capabilities": ["dimension scoring (descriptive)", "gap extraction",
                      "failure-condition listing", "contradiction detection"],
     "inputs": ["direction records", "knowledge items", "simulation outcomes"],
     "outputs": ["dimension blocks", "gaps", "unknowns", "failure conditions"],
     "stages": ("mechanisms",),
     "collections": ()},
    {"key": "experiment_designer", "label": "EXPERIMENT DESIGNER",
     "module": "lab.experiments.ExperimentManager · lab.simulation.registry",
     "role": "Selects an executable model from the simulation registry for each "
             "hypothesis and fixes the seeded baseline-vs-proposal comparison.",
     "capabilities": ["adapter registry", "safe parameter bounds",
                      "baseline selection", "reproducible seeds"],
     "inputs": ["hypothesis parameters", "constraints", "registered adapters"],
     "outputs": ["experiment plan", "reproducibility block (seed, input hash)"],
     "stages": ("mechanisms",),
     "collections": ()},
    {"key": "simulation_engine", "label": "SIMULATION ENGINE",
     "module": "lab.simulation.optimizer_adapter · domain_adapter · custom_adapter",
     "role": "Executes the seeded runs, measures baseline and proposal, and writes "
             "the numbers — measured values only, never model prose.",
     "capabilities": ["seeded execution", "baseline vs proposal", "metric deltas",
                      "constraint evaluation", "loss curves"],
     "inputs": ["experiment plan", "parameter block", "iteration budget"],
     "outputs": ["simulation records (EXP-###) with metrics and curves"],
     "stages": ("simulations",),
     "collections": ("simulations",)},
    {"key": "validation_agent", "label": "VALIDATION AGENT",
     "module": "lab.discovery.status · lab.failures.FailureAnalyzer · lab.lessons",
     "role": "Classifies each direction from stored records only, and keeps every "
             "failed attempt as a failure record plus a lesson for the next round.",
     "capabilities": ["rule-based status", "contradiction accounting",
                      "failure analysis", "lesson extraction"],
     "inputs": ["knowledge items", "simulation outcomes", "recorded contradictions"],
     "outputs": ["direction status (EVIDENCE_BACKED … FAILED_WEAK)",
                 "failure records (FAIL-###)", "lessons (LESSON-###)"],
     "stages": ("evaluating",),
     "collections": ("failures", "lessons")},
    {"key": "synthesis_agent", "label": "SYNTHESIS AGENT",
     "module": "lab.opportunity · lab.reporting.FinalReportGenerator",
     "role": "Assembles the opportunity report and the report layer: what is "
             "supported, what is speculative, what failed and what to do next.",
     "capabilities": ["opportunity report", "report sections", "limitations",
                      "next-step derivation", "graph export"],
     "inputs": ["directions + statuses", "experiments", "failures", "lessons"],
     "outputs": ["research report", "next research directions", "typed graph export"],
     "stages": (),
     "collections": ()},
)


def _agent_counts(spec: Dict[str, Any], records: Dict[str, List[Dict[str, Any]]],
                  knowledge_items: int) -> List[Dict[str, Any]]:
    counts: List[Dict[str, Any]] = []
    for collection in spec["collections"]:
        counts.append({"label": "stored %s" % collection,
                       "value": len(records.get(collection, []))})
    if spec["key"] == "knowledge_engine":
        counts.append({"label": "knowledge items", "value": knowledge_items})
    if spec["key"] == "experiment_designer":
        adapters = {((s.get("adapter") or {}).get("name") or "")
                    for s in records.get("simulations", [])}
        counts.append({"label": "adapters seen in records",
                       "value": len([a for a in adapters if a])})
    if spec["key"] == "critic_agent":
        counts.append({"label": "failure records analysed",
                       "value": len(records.get("failures", []))})
    if spec["key"] == "synthesis_agent":
        counts.append({"label": "lessons available for synthesis",
                       "value": len(records.get("lessons", []))})
    return counts


def _agent_recent(spec: Dict[str, Any], records: Dict[str, List[Dict[str, Any]]],
                  limit: int = 3) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for collection in spec["collections"]:
        for record in _sorted_recent(records.get(collection, []), limit):
            rows.append({
                "kind": collection,
                "id": _stamp(record.get("record_id")),
                "at": _stamp(record.get("stored_at") or record.get("timestamp")),
                "title": _stamp(record.get("title") or record.get("hypothesis") or
                                record.get("problem"))[:150],
            })
    rows.sort(key=lambda row: row["at"], reverse=True)
    return rows[:limit]


def _agent_state(spec: Dict[str, Any], stages: Dict[str, Dict[str, Any]],
                 run_status: str, has_records: bool) -> Tuple[str, str]:
    """(state, reason) — derived only from run stage state and stored records."""
    owned = [stages[key] for key in spec["stages"] if key in stages]
    states = [_stamp(stage.get("state")) for stage in owned]
    if "running" in states and run_status == "failed":
        return "FAULT", "the run that owns this stage failed"
    if "running" in states and run_status == "stalled":
        return "STALLED", "the run stopped mid-stage and never resumed"
    if "running" in states:
        return "ACTIVE", "its backend stage is running right now"
    if states and all(state == "done" for state in states):
        return "COMPLETE", "all of its backend stages finished in the selected run"
    if run_status == "running":
        return "STANDBY", "the selected run is active but has not reached this stage yet"
    if has_records:
        return "IDLE", "no run is active; its records are stored in research memory"
    return "NO DATA", "no run is active and no record of this engine exists yet"


def agents(memory: Any, knowledge: Any, run: Optional[Dict[str, Any]],
           records: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    """The specialised engines of AARL as a live multi-agent system."""
    records = records if records is not None else _records(memory)
    items = _knowledge_items(knowledge)
    stages = _run_stages(run)
    run_state = live_status(run)
    run_status = run_state["status"]
    scope = run_scope(memory, run, records)

    recent_counts: Dict[str, int] = {}
    for spec in AGENT_SPECS:
        recent = 0
        for collection in spec["collections"]:
            for record in records.get(collection, []):
                age = _age_seconds(record.get("stored_at") or record.get("timestamp"))
                if age is not None and age <= 86400:
                    recent += 1
        recent_counts[spec["key"]] = recent
    busiest = max([1] + list(recent_counts.values()))

    scoped_counts = {
        "problem_analyst": 1 if run else 0,
        "literature_researcher": len((run or {}).get("literature_overlap") or []),
        "knowledge_engine": len((run or {}).get("literature_overlap") or []),
        "direction_discovery": scope["counts"]["directions"],
        "hypothesis_engine": scope["counts"]["hypotheses"],
        "critic_agent": scope["counts"]["hypotheses"],
        "experiment_designer": scope["counts"]["experiments"],
        "simulation_engine": scope["counts"]["experiments"],
        "validation_agent": scope["counts"]["contradictions"],
        "synthesis_agent": scope["counts"]["directions"],
    }

    out: List[Dict[str, Any]] = []
    for spec in AGENT_SPECS:
        rows: List[Dict[str, Any]] = []
        for collection in spec["collections"]:
            rows.extend(records.get(collection, []))
        state, reason = _agent_state(spec, stages, run_status, bool(rows))
        last_active = _newest(rows)
        pulse = 1.0 if state == "ACTIVE" else min(
            1.0, recent_counts[spec["key"]] / float(busiest))
        out.append({
            "key": spec["key"], "label": spec["label"], "module": spec["module"],
            "role": spec["role"], "capabilities": list(spec["capabilities"]),
            "inputs": list(spec["inputs"]), "outputs": list(spec["outputs"]),
            "stages": [{"key": key, "label": _stamp(stages.get(key, {}).get("label")),
                        "state": _stamp(stages.get(key, {}).get("state"))}
                       for key in spec["stages"]],
            "state": state, "state_reason": reason,
            "counts": _agent_counts(spec, records, len(items)),
            "records_in_run": scoped_counts.get(spec["key"], 0),
            "records_24h": recent_counts[spec["key"]],
            "last_active": last_active,
            "idle_seconds": _age_seconds(last_active),
            "pulse": round(pulse, 3),
            "pulse_meaning": "share of this engine's record activity in the last 24h "
                             "relative to the busiest engine",
            "recent_records": _agent_recent(spec, records),
        })

    order = {"ACTIVE": 0, "FAULT": 1, "STALLED": 2, "STANDBY": 3,
             "COMPLETE": 4, "IDLE": 5, "NO DATA": 6}
    out.sort(key=lambda row: (order.get(row["state"], 9), row["label"]))
    return {
        "run_id": _stamp((run or {}).get("run_id")),
        "run_status": run_status or "idle",
        "run_state": run_state,
        "agents": out,
        "working": [row["key"] for row in out if row["state"] == "ACTIVE"],
        "generated_at": _now(),
        "note": ("Agent state is derived from which backend stage of the selected run is "
                 "running and from the records each engine has stored. AARL does not "
                 "simulate personas: an engine with no record stays NO DATA."),
    }


# -------------------------------------------------------------------------- events
#: event type -> (kind, severity, sentence template). Severities: ``info``
#: (routine progress), ``notice`` (worth reading), ``success`` (a result stored),
#: ``alert`` (contradiction / failure / error), ``trace`` (append-only store trace).
_EVENT_KINDS: Dict[str, Tuple[str, str, str]] = {
    "research_cycle_started": ("cycle", "notice", "Research cycle opened for %(0)s"),
    "exploration_created": ("run", "notice", "Research problem accepted: %(0)s"),
    "exploration_completed": ("run", "success",
                              "Run closed — %(directions)s research direction(s) recorded."),
    "exploration_failed": ("run", "alert", "Run aborted: %(0)s"),
    "hypothesis_generated": ("hypothesis", "info",
                             "Hypothesis %(hypothesis_id)s generated (iteration "
                             "%(iteration)s) from %(lessons)s stored lesson(s)."),
    "simulation_complete": ("experiment", "info",
                            "%(experiment_id)s finished with status '%(status)s' via "
                            "%(simulation_type)s for %(hypothesis_id)s."),
    "failure_analyzed": ("contradiction", "alert",
                         "Contradiction recorded: %(failure_id)s on %(experiment_id)s "
                         "(%(failure_type)s, severity %(severity)s)."),
    "lesson_recorded": ("lesson", "notice",
                        "Lesson %(lesson_id)s stored from %(failure_id)s — it feeds the "
                        "next hypothesis round for %(hypothesis_id)s."),
    "direction_discovered": ("direction", "success",
                             "Direction %(direction_id)s discovered: %(title)s"),
    "paper_ingested": ("literature", "success", "Paper %(paper_id)s ingested: %(title)s"),
    "paper_removed_from_knowledge": ("literature", "alert",
                                     "Knowledge withdrawn for %(paper_id)s — "
                                     "%(knowledge_items_removed)s item(s) removed."),
    "paper_knowledge_withdrawn": ("literature", "alert",
                                  "Knowledge withdrawn for %(paper_id)s — "
                                  "%(knowledge_items_removed)s item(s) removed."),
    "record_revision": ("store", "trace",
                        "%(record_id)s revised (%(revision_id)s) in %(collection)s"),
}

#: Severities the activity stream can carry.
EVENT_SEVERITIES = ("info", "notice", "success", "alert", "trace")


def _all_ids(event: Dict[str, Any]) -> str:
    """Every id embedded in one ledger event, as one searchable string."""
    return " ".join([_stamp(event.get("run_id"))] +
                    [_stamp(value) for key, value in event.items() if key.endswith("_id")])


def _event_text(event: Dict[str, Any]) -> Tuple[str, str, str]:
    name = _stamp(event.get("event"))
    kind, severity, template = _EVENT_KINDS.get(name, ("store", "info", name))
    payload = dict(event)
    if name == "hypothesis_generated":
        payload["lessons"] = len(event.get("based_on_lessons") or [])
    elif name in ("exploration_created", "research_cycle_started"):
        payload["0"] = _stamp(event.get("problem") or event.get("research_question"))
    elif name == "exploration_failed":
        payload["0"] = _stamp(event.get("error")) or "cause not recorded"
    try:
        text = template % payload
    except (KeyError, TypeError, ValueError):
        text = name
    return kind, severity, text


def _stage_events(runs: Sequence[Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Live stage transitions of the selected runs, as stream entries."""
    out: List[Dict[str, Any]] = []
    for run in runs:
        if not run:
            continue
        for stage in (run.get("stages") or []):
            state = _stamp(stage.get("state"))
            if state not in ("running", "done"):
                continue
            out.append({
                "at": _stamp(stage.get("at")),
                "kind": "stage",
                "severity": "success" if state == "done" else "notice",
                "text": "Stage %s — %s" % (state.upper(), _stamp(stage.get("label"))),
                "detail": _stamp(stage.get("detail")),
                "refs": {"run_id": _stamp(run.get("run_id")), "stage": _stamp(stage.get("key"))},
                "source": "live run stage state",
            })
    return out


def events(memory: Any, runs: Sequence[Optional[Dict[str, Any]]] = (), limit: int = 60,
           run_id: str = "", include_trace: bool = False) -> Dict[str, Any]:
    """Chronological research activity stream built from recorded events only."""
    limit = max(1, min(int(limit or 60), 400))
    signature = _memory_signature(memory, ())

    def load() -> List[Dict[str, Any]]:
        return list(memory.load_history())

    history = _CACHE.get("history", signature, load)
    rows: List[Dict[str, Any]] = []
    for event in history:
        if run_id and run_id not in _all_ids(event):
            continue
        kind, severity, text = _event_text(event)
        if severity == "trace" and not include_trace:
            continue
        rows.append({
            "at": _stamp(event.get("timestamp")),
            "kind": kind, "severity": severity, "text": text,
            "detail": _stamp(event.get("failure_type") or event.get("error")),
            "refs": {key: _stamp(value) for key, value in event.items()
                     if key.endswith("_id") or key in ("run_id", "collection")},
            "source": _stamp(event.get("event")),
        })
    rows.extend(_stage_events(list(runs)))
    rows.sort(key=lambda row: row["at"], reverse=True)
    shown = rows[:limit]
    severity_counts: Dict[str, int] = {}
    for row in shown:
        severity_counts[row["severity"]] = severity_counts.get(row["severity"], 0) + 1
    return {
        "events": shown,
        "total_recorded": len(rows),
        "shown": len(shown),
        "severity_counts": severity_counts,
        "kinds": sorted({row["kind"] for row in rows}),
        "truncated": _truncation(len(rows), len(shown), "events"),
        "generated_at": _now(),
        "note": ("Every line comes from the append-only research history ledger or from a "
                 "live run stage: real timestamps, newest first. Nothing is generated to "
                 "look busy; a quiet system shows a quiet stream."),
    }


# --------------------------------------------------------------------- hypotheses
#: Hypothesis Laboratory states. Each one is derived from stored records only, and
#: every row carries the rule id plus the reason, exactly like the direction status
#: system in ``lab.discovery.status``.
HYPOTHESIS_RULES: Dict[str, Dict[str, str]] = {
    "GENERATED": {
        "rule": "H1_no_experiment_recorded",
        "meaning": "The hypothesis exists as a record; no experiment has been stored for it yet.",
    },
    "UNDER INVESTIGATION": {
        "rule": "H2_experiment_without_measurable_outcome",
        "meaning": "An experiment record exists but it carries no measurable improvement, "
                   "so nothing has been decided either way.",
    },
    "SUPPORTED": {
        "rule": "H3_measured_improvement",
        "meaning": "A stored simulation measured a positive improvement and the record "
                   "carries no contradiction.",
    },
    "CONTRADICTED": {
        "rule": "H4_contradiction_with_support",
        "meaning": "A contradiction is recorded (constraint violation / failed run) while a "
                   "measured improvement also exists: the two records disagree.",
    },
    "REJECTED": {
        "rule": "H5_contradiction_without_support",
        "meaning": "A high-severity contradiction is recorded and no measured support "
                   "exists. The idea is kept in memory as a failure record.",
    },
    "REFINED": {
        "rule": "H6_contradiction_refined_into_child",
        "meaning": "A contradiction was recorded and AARL stored a refined child "
                   "hypothesis from it (parent_hypothesis chain).",
    },
    "SURVIVED": {
        "rule": "H7_supported_and_clean",
        "meaning": "Measured improvement on record, no recorded contradiction, and the "
                   "idea was carried forward into later records.",
    },
}


def _hypothesis_index(records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """One pass over the store: everything the laboratory needs to classify."""
    children: Dict[str, List[str]] = {}
    for hypothesis in records.get("hypotheses", []):
        parent = _stamp(hypothesis.get("parent_hypothesis"))
        if parent:
            children.setdefault(parent, []).append(_stamp(hypothesis.get("hypothesis_id")))
    simulations: Dict[str, List[Dict[str, Any]]] = {}
    for simulation in records.get("simulations", []):
        simulations.setdefault(_stamp(simulation.get("hypothesis_id")), []).append(simulation)
    failures: Dict[str, List[Dict[str, Any]]] = {}
    for failure in records.get("failures", []):
        failures.setdefault(_stamp(failure.get("hypothesis_id")), []).append(failure)
    lessons: Dict[str, List[Dict[str, Any]]] = {}
    for lesson in records.get("lessons", []):
        lessons.setdefault(_stamp(lesson.get("hypothesis_id")), []).append(lesson)
    return {"children": children, "simulations": simulations,
            "failures": failures, "lessons": lessons}


def _hypothesis_state(row: Dict[str, Any]) -> Dict[str, Any]:
    """(state, rule, reason) from the linked records — never from prose."""
    simulations = row.get("experiments", [])
    contradictions = row.get("contradictions", [])
    children = row.get("refinements", [])
    improvements = [value for value in
                    (_measured_improvement(s.get("metrics") or {}) for s in simulations)
                    if value is not None]
    best = max(improvements) if improvements else None
    supported = bool(best is not None and best > 0)
    if not simulations:
        state, reason = "GENERATED", "no experiment record refers to this hypothesis yet"
    elif contradictions and children:
        state, reason = "REFINED", (
            "%d contradiction(s) recorded and %d refined child hypothesis record(s) exist"
            % (len(contradictions), len(children)))
    elif contradictions and supported:
        state, reason = "CONTRADICTED", (
            "%d contradiction(s) recorded while a simulation measured %+.2f%% improvement"
            % (len(contradictions), best))
    elif contradictions:
        state, reason = "REJECTED", (
            "%d high-severity contradiction(s) recorded and no measured improvement"
            % len(contradictions))
    elif supported and children:
        state, reason = "SURVIVED", (
            "simulation measured %+.2f%% improvement, no contradiction recorded, and the "
            "idea was carried into %d later hypothesis record(s)" % (best, len(children)))
    elif supported:
        state, reason = "SURVIVED", (
            "simulation measured %+.2f%% improvement and no contradiction is recorded" % best)
    else:
        state, reason = "UNDER INVESTIGATION", (
            "%d experiment record(s) stored but none carries a measurable improvement"
            % len(simulations))
    meta = HYPOTHESIS_RULES[state]
    return {"state": state, "state_rule": meta["rule"], "state_reason": reason,
            "state_meaning": meta["meaning"], "best_measured_improvement": best,
            "supported": supported, "contradicted": bool(contradictions),
            "refined": bool(children)}


def _experiment_summary(simulation: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, measured view of one stored simulation (used across views)."""
    observed = simulation.get("observed_results") or {}
    baseline = observed.get("baseline") or {}
    proposed = observed.get("proposed") or {}
    reproducibility = simulation.get("reproducibility") or {}
    parameters = simulation.get("parameters") or {}
    violations = [v for v in (simulation.get("constraints_violated") or [])
                  if str(v.get("violated")).lower() in ("true", "1")]
    undecidable = [v for v in (simulation.get("constraints_violated") or [])
                   if str(v.get("violated")).lower() not in ("true", "1", "false", "0")]
    return {
        "experiment_id": _stamp(simulation.get("experiment_id")),
        "hypothesis_id": _stamp(simulation.get("hypothesis_id")),
        "simulation_type": _stamp(simulation.get("simulation_type")),
        "status": _stamp(simulation.get("status")),
        "iteration": simulation.get("iteration", ""),
        "stored_at": _stamp(simulation.get("stored_at") or simulation.get("timestamp")),
        "adapter": {
            "name": _stamp((simulation.get("adapter") or {}).get("name")),
            "version": _stamp((simulation.get("adapter") or {}).get("version")),
            "module": _stamp((simulation.get("adapter") or {}).get("module")),
            "description": _stamp((simulation.get("adapter") or {}).get("description")),
        },
        "improvement_pct": _measured_improvement(simulation.get("metrics") or {}),
        "metrics": simulation.get("metrics") or {},
        "parameters": parameters,
        "compute": {
            "runs": parameters.get("num_runs"),
            "dimensionality": parameters.get("dimensionality"),
            "max_iterations": parameters.get("max_iterations"),
            "iteration_budget": (int(parameters.get("num_runs") or 0) *
                                 int(parameters.get("max_iterations") or 0)),
            "seed": reproducibility.get("seed"),
            "input_hash": _stamp(reproducibility.get("input_hash")),
            "deterministic": reproducibility.get("deterministic"),
        },
        "runtime": {
            "baseline_seconds": baseline.get("elapsed_seconds"),
            "proposed_seconds": proposed.get("elapsed_seconds"),
            "total_seconds": (float(baseline.get("elapsed_seconds") or 0.0) +
                              float(proposed.get("elapsed_seconds") or 0.0)),
        },
        "outcome": {
            "converged_baseline": baseline.get("converged_count"),
            "converged_proposed": proposed.get("converged_count"),
            "mean_iterations_baseline": baseline.get("mean_iterations"),
            "mean_iterations_proposed": proposed.get("mean_iterations"),
            "mean_loss_baseline": baseline.get("mean_final_loss"),
            "mean_loss_proposed": proposed.get("mean_final_loss"),
        },
        "constraints": {"violated": len(violations), "undecidable": len(undecidable),
                        "records": simulation.get("constraints_violated") or []},
        "errors": simulation.get("errors") or [],
        "notes": simulation.get("notes") or [],
        "has_curves": bool(observed.get("loss_curve_baseline") or observed.get("loss_curve_proposed")),
    }


def _contradiction_summary(failure: Dict[str, Any],
                           lessons: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    fid = _stamp(failure.get("failure_id"))
    related = [l for l in lessons.get(_stamp(failure.get("hypothesis_id")), [])
               if _stamp(l.get("failure_id")) == fid]
    return {
        "failure_id": fid,
        "experiment_id": _stamp(failure.get("experiment_id")),
        "hypothesis_id": _stamp(failure.get("hypothesis_id")),
        "failure_type": _stamp(failure.get("failure_type")),
        "severity": _stamp(failure.get("severity")),
        "observed_behavior": _stamp(failure.get("observed_behavior"))[:600],
        "failed_assumption": _stamp(failure.get("failed_assumption"))[:600],
        "lesson": _stamp(failure.get("lesson"))[:600],
        "recommended_changes": list(failure.get("recommended_changes") or []),
        "stored_at": _stamp(failure.get("stored_at") or failure.get("timestamp")),
        "lesson_ids": [_stamp(l.get("lesson_id")) for l in related],
    }


def _direction_map(run: Optional[Dict[str, Any]],
                   records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """direction_id -> display info. Run summaries win (they carry the status)."""
    out: Dict[str, Dict[str, Any]] = {}
    for direction in records.get("directions", []):
        identifier = _stamp(direction.get("direction_id"))
        if not identifier:
            continue
        status = direction.get("status") or {}
        if not isinstance(status, dict):
            status = {}
        out[identifier] = {
            "direction_id": identifier,
            "title": _stamp(direction.get("title")),
            "core_question": _stamp(direction.get("core_question")),
            "status": _stamp(status.get("status")),
            "dot": _stamp(status.get("dot")),
            "source": "stored direction record",
        }
    for summary in ((run or {}).get("directions") or []):
        if not isinstance(summary, dict):
            continue
        identifier = _stamp(summary.get("direction_id"))
        if not identifier:
            continue
        out[identifier] = {
            "direction_id": identifier,
            "title": _stamp(summary.get("title")),
            "core_question": _stamp(summary.get("core_question")),
            "status": _stamp(summary.get("status")),
            "dot": _stamp(summary.get("dot")),
            "status_label": _stamp(summary.get("status_label")),
            "source": "selected run evaluation",
        }
    return out


def _hypothesis_row(hypothesis: Dict[str, Any], index: Dict[str, Any],
                    directions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    hid = _stamp(hypothesis.get("hypothesis_id"))
    simulations = index["simulations"].get(hid, [])
    contradictions = index["failures"].get(hid, [])
    direction_id, link = _direction_id_of(hypothesis)
    row = {
        "hypothesis_id": hid,
        "hypothesis": _stamp(hypothesis.get("hypothesis")),
        "research_question": _stamp(hypothesis.get("research_question")),
        "direction_id": direction_id,
        "direction_link": link,
        "direction": directions.get(direction_id, {}),
        "iteration": hypothesis.get("iteration", ""),
        "domain": _stamp(hypothesis.get("domain")),
        "confidence_recorded": hypothesis.get("confidence"),
        "expected_outcome": hypothesis.get("expected_outcome") or {},
        "measurable_metrics": list(hypothesis.get("measurable_metrics") or []),
        "parameters": hypothesis.get("parameters") or {},
        "constraints": hypothesis.get("constraints") or [],
        "assumptions": list(hypothesis.get("assumptions") or []),
        "risks": list(hypothesis.get("risks") or []),
        "parent_hypothesis": _stamp(hypothesis.get("parent_hypothesis")),
        "based_on_lessons": list(hypothesis.get("based_on_lessons") or []),
        "based_on_knowledge": list(hypothesis.get("based_on_knowledge") or []),
        "contradictory_evidence": list(hypothesis.get("contradictory_evidence") or []),
        "supporting_evidence": list(hypothesis.get("supporting_evidence") or []),
        "generated_by": _stamp((hypothesis.get("provenance") or {}).get("generated_by")),
        "claim_type": _stamp(hypothesis.get("claim_type")),
        "stored_at": _stamp(hypothesis.get("stored_at") or hypothesis.get("timestamp")),
        "experiments": [_experiment_summary(s) for s in simulations],
        "contradictions": [_contradiction_summary(f, index["lessons"])
                           for f in contradictions],
        "refinements": sorted(index["children"].get(hid, [])),
        "lesson_ids": [_stamp(l.get("lesson_id")) for l in index["lessons"].get(hid, [])],
    }
    row.update(_hypothesis_state(row))
    row["what_would_change_it"] = _next_steps_for(row)
    row["sources"] = sorted({hid} | {_stamp(s.get("experiment_id")) for s in simulations} |
                            {_stamp(f.get("failure_id")) for f in contradictions})
    return row


def _next_steps_for(row: Dict[str, Any]) -> List[str]:
    """Concrete next actions that would move this hypothesis's recorded state."""
    out: List[str] = []
    if not row.get("experiments"):
        out.append("No experiment record refers to this hypothesis: it is untested, not refuted.")
    if row.get("contradictions") and not row.get("refined"):
        out.append("Refine the mechanism from the recorded contradiction instead of re-running "
                   "the same parameter block.")
    if row.get("contradictions"):
        out.append("Resolve the contradicting record experimentally before relying on the idea.")
    if not row.get("based_on_knowledge"):
        out.append("Link source-backed knowledge items so the claim inherits provenance.")
    if row.get("state") == "SURVIVED":
        out.append("Independent replication outside the model (by a human researcher) is still "
                   "required before stronger wording is justified.")
    return out or ["Nothing recorded stands in the way; the next honest step is independent "
                   "replication or a larger-scale run."]


def hypotheses_view(memory: Any, run: Optional[Dict[str, Any]], limit: int = 60,
                    state: str = "", direction_id: str = "",
                    records: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    """Hypothesis Laboratory: stored hypotheses with record-derived states."""
    records = records if records is not None else _records(memory)
    index = _hypothesis_index(records)
    directions = _direction_map(run, records)
    rows = [_hypothesis_row(h, index, directions)
            for h in _sorted_recent(records.get("hypotheses", []))]
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    filtered = rows
    if state:
        filtered = [row for row in filtered if row["state"] == state.upper()]
    if direction_id:
        filtered = [row for row in filtered if row["direction_id"] == direction_id]
    limit = max(1, min(int(limit or 60), 300))
    shown = filtered[:limit]
    return {
        "hypotheses": shown,
        "counts": counts,
        "states": [{"state": key, "rule": value["rule"], "meaning": value["meaning"],
                    "count": counts.get(key, 0)}
                   for key, value in HYPOTHESIS_RULES.items()],
        "total_stored": len(rows),
        "filtered_total": len(filtered),
        "truncated": _truncation(len(filtered), len(shown), "hypotheses"),
        "state_note": ("State is computed from stored records: experiment records, contradiction "
                       "records and the parent_hypothesis refinement chain. The recorded "
                       "'confidence' field is displayed separately and never decides the state."),
        "generated_at": _now(),
    }


def hypothesis_detail(memory: Any, hypothesis_id: str,
                      run: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One hypothesis as a research object: records, evidence, chain, next steps."""
    records = _records(memory)
    hypothesis = None
    for candidate in records.get("hypotheses", []):
        if _stamp(candidate.get("hypothesis_id")) == _stamp(hypothesis_id):
            hypothesis = candidate
            break
    if hypothesis is None:
        return {"error": "unknown hypothesis", "hypothesis_id": _stamp(hypothesis_id)}
    index = _hypothesis_index(records)
    directions = _direction_map(run, records)
    row = _hypothesis_row(hypothesis, index, directions)
    direction_id = row["direction_id"]
    direction = latest_stored(memory, "directions", direction_id) if direction_id else None
    evidence = (direction or {}).get("evidence") or {}
    row["direction_record"] = {
        "found": bool(direction),
        "direction_id": direction_id,
        "link_provenance": row["direction_link"] or "no link recorded",
        "title": _stamp((direction or {}).get("title")) or row["direction"].get("title", ""),
        "core_question": _stamp((direction or {}).get("core_question")),
        "status": (direction or {}).get("status") or {},
        "why_interesting": (direction or {}).get("why_interesting") or [],
        "mechanism": (direction or {}).get("mechanism") or {},
        "evidence_counts": evidence.get("counts") or {},
        "evidence_items": (evidence.get("items") or evidence.get("direct") or [])[:12],
        "gaps": (direction or {}).get("gaps") or [],
        "unknowns": (direction or {}).get("unknowns") or [],
        "failure_conditions": (direction or {}).get("failure_conditions") or [],
        "risks": (direction or {}).get("risks") or [],
        "dimensions": (direction or {}).get("dimensions") or {},
        "refinement": (direction or {}).get("refinement") or {},
        "first_experiment": (direction or {}).get("first_experiment") or {},
        "possibility_space_path": (direction or {}).get("possibility_space_path") or [],
        "source": ("live run direction summary" if row["direction"]
                   else ("latest stored DIR record" if direction else "no direction record")),
    }
    parent = row.get("parent_hypothesis")
    chain: List[Dict[str, Any]] = []
    seen = set()
    while parent and parent not in seen and len(chain) < 6:
        seen.add(parent)
        found = next((h for h in records.get("hypotheses", [])
                      if _stamp(h.get("hypothesis_id")) == parent), None)
        if not found:
            break
        chain.append({"hypothesis_id": parent,
                      "hypothesis": _stamp(found.get("hypothesis"))[:400],
                      "stored_at": _stamp(found.get("stored_at"))})
        parent = _stamp(found.get("parent_hypothesis"))
    row["parent_chain"] = chain
    row["refinement_children"] = [
        _hypothesis_row(h, index, directions)
        for h in records.get("hypotheses", [])
        if _stamp(h.get("hypothesis_id")) in set(row.get("refinements") or [])][:6]
    row["generated_at"] = _now()
    row["honesty"] = ("Every field above is a stored record value or a link derived from "
                      "stored ids. 'confidence_recorded' is the number the engine wrote on "
                      "the hypothesis; it is not a probability that the idea is true.")
    return row


# -------------------------------------------------------------------- experiments
def _hypothesis_direction_map(records: Dict[str, List[Dict[str, Any]]]
                              ) -> Dict[str, Tuple[str, str]]:
    return {_stamp(h.get("hypothesis_id")): _direction_id_of(h)
            for h in records.get("hypotheses", [])}


def experiments_view(memory: Any, run: Optional[Dict[str, Any]], limit: int = 40,
                     hypothesis_id: str = "", direction_id: str = "",
                     status: str = "",
                     records: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    """Experiment Monitor: measured simulation records plus honest aggregates."""
    records = records if records is not None else _records(memory)
    links = _hypothesis_direction_map(records)
    directions = _direction_map(run, records)
    rows: List[Dict[str, Any]] = []
    for simulation in _sorted_recent(records.get("simulations", [])):
        summary = _experiment_summary(simulation)
        hid = summary["hypothesis_id"]
        did, link = links.get(hid, ("", ""))
        summary["direction_id"] = did
        summary["direction_link"] = link
        summary["direction_title"] = directions.get(did, {}).get("title", "")
        rows.append(summary)
    filtered = rows
    if hypothesis_id:
        filtered = [row for row in filtered if row["hypothesis_id"] == hypothesis_id]
    if direction_id:
        filtered = [row for row in filtered if row["direction_id"] == direction_id]
    if status:
        filtered = [row for row in filtered if row["status"] == status]
    limit = max(1, min(int(limit or 40), 200))
    shown = filtered[:limit]
    improvements = [row["improvement_pct"] for row in filtered
                    if isinstance(row["improvement_pct"], (int, float))]
    status_counts: Dict[str, int] = {}
    for row in filtered:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    metric_names: List[str] = []
    for row in shown:
        for name in row["metrics"]:
            if name not in metric_names:
                metric_names.append(name)
    runtimes = [row["runtime"]["total_seconds"] for row in filtered
                if isinstance(row["runtime"]["total_seconds"], (int, float))]
    budgets = [row["compute"]["iteration_budget"] for row in filtered
               if isinstance(row["compute"]["iteration_budget"], int)]
    return {
        "experiments": shown,
        "total_stored": len(rows),
        "filtered_total": len(filtered),
        "status_counts": status_counts,
        "metric_names": metric_names,
        "aggregate": {
            "measured_improvements": len(improvements),
            "mean_improvement_pct": (sum(improvements) / len(improvements)
                                     if improvements else None),
            "best_improvement_pct": max(improvements) if improvements else None,
            "worst_improvement_pct": min(improvements) if improvements else None,
            "total_runtime_seconds": sum(runtimes) if runtimes else None,
            "total_iteration_budget": sum(budgets) if budgets else None,
            "adapters": sorted({row["adapter"]["name"] for row in filtered
                                if row["adapter"]["name"]}),
            "simulation_types": sorted({row["simulation_type"] for row in filtered
                                        if row["simulation_type"]}),
        },
        "truncated": _truncation(len(filtered), len(shown), "experiments"),
        "generated_at": _now(),
        "note": ("Every value on this screen is a stored measurement from a seeded run "
                 "(baseline vs proposal). 'Compute' is the recorded iteration budget "
                 "(runs x max_iterations); 'runtime' is the elapsed seconds the adapter "
                 "recorded. Nothing here is extrapolated."),
    }


def experiment_detail(memory: Any, experiment_id: str,
                      run: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One experiment with its stored curves, per-run results and records."""
    records = _records(memory)
    simulation = None
    for candidate in records.get("simulations", []):
        if _stamp(candidate.get("experiment_id")) == _stamp(experiment_id):
            simulation = candidate
            break
    if simulation is None:
        return {"error": "unknown experiment", "experiment_id": _stamp(experiment_id)}
    summary = _experiment_summary(simulation)
    observed = simulation.get("observed_results") or {}
    links = _hypothesis_direction_map(records)
    did, link = links.get(summary["hypothesis_id"], ("", ""))
    index = _hypothesis_index(records)
    relevant = [f for f in index["failures"].get(summary["hypothesis_id"], [])
                if _stamp(f.get("experiment_id")) == summary["experiment_id"]]
    summary["direction_id"] = did
    summary["direction_link"] = link
    summary["direction"] = _direction_map(run, records).get(did, {})
    summary["curves"] = {
        "loss_curve_baseline": _downsample(observed.get("loss_curve_baseline") or []),
        "loss_curve_proposed": _downsample(observed.get("loss_curve_proposed") or []),
        "samples_baseline": len(observed.get("loss_curve_baseline") or []),
        "samples_proposed": len(observed.get("loss_curve_proposed") or []),
        "per_run_baseline": observed.get("per_run_baseline") or [],
        "per_run_proposed": observed.get("per_run_proposed") or [],
    }
    summary["expected_results"] = simulation.get("expected_results") or {}
    summary["reproducibility"] = simulation.get("reproducibility") or {}
    summary["contradictions"] = [_contradiction_summary(f, index["lessons"])
                                 for f in relevant]
    summary["generated_at"] = _now()
    summary["honesty"] = ("Charts plot the stored loss curves sampled down to 44 points. "
                          "Constraints marked 'not verifiable from this simulation's metrics' "
                          "are reported as undecidable rather than as passed.")
    return summary


# ----------------------------------------------------------------------- thinking
#: Layers of the computational research field, in the order AARL works through
#: them. Each layer is fed by a different kind of stored record.
THINKING_LAYERS: Tuple[Dict[str, str], ...] = (
    {"key": "question", "label": "QUESTION",
     "meaning": "The problem AARL was given, exactly as recorded."},
    {"key": "concepts", "label": "CONCEPTS · DIRECTIONS",
     "meaning": "Structural facets mated with transferable mechanisms, stored as "
                "direction records."},
    {"key": "evidence", "label": "EVIDENCE",
     "meaning": "Source-backed knowledge items (claim_type 'sourced') and the papers "
                "they came from."},
    {"key": "relationships", "label": "RELATIONSHIPS",
     "meaning": "Typed links between records: tested_for, recorded_against, contradicts, "
                "refines, contributes_claim."},
    {"key": "hypotheses", "label": "HYPOTHESES",
     "meaning": "Stored hypothesis records with parameters, constraints and expectations."},
    {"key": "experiments", "label": "EXPERIMENTS",
     "meaning": "Seeded baseline-vs-proposal runs and their measurements."},
    {"key": "validation", "label": "VALIDATION",
     "meaning": "Recorded contradictions, lessons and the rule-derived direction status."},
)


def _topology_hash(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> str:
    """Short hash of the recorded topology, so the UI can animate real change."""
    import hashlib

    payload = "|".join(sorted(_stamp(node.get("id")) for node in nodes))
    payload += "#%d" % len(edges)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _direction_status_counts(run: Optional[Dict[str, Any]],
                             records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    summaries = [s for s in ((run or {}).get("directions") or []) if isinstance(s, dict)]
    if summaries:
        for summary in summaries:
            key = _stamp(summary.get("status")) or "UNKNOWN"
            out[key] = out.get(key, 0) + 1
        return out
    for direction in records.get("directions", []):
        status = direction.get("status")
        key = _stamp(status.get("status") if isinstance(status, dict) else status)
        key = key or "UNCLASSIFIED"
        out[key] = out.get(key, 0) + 1
    return out


def _thinking_field(memory: Any, knowledge: Any, run: Optional[Dict[str, Any]],
                    layer_limit: int,
                    records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Question, direction, evidence and paper layers of the research field."""
    scope = run_scope(memory, run, records)
    items = _knowledge_items(knowledge)
    layer_limit = max(4, min(int(layer_limit or 28), 80))
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    truncated: List[str] = []

    def add_nodes(rows: List[Dict[str, Any]], layer: str, ring: int, kind: str,
                  payloads: List[Dict[str, Any]]) -> None:
        for payload in payloads[:layer_limit]:
            payload.update({"layer": layer, "ring": ring, "kind": kind})
            nodes.append(payload)
        truncated.extend(_truncation(len(rows), min(len(payloads), layer_limit),
                                     "%s nodes" % layer))

    question = _stamp(scope.get("question")) or "No research problem recorded yet."
    nodes.append({"id": "Q", "kind": "question", "layer": "question", "ring": 0,
                  "label": question[:110], "sub": "the recorded problem",
                  "detail": "Research question", "weight": 1.0})

    summaries = scope.get("directions", [])
    if not summaries:
        summaries = []
        for direction in _sorted_recent(records.get("directions", []), layer_limit):
            status = direction.get("status")
            summaries.append({
                "direction_id": _stamp(direction.get("direction_id")),
                "title": _stamp(direction.get("title")),
                "core_question": _stamp(direction.get("core_question")),
                "status": _stamp(status.get("status") if isinstance(status, dict) else status),
            })
    add_nodes(summaries, "concepts", 1, "direction", [
        {"id": "D:%s" % s.get("direction_id"),
         "label": _stamp(s.get("title"))[:70] or _stamp(s.get("direction_id")),
         "sub": _stamp(s.get("status")) or "status not recorded",
         "detail": _stamp(s.get("core_question")), "weight": 0.8}
        for s in summaries])
    known = {node["id"] for node in nodes}
    for node in list(nodes):
        if node["layer"] == "concepts":
            edges.append({"source": "Q", "target": node["id"], "type": "facets_into"})

    sourced = _sorted_recent([item for item in items
                              if _stamp(item.get("claim_type")) == "sourced"], layer_limit)
    papers = {_stamp(p.get("paper_id")): _stamp(p.get("title"))
              for p in records.get("papers", [])}
    add_nodes(sourced, "evidence", 2, "claim", [
        {"id": "K:%s" % item.get("item_id"),
         "label": _stamp(item.get("statement"))[:80],
         "sub": _stamp(item.get("source")) or "sourced item",
         "detail": "section: %s" % _stamp(item.get("section")), "weight": 0.6}
        for item in sourced])
    for item in sourced:
        paper_id = _stamp((item.get("provenance") or {}).get("paper_id") or
                          item.get("source_paper"))
        if not paper_id or paper_id not in papers:
            continue
        paper_node_id = "P:%s" % paper_id
        if paper_node_id not in known:
            nodes.append({"id": paper_node_id, "kind": "paper", "layer": "evidence",
                          "ring": 2, "label": papers[paper_id][:70], "weight": 0.5,
                          "sub": "paper record", "detail": "Source document in the library"})
            known.add(paper_node_id)
        edges.append({"source": "K:%s" % item.get("item_id"), "target": paper_node_id,
                      "type": "contributes_claim"})
    return {"nodes": nodes, "edges": edges, "truncated": truncated, "known": known,
            "scope": scope, "summaries": summaries, "records": records,
            "layer_limit": layer_limit, "papers": papers}


def thinking(memory: Any, knowledge: Any, run: Optional[Dict[str, Any]],
             layer_limit: int = 28,
             records: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    """The computational research field: stored records arranged as a research loop."""
    records = records if records is not None else _records(memory)
    limit = max(4, min(int(layer_limit or 28), 80))
    field = _thinking_field(memory, knowledge, run, limit, records)
    nodes: List[Dict[str, Any]] = field["nodes"]
    edges: List[Dict[str, Any]] = field["edges"]
    truncated: List[str] = field["truncated"]
    known: set = field["known"]
    scope = field["scope"]
    node_ids = lambda: {node["id"] for node in nodes}  # noqa: E731 - local shorthand

    hypotheses = scope.get("hypotheses", [])[:limit]
    nodes.extend([{"id": "H:%s" % h.get("hypothesis_id"), "kind": "hypothesis",
                   "layer": "hypotheses", "ring": 4,
                   "label": _stamp(h.get("hypothesis"))[:80],
                   "sub": _stamp(h.get("domain")) or "hypothesis record",
                   "detail": "iteration %s" % h.get("iteration", ""), "weight": 0.7}
                  for h in hypotheses])
    truncated.extend(_truncation(len(scope.get("hypotheses", [])), len(hypotheses),
                                 "hypotheses nodes"))
    for hypothesis in hypotheses:
        direction_id, _link = _direction_id_of(hypothesis)
        hid = "H:%s" % hypothesis.get("hypothesis_id")
        if direction_id and ("D:%s" % direction_id) in node_ids():
            edges.append({"source": "D:%s" % direction_id, "target": hid,
                          "type": "tested_for"})
        parent = _stamp(hypothesis.get("parent_hypothesis"))
        if parent and ("H:%s" % parent) in node_ids():
            edges.append({"source": hid, "target": "H:%s" % parent, "type": "refines"})

    simulations = scope.get("simulations", [])[:limit]
    nodes.extend([{"id": "E:%s" % s.get("experiment_id"), "kind": "experiment",
                   "layer": "experiments", "ring": 5,
                   "label": _stamp(s.get("experiment_id")),
                   "sub": _stamp(s.get("status")) or "status not recorded",
                   "detail": _stamp(s.get("simulation_type")), "weight": 0.8}
                  for s in simulations])
    truncated.extend(_truncation(len(scope.get("simulations", [])), len(simulations),
                                 "experiment nodes"))
    for simulation in simulations:
        target = "H:%s" % simulation.get("hypothesis_id")
        if target in node_ids():
            edges.append({"source": "E:%s" % simulation.get("experiment_id"),
                          "target": target, "type": "recorded_against"})

    validation: List[Dict[str, Any]] = []
    for failure in scope.get("failures", [])[:limit]:
        validation.append({"id": "F:%s" % failure.get("failure_id"),
                           "label": _stamp(failure.get("failure_type")) or "contradiction",
                           "sub": "severity %s" % _stamp(failure.get("severity")),
                           "detail": _stamp(failure.get("lesson"))[:180], "weight": 0.9,
                           "target": "E:%s" % failure.get("experiment_id"),
                           "type": "contradicts"})
    for lesson in scope.get("lessons", [])[:limit]:
        validation.append({"id": "L:%s" % lesson.get("lesson_id"),
                           "label": _stamp(lesson.get("lesson"))[:70] or "stored lesson",
                           "sub": "lesson",
                           "detail": _stamp(lesson.get("recommended_changes")), "weight": 0.5,
                           "target": "F:%s" % lesson.get("failure_id"),
                           "type": "learned_from"})
    nodes.extend([{"id": row["id"], "kind": "validation", "layer": "validation", "ring": 6,
                   "label": row["label"], "sub": row["sub"], "detail": row["detail"],
                   "weight": row["weight"]} for row in validation[:limit]])
    truncated.extend(_truncation(len(validation), min(len(validation), limit),
                                 "validation nodes"))
    for row in validation[:limit]:
        if row["id"] in node_ids() and row["target"] in node_ids():
            edges.append({"source": row["id"], "target": row["target"], "type": row["type"]})

    recent = 0
    for collection in ("hypotheses", "simulations", "failures", "lessons", "directions"):
        for record in records.get(collection, []):
            age = _age_seconds(record.get("stored_at") or record.get("timestamp"))
            if age is not None and age <= 3600:
                recent += 1
    counts = {layer["key"]: len([n for n in nodes if n["layer"] == layer["key"]])
              for layer in THINKING_LAYERS}
    counts["relationships"] = len(edges)
    return {
        "run_id": _stamp(scope.get("run_id")),
        "question": _stamp(scope.get("question")) or "No research problem recorded yet.",
        "nodes": nodes,
        "edges": edges,
        "layers": [dict(layer, count=counts.get(layer["key"], 0)) for layer in THINKING_LAYERS],
        "counts": counts,
        "records_last_hour": recent,
        "topology_hash": _topology_hash(nodes, edges),
        "status_counts": _direction_status_counts(run, records),
        "truncated": sorted(set(truncated)),
        "generated_at": _now(),
        "note": ("Every node is a stored record and every link is an id AARL wrote (or a "
                 "declared text reference). When the layer is empty the field says so "
                 "instead of drawing an invented node."),
    }


# -------------------------------------------------------------------------- graph
#: Node budget per record kind. The graph is a window over the store, never a full
#: dump: what was left out is reported in ``truncated``.
GRAPH_NODE_CAPS: Dict[str, int] = {
    "claim": 240, "paper": 80, "direction": 100,
    "hypothesis": 200, "experiment": 200, "failure": 160,
}


def _graph_build(memory: Any, knowledge: Any, run: Optional[Dict[str, Any]],
                 records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Claim and paper layer of the record graph, plus the node/edge builders."""
    items = _knowledge_items(knowledge)
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    truncated: List[str] = []

    def add_nodes(rows: List[Dict[str, Any]], kind: str,
                  make: Callable[[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
        cap = GRAPH_NODE_CAPS.get(kind, 100)
        kept: List[Dict[str, Any]] = []
        for row in rows[:cap]:
            node = make(row)
            nodes[node["id"]] = node
            kept.append(row)
        truncated.extend(_truncation(len(rows), len(kept), "%s nodes" % kind))
        return rows[:cap]

    def edge(source: str, target: str, etype: str, note: str = "") -> None:
        if source in nodes and target in nodes:
            edges.append({"source": source, "target": target, "type": etype, "note": note})

    claim_rows = _sorted_recent(items)
    add_nodes(claim_rows, "claim", lambda item: {
        "id": "claim:%s" % item.get("item_id"), "kind": "claim",
        "label": _stamp(item.get("statement"))[:90],
        "section": _stamp(item.get("section")),
        "claim_type": _stamp(item.get("claim_type")),
        "confidence_recorded": item.get("confidence"),
        "source": _stamp(item.get("source")),
        "source_paper": _stamp(item.get("source_paper")),
        "at": _stamp(item.get("timestamp")),
        "note": ("'sourced' claims come from a supplied document; 'inferred' and "
                 "'generated_hypothesis' claims are AARL's own reasoning."),
    })
    add_nodes(_sorted_recent(records.get("papers", [])), "paper", lambda paper: {
        "id": "paper:%s" % paper.get("paper_id"), "kind": "paper",
        "label": _stamp(paper.get("title"))[:90] or _stamp(paper.get("paper_id")),
        "at": _stamp(paper.get("stored_at")),
        "source_url": _stamp(paper.get("source_url")),
        "note": "Source document record (metadata and extracted claims only).",
    })
    return {"nodes": nodes, "edges": edges, "truncated": truncated,
            "items": items, "records": records, "add_nodes": add_nodes, "edge": edge}


def _graph_records(memory: Any, knowledge: Any, run: Optional[Dict[str, Any]],
                   records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """All node layers of the record graph, built in one pass per collection."""
    build = _graph_build(memory, knowledge, run, records)
    nodes: Dict[str, Dict[str, Any]] = build["nodes"]
    edges: List[Dict[str, Any]] = build["edges"]
    truncated: List[str] = build["truncated"]
    items: List[Dict[str, Any]] = build["items"]
    add_nodes = build["add_nodes"]
    edge = build["edge"]

    direction_rows = _sorted_recent(records.get("directions", []))
    summaries = [s for s in ((run or {}).get("directions") or []) if isinstance(s, dict)]
    if summaries:
        by_id = {_stamp(d.get("direction_id")): d for d in direction_rows}
        direction_rows = [by_id.get(_stamp(s.get("direction_id")), s) for s in summaries]
    add_nodes(direction_rows, "direction", lambda direction: {
        "id": "direction:%s" % direction.get("direction_id"), "kind": "direction",
        "label": _stamp(direction.get("title"))[:90] or _stamp(direction.get("direction_id")),
        "status": _stamp((direction.get("status") or {}).get("status"))
                  if isinstance(direction.get("status"), dict)
                  else _stamp(direction.get("status")),
        "core_question": _stamp(direction.get("core_question"))[:200],
        "at": _stamp(direction.get("stored_at")),
        "note": "Discovered research direction with a proposed (conceptual) mechanism.",
    })
    hypothesis_rows = add_nodes(_sorted_recent(records.get("hypotheses", [])), "hypothesis",
                                lambda hypothesis: {
        "id": "hypothesis:%s" % hypothesis.get("hypothesis_id"), "kind": "hypothesis",
        "label": _stamp(hypothesis.get("hypothesis"))[:90],
        "domain": _stamp(hypothesis.get("domain")),
        "confidence_recorded": hypothesis.get("confidence"),
        "at": _stamp(hypothesis.get("stored_at")),
        "note": "Stored hypothesis (claim_type 'generated_hypothesis').",
    })
    experiment_rows = add_nodes(_sorted_recent(records.get("simulations", [])), "experiment",
                                lambda simulation: {
        "id": "experiment:%s" % simulation.get("experiment_id"), "kind": "experiment",
        "label": _stamp(simulation.get("experiment_id")),
        "status": _stamp(simulation.get("status")),
        "simulation_type": _stamp(simulation.get("simulation_type")),
        "improvement_pct": _measured_improvement(simulation.get("metrics") or {}),
        "at": _stamp(simulation.get("stored_at")),
        "note": "Measured baseline-vs-proposal run.",
    })
    failure_rows = add_nodes(_sorted_recent(records.get("failures", [])), "failure",
                             lambda failure: {
        "id": "failure:%s" % failure.get("failure_id"), "kind": "failure",
        "label": _stamp(failure.get("failure_type")) or _stamp(failure.get("failure_id")),
        "severity": _stamp(failure.get("severity")),
        "at": _stamp(failure.get("stored_at")),
        "note": "Recorded contradiction / failure — kept as research memory, never deleted.",
    })
    return {"nodes": nodes, "edges": edges, "truncated": truncated, "items": items,
            "records": records, "hypothesis_rows": hypothesis_rows,
            "experiment_rows": experiment_rows, "failure_rows": failure_rows,
            "edge_fn": edge}


def graph(memory: Any, knowledge: Any, run: Optional[Dict[str, Any]] = None,
          records: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    """Typed record graph — a fast variant of ``/api/graph``.

    Every node is a stored record and every edge is an id AARL wrote. Node size is
    the record's link count, never an evidence claim.
    """
    records = records if records is not None else _records(memory)
    built = _graph_records(memory, knowledge, run, records)
    nodes: Dict[str, Dict[str, Any]] = built["nodes"]
    edges: List[Dict[str, Any]] = built["edges"]
    edge = built["edge_fn"]

    for hypothesis in built["hypothesis_rows"]:
        hid = "hypothesis:%s" % hypothesis.get("hypothesis_id")
        direction_id, _link = _direction_id_of(hypothesis)
        if direction_id:
            edge("direction:%s" % direction_id, hid, "tested_for",
                 "the hypothesis names this direction")
        parent = _stamp(hypothesis.get("parent_hypothesis"))
        if parent:
            edge(hid, "hypothesis:%s" % parent, "refines",
                 "parent_hypothesis recorded on the record")
        for knowledge_id in (hypothesis.get("based_on_knowledge") or [])[:6]:
            edge(hid, "claim:%s" % knowledge_id, "based_on")
    for simulation in built["experiment_rows"]:
        edge("experiment:%s" % simulation.get("experiment_id"),
             "hypothesis:%s" % simulation.get("hypothesis_id"), "recorded_against")
    for failure in built["failure_rows"]:
        edge("failure:%s" % failure.get("failure_id"),
             "experiment:%s" % failure.get("experiment_id"), "contradicts")
    for item in built["items"]:
        paper_id = _stamp((item.get("provenance") or {}).get("paper_id") or
                          item.get("source_paper"))
        if paper_id:
            edge("paper:%s" % paper_id, "claim:%s" % item.get("item_id"),
                 "contributes_claim")

    degree: Dict[str, int] = {nid: 0 for nid in nodes}
    for item in edges:
        degree[item["source"]] = degree.get(item["source"], 0) + 1
        degree[item["target"]] = degree.get(item["target"], 0) + 1
    for nid, node in nodes.items():
        node["degree"] = degree.get(nid, 0)
    counts: Dict[str, int] = {}
    for node in nodes.values():
        counts[node["kind"]] = counts.get(node["kind"], 0) + 1
    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "counts": counts,
        "edge_types": sorted({item["type"] for item in edges}),
        "legend": {
            "claim": "knowledge item (provenance-carrying)",
            "paper": "research paper / source document",
            "direction": "discovered research direction",
            "hypothesis": "stored hypothesis",
            "experiment": "measured simulation record",
            "failure": "recorded contradiction / failure",
        },
        "truncated": sorted(set(built["truncated"])),
        "generated_at": _now(),
        "note": ("Edges describe how records are linked in the store. A high-degree node "
                 "is a well-connected record, not a scientific discovery."),
    }


# ------------------------------------------------------------------------- report
def _dir_view(direction: Dict[str, Any]) -> Dict[str, Any]:
    status = direction.get("status") if isinstance(direction.get("status"), dict) else {}
    mechanism = direction.get("mechanism") or {}
    return {
        "direction_id": _stamp(direction.get("direction_id")),
        "title": _stamp(direction.get("title")),
        "core_question": _stamp(direction.get("core_question")),
        "status": _stamp((status or {}).get("status")),
        "status_label": _stamp((status or {}).get("label")),
        "status_meaning": _stamp((status or {}).get("meaning")),
        "status_reasons": list((status or {}).get("reasons") or []),
        "rule_id": _stamp((status or {}).get("rule_id")),
        "what_would_change_it": list((status or {}).get("what_would_change_it") or []),
        "inspiration": _stamp((direction.get("inspiration") or {}).get("display")),
        "why_interesting": list(direction.get("why_interesting") or []),
        "mechanism_steps": list(mechanism.get("steps") or []),
        "mechanism_status": _stamp(mechanism.get("implementation_status")),
        "evidence_counts": (direction.get("evidence") or {}).get("counts") or {},
        "speculative_aspects": list(direction.get("speculative_aspects") or []),
        "gaps": list(direction.get("gaps") or []),
        "unknowns": list(direction.get("unknowns") or []),
        "failure_conditions": list(direction.get("failure_conditions") or []),
        "risks": list(direction.get("risks") or []),
        "first_experiment": direction.get("first_experiment") or {},
        "possibility_space_path": list(direction.get("possibility_space_path") or []),
        "already_failed": bool(direction.get("already_failed")),
    }


def _section(number: int, key: str, title: str, paragraphs: List[str],
             items: List[Dict[str, Any]], sources: List[str], empty_note: str,
             counts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"n": number, "key": key, "title": title, "paragraphs": paragraphs,
            "items": items, "sources": sources, "counts": counts or {},
            "empty": not (paragraphs or items), "empty_note": empty_note}


def _report_core(scope: Dict[str, Any], run: Optional[Dict[str, Any]],
                 records: Dict[str, List[Dict[str, Any]]],
                 sourced: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sections 1-3: research question, problem decomposition, literature."""
    stats = (run or {}).get("stats") or {}
    rejected = (run or {}).get("rejected") or []
    papers = records.get("papers", [])
    question = _section(
        1, "question", "RESEARCH QUESTION",
        [scope.get("question") or "No research problem recorded yet."],
        [{"label": "Run", "value": _stamp(scope.get("run_id")) or "—"},
         {"label": "Status", "value": _stamp(scope.get("status")) or "idle"},
         {"label": "Opened", "value": _stamp((run or {}).get("created_at")) or "—"},
         {"label": "Closed", "value": _stamp((run or {}).get("completed_at")) or "in progress"},
         {"label": "Primary domain", "value": _stamp((run or {}).get("primary_domain")) or "—"},
         {"label": "Domains considered",
          "value": ", ".join(_stamp(d) for d in ((run or {}).get("domains") or [])) or "—"}],
        [_stamp(scope.get("run_id"))], "No exploration run is recorded yet.")
    decomposition = _section(
        2, "decomposition", "PROBLEM DECOMPOSITION",
        [_stamp(stats.get("how_directions_were_built")) or
         "The decomposition procedure is not recorded for this run."],
        [{"label": "Subject", "value": _stamp(stats.get("subject_label")) or "—"},
         {"label": "Facets explored", "value": stats.get("facets_explored", 0)},
         {"label": "Concepts considered", "value": stats.get("concepts_considered", 0)},
         {"label": "Directions built",
          "value": stats.get("directions", len(scope.get("directions", [])))},
         {"label": "Candidates considered and not taken",
          "value": stats.get("candidates_rejected", len(rejected))},
         {"label": "Branches already failed in memory",
          "value": stats.get("already_failed_in_memory", 0)}],
        [_stamp(scope.get("run_id"))], "No possibility space is recorded for this run yet.")
    literature = _section(
        3, "literature", "LITERATURE",
        ["Only claims extracted from documents the researcher supplied are stored as "
         "'sourced'. Generated hypotheses are never counted as literature evidence."],
        [{"label": "Papers in the library", "value": len(papers)},
         {"label": "Sourced knowledge items", "value": len(sourced)},
         {"label": "Items overlapping this problem",
          "value": len((run or {}).get("literature_overlap") or [])}] +
        [{"label": _stamp(paper.get("title"))[:90] or _stamp(paper.get("paper_id")),
          "value": " · ".join(part for part in (
              _stamp(paper.get("year")),
              "%s claim(s)" % ((paper.get("counts") or {}).get("claims", 0)),
              "contributing knowledge" if paper.get("in_knowledge") else "not contributing",
              _stamp(paper.get("source_url"))) if part),
          "refs": [_stamp(paper.get("paper_id"))]} for paper in papers[:12]],
        [_stamp(paper.get("paper_id")) for paper in papers[:12]],
        "No papers have been added to the library yet.")
    return [question, decomposition, literature]


def _report_evidence(scope: Dict[str, Any], run: Optional[Dict[str, Any]],
                     index: Dict[str, Any], sourced: List[Dict[str, Any]],
                     graph_counts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sections 4-6: knowledge graph, hypotheses, experiments."""
    counts = graph_counts.get("counts", {})
    graph_section = _section(
        4, "graph", "KNOWLEDGE GRAPH",
        [_stamp(graph_counts.get("note"))],
        [{"label": "Claims", "value": counts.get("claim", 0)},
         {"label": "Papers", "value": counts.get("paper", 0)},
         {"label": "Directions", "value": counts.get("direction", 0)},
         {"label": "Hypotheses", "value": counts.get("hypothesis", 0)},
         {"label": "Experiments", "value": counts.get("experiment", 0)},
         {"label": "Contradictions", "value": counts.get("failure", 0)},
         {"label": "Typed edges", "value": len(graph_counts.get("edges", []))},
         {"label": "Window", "value": "; ".join(graph_counts.get("truncated", [])) or "whole store"}],
        [], "No records exist yet, so the graph is empty.")
    graph_section["items"].extend([
        {"label": _stamp(item.get("statement"))[:120],
         "value": "section %s · source %s" % (_stamp(item.get("section")),
                                             _stamp(item.get("source"))),
         "refs": [_stamp(item.get("item_id"))]} for item in sourced[:10]])

    hypothesis_rows = [_hypothesis_row(h, index, {}) for h in scope.get("hypotheses", [])]
    state_counts: Dict[str, int] = {}
    for row in hypothesis_rows:
        state_counts[row["state"]] = state_counts.get(row["state"], 0) + 1
    hypotheses_section = _section(
        5, "hypotheses", "HYPOTHESES",
        ["Each hypothesis is stored with its parameter block, constraints and expected "
         "outcome. Its laboratory state is derived from experiment and contradiction "
         "records, never from the wording of the claim."],
        [{"label": row["hypothesis_id"],
          "value": "%s · %s" % (row["state"], _stamp(row["hypothesis"])[:160]),
          "refs": [row["hypothesis_id"]]} for row in hypothesis_rows[:25]],
        [row["hypothesis_id"] for row in hypothesis_rows[:25]],
        "No hypothesis record is attributed to this run yet.", counts=state_counts)

    experiment_rows: List[Dict[str, Any]] = []
    for simulation in scope.get("simulations", [])[:25]:
        summary = _experiment_summary(simulation)
        metrics = ", ".join(
            "%s %s → %s (%+.2f%%)" % (name, _stamp(value.get("baseline_value")),
                                      _stamp(value.get("observed_value")),
                                      float(value.get("improvement_pct") or 0.0))
            for name, value in (summary["metrics"] or {}).items()
            if isinstance(value, dict) and value.get("improvement_pct") is not None)
        experiment_rows.append({
            "label": summary["experiment_id"],
            "value": "%s · %s · %s" % (summary["hypothesis_id"], _stamp(summary["status"]),
                                       metrics or "no metric deltas recorded"),
            "refs": [summary["experiment_id"], summary["hypothesis_id"]]})
    experiments_section = _section(
        6, "experiments", "EXPERIMENTS",
        ["Every experiment listed here was executed as a seeded baseline-vs-proposal run "
         "and stored as a Simulation JSON record with its metrics and loss curves."],
        experiment_rows, [row["refs"][0] for row in experiment_rows],
        "No experiment is attributed to this run yet.")
    return [graph_section, hypotheses_section, experiments_section]


def _report_results(scope: Dict[str, Any]) -> Dict[str, Any]:
    """Section 7: the measured results table."""
    rows: List[Dict[str, Any]] = []
    improvements: List[float] = []
    for simulation in scope.get("simulations", [])[:30]:
        summary = _experiment_summary(simulation)
        for name, value in (summary["metrics"] or {}).items():
            if not isinstance(value, dict):
                continue
            if value.get("improvement_pct") is None:
                continue
            improvements.append(float(value["improvement_pct"]))
            rows.append({
                "label": "%s · %s" % (summary["experiment_id"], name),
                "value": "%s → %s %s  (%+.2f%%, %s)" % (
                    _stamp(value.get("baseline_value")), _stamp(value.get("observed_value")),
                    _stamp(value.get("unit")), float(value["improvement_pct"]),
                    "improvement" if value.get("is_improvement") else "no improvement"),
                "refs": [summary["experiment_id"]],
            })
    best = max(improvements) if improvements else None
    paragraphs = ["Baseline and proposed values are engine measurements from the stored "
                  "simulation records; the percentage is the engine's own improvement "
                  "calculation for that metric."]
    if improvements:
        paragraphs.append(
            "%d metric delta(s) for this run, best %+.2f%%, mean %+.2f%%." % (
                len(improvements), best, sum(improvements) / len(improvements)))
    return _section(7, "results", "RESULTS", paragraphs, rows,
                    [row["refs"][0] for row in rows],
                    "No measured metric delta is recorded for this run yet.",
                    counts={"metric_deltas": len(rows),
                            "best_improvement_pct": best})


def _report_surviving(scope: Dict[str, Any], index: Dict[str, Any]) -> Dict[str, Any]:
    """Section 8: surviving ideas (support + no contradiction)."""
    survivors = [s for s in scope.get("directions", [])
                 if _stamp(s.get("status")) in ("EVIDENCE_BACKED", "PROMISING")]
    rows = [{"label": "%s · %s" % (_stamp(s.get("direction_id")), _stamp(s.get("title"))),
             "value": "%s — %s" % (_stamp(s.get("status")), _stamp(s.get("status_meaning"))[:220]),
             "refs": [_stamp(s.get("direction_id"))]} for s in survivors]
    hypothesis_rows = [_hypothesis_row(h, index, {}) for h in scope.get("hypotheses", [])]
    breathing = [row for row in hypothesis_rows if row["state"] in ("SURVIVED", "SUPPORTED")]
    rows.extend([{"label": row["hypothesis_id"],
                  "value": "%s · %s" % (row["state"], _stamp(row["hypothesis"])[:180]),
                  "refs": [row["hypothesis_id"]]} for row in breathing[:20]])
    return _section(
        8, "surviving", "SURVIVING IDEAS",
        ["'Surviving' means: a stored measurement supports the idea and no contradiction "
         "is recorded. It is not a statement that the idea is true, and it never replaces "
         "independent replication outside the model."],
        rows, [row["refs"][0] for row in rows],
        "Nothing has survived validation for this run yet.",
        counts={"directions": len(survivors), "hypotheses": len(breathing)})


def _report_rejected(scope: Dict[str, Any], index: Dict[str, Any]) -> Dict[str, Any]:
    """Section 9: rejected ideas and the lessons kept from them."""
    rows: List[Dict[str, Any]] = []
    failed_dirs = [s for s in scope.get("directions", [])
                   if _stamp(s.get("status")) == "FAILED_WEAK"]
    rows.extend([{"label": "%s · %s" % (_stamp(s.get("direction_id")), _stamp(s.get("title"))),
                  "value": _stamp(s.get("status_meaning"))[:240] or "recorded as failed / weak",
                  "refs": [_stamp(s.get("direction_id"))]} for s in failed_dirs])
    rejected = [row for row in
                (_hypothesis_row(h, index, {}) for h in scope.get("hypotheses", []))
                if row["state"] in ("REJECTED", "CONTRADICTED", "REFINED")]
    rows.extend([{"label": row["hypothesis_id"],
                  "value": "%s · %s" % (row["state"], row["state_reason"][:200]),
                  "refs": [row["hypothesis_id"]]} for row in rejected[:20]])
    rows.extend([{"label": _stamp(failure.get("failure_id")),
                  "value": "%s · %s" % (_stamp(failure.get("failure_type")),
                                        _stamp(failure.get("lesson"))[:200]),
                  "refs": [_stamp(failure.get("failure_id")),
                           _stamp(failure.get("experiment_id"))]}
                 for failure in scope.get("failures", [])[:20]])
    return _section(
        9, "rejected", "REJECTED IDEAS",
        ["Rejected ideas are not deleted. Every failure stays in the research memory with "
         "its failed assumption, its lesson and the refined direction grown from it, so the "
         "same mechanism is not proposed again unchanged."],
        rows, [row["refs"][0] for row in rows],
        "No rejected idea or recorded contradiction exists for this run yet.",
        counts={"failed_directions": len(failed_dirs), "contradictions": len(scope.get("failures", []))})


def _report_limits_next(memory: Any, scope: Dict[str, Any],
                        items: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Sections 10-11: limitations and next research directions."""
    limitation_rows: List[Dict[str, Any]] = []
    next_rows: List[Dict[str, Any]] = []
    for summary in scope.get("directions", [])[:20]:
        direction_id = _stamp(summary.get("direction_id"))
        stored = latest_stored(memory, "directions", direction_id)
        if not stored:
            continue
        view = _dir_view(stored)
        for gap in view["gaps"][:3]:
            text = _stamp(gap.get("question") or gap.get("gap")) if isinstance(gap, dict) \
                else _stamp(gap)
            if text:
                limitation_rows.append({"label": direction_id, "value": text[:220],
                                        "refs": [direction_id]})
        for unknown in view["unknowns"][:2]:
            limitation_rows.append({"label": direction_id, "value": _stamp(unknown)[:220],
                                    "refs": [direction_id]})
        for aspect in view["speculative_aspects"][:2]:
            limitation_rows.append({"label": "%s · speculative" % direction_id,
                                    "value": _stamp(aspect)[:220], "refs": [direction_id]})
        for step in view["mechanism_steps"]:
            if _stamp(step.get("evidence_state")) == "unsupported":
                limitation_rows.append({
                    "label": "%s · mechanism stage %s" % (direction_id, step.get("index")),
                    "value": "no stored evidence addresses this stage: %s" %
                             _stamp(step.get("text"))[:180],
                    "refs": [direction_id]})
        for action in view["what_would_change_it"][:2]:
            next_rows.append({"label": direction_id, "value": _stamp(action)[:240],
                              "refs": [direction_id]})
        if view["first_experiment"].get("design"):
            next_rows.append({"label": "%s · suggested first experiment" % direction_id,
                              "value": _stamp(view["first_experiment"].get("design"))[:240],
                              "refs": [direction_id]})
    stored_limits = [item for item in items if _stamp(item.get("section")) == "limitations"]
    limitation_rows.extend([{"label": _stamp(item.get("item_id")),
                             "value": _stamp(item.get("statement"))[:240],
                             "refs": [_stamp(item.get("item_id"))]}
                            for item in stored_limits[:10]])
    limitations = _section(
        10, "limitations", "LIMITATIONS",
        ["AARL's measurements come from simulation models, not from the physical world. "
         "Counter-claims on stored records that could not be tied to a direction are "
         "listed separately in the knowledge store.",
         "Every gap below was stored by the engines; open gaps are research targets, not "
         "hidden assumptions."],
        limitation_rows, [row["refs"][0] for row in limitation_rows],
        "No limitation, gap or unknown is recorded for this run yet.",
        counts={"items": len(limitation_rows), "stored_limitation_items": len(stored_limits)})
    next_steps = _section(
        11, "next", "NEXT RESEARCH DIRECTIONS",
        ["Each line is the action the engines recorded as necessary to move a direction's "
         "status: new evidence, a resolvable contradiction, or an executable model."],
        next_rows, [row["refs"][0] for row in next_rows],
        "No next step is recorded for this run yet.")
    return limitations, next_steps


def report(memory: Any, knowledge: Any, run: Optional[Dict[str, Any]],
           records: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    """Research report assembled only from stored records."""
    records = records if records is not None else _records(memory)
    scope = run_scope(memory, run, records)
    index = _hypothesis_index(records)
    items = _knowledge_items(knowledge)
    sourced = [item for item in items if _stamp(item.get("claim_type")) == "sourced"]
    graph_payload = graph(memory, knowledge, run, records)
    sections: List[Dict[str, Any]] = []
    sections.extend(_report_core(scope, run, records, sourced))
    sections.extend(_report_evidence(scope, run, index, sourced, graph_payload))
    sections.append(_report_results(scope))
    sections.append(_report_surviving(scope, index))
    sections.append(_report_rejected(scope, index))
    limitations, next_steps = _report_limits_next(memory, scope, items)
    sections.append(limitations)
    sections.append(next_steps)
    recorded = sum(1 for section in sections if not section["empty"])
    return {
        "run_id": _stamp(scope.get("run_id")),
        "question": _stamp(scope.get("question")),
        "run_status": _stamp(scope.get("status")),
        "sections": sections,
        "sections_with_records": recorded,
        "sections_total": len(sections),
        "scope_counts": scope.get("counts", {}),
        "knowledge_summary": knowledge.summary(),
        "link_provenance": scope.get("link_provenance", []),
        "generated_at": _now(),
        "honesty": ("This report is assembled from stored records. Where a section has no "
                    "recorded content it says so instead of filling the gap with prose."),
    }


# --------------------------------------------------------------------------- runs
def runs(memory: Any, live_states: Sequence[Optional[Dict[str, Any]]] = (),
         limit: int = 40) -> Dict[str, Any]:
    """Slim exploration-run list (the heavy possibility space stays out of it)."""
    signature = _memory_signature(memory, ("explorations",))

    def build() -> Dict[str, Dict[str, Any]]:
        revision = _revision_index(memory, "explorations")
        states: Dict[str, Dict[str, Any]] = {}
        for record in memory.load_collection("explorations"):
            record_id = _stamp(record.get("record_id"))
            if not record_id:
                continue
            state = record
            for candidate in range(revision.get(record_id, 0), 0, -1):
                body = memory.get("explorations", "%s.rev%d" % (record_id, candidate))
                if isinstance(body, dict):
                    state = body
                    break
            states[record_id] = state
        return states

    states = dict(_CACHE.get("runs:states", signature, build))
    for state in live_states:
        if isinstance(state, dict) and _stamp(state.get("run_id")):
            states[_stamp(state.get("run_id"))] = state

    rows: List[Dict[str, Any]] = []
    for run_id, state in states.items():
        stages = state.get("stages") or []
        done = len([s for s in stages if _stamp(s.get("state")) == "done"])
        running = len([s for s in stages if _stamp(s.get("state")) == "running"])
        stats = state.get("stats") or {}
        counts = state.get("counts") or {}
        run_state = live_status(state)
        rows.append({
            "run_id": run_id,
            "problem": _stamp(state.get("problem")),
            "subject": _stamp(state.get("subject") or stats.get("subject_label")),
            "status": run_state["status"],
            "recorded_status": run_state["recorded_status"],
            "stalled": run_state["status"] == "stalled",
            "run_state": run_state,
            "primary_domain": _stamp(state.get("primary_domain")),
            "domains": list(state.get("domains") or []),
            "created_at": _stamp(state.get("created_at")),
            "updated_at": _stamp(state.get("updated_at")),
            "completed_at": _stamp(state.get("completed_at")),
            "stages_total": len(stages),
            "stages_done": done,
            "stages_running": running,
            "directions": len(state.get("directions") or []),
            "counts": counts,
            "stats": {key: stats.get(key) for key in
                      ("facets_explored", "concepts_considered", "directions",
                       "candidates_rejected", "already_failed_in_memory")},
            "has_space": bool(((state.get("space") or {}).get("nodes"))),
            "error": _stamp(state.get("error")),
        })
    rows.sort(key=lambda row: row["created_at"] or row["updated_at"], reverse=True)
    limit = max(1, min(int(limit or 40), 200))
    shown = rows[:limit]
    return {
        "runs": shown,
        "total": len(rows),
        "truncated": _truncation(len(rows), len(shown), "runs"),
        "active_run_id": next((row["run_id"] for row in rows
                               if row["status"] == "running"), ""),
        "stalled_run_ids": [row["run_id"] for row in rows if row["stalled"]],
        "stall_after_seconds": STALL_AFTER_SECONDS,
        "latest_run_id": rows[0]["run_id"] if rows else "",
        "generated_at": _now(),
        "note": ("Slim run list for the interface. The full possibility space of a run is "
                 "served separately by /api/explorations/<run_id>/space."),
    }


# ------------------------------------------------------------------- architecture
#: System map of AARL. Each component names the real modules it wraps and the agent
#: engines that do its work, so its live state is the state of those engines.
ARCHITECTURE_COMPONENTS: Tuple[Dict[str, Any], ...] = (
    {"key": "researcher", "label": "RESEARCHER", "agents": (),
     "modules": "lab.web_ui (the interface you are looking at)",
     "purpose": "Submits the research problem, inspects evidence and decides what is "
                "worth pursuing. AARL proposes; the researcher decides.",
     "inputs": ["research problem", "documents the researcher supplies"],
     "outputs": ["approved directions", "human replication decisions"]},
    {"key": "core", "label": "AARL CORE", "agents": ("problem_analyst",),
     "modules": "lab.config.LabConfig · AARL_run.py · lab.web_api",
     "purpose": "Owns the cycle: configuration snapshot, run orchestration, stage "
                "bookkeeping and the API the interface reads.",
     "inputs": ["research problem", "engine configuration"],
     "outputs": ["run records with stage states", "research history ledger"]},
    {"key": "problem_engine", "label": "PROBLEM ENGINE", "agents": ("problem_analyst",),
     "modules": "lab.domains · lab.config",
     "purpose": "Parses the problem, detects subject and domain(s) and fixes the "
                "configuration every later stage is reproducible from.",
     "inputs": ["problem text"],
     "outputs": ["subject", "primary and relevant domains", "config snapshot"]},
    {"key": "knowledge_engine", "label": "KNOWLEDGE ENGINE",
     "agents": ("literature_researcher", "knowledge_engine"),
     "modules": "lab.knowledge · lab.papers · lab.literature",
     "purpose": "Maintains the knowledge base with provenance: sourced claims from "
                "documents, typed sections, dedupe index and honest claim labels.",
     "inputs": ["paper extractions", "researcher data", "search queries"],
     "outputs": ["knowledge items", "overlap hits", "paper records"]},
    {"key": "research_agents", "label": "RESEARCH AGENTS",
     "agents": ("direction_discovery", "critic_agent"),
     "modules": "lab.discovery.directions · facets · concepts · dimensions · gaps",
     "purpose": "Builds the possibility space: facets x transferable mechanisms, "
                "critiqued descriptively (novelty, plausibility, gaps, failure conditions).",
     "inputs": ["problem + domains", "concept library", "recorded failures"],
     "outputs": ["direction records", "dimensions", "gaps", "failure conditions"]},
    {"key": "hypothesis_engine", "label": "HYPOTHESIS ENGINE", "agents": ("hypothesis_engine",),
     "modules": "lab.hypotheses.HypothesisEngine",
     "purpose": "Turns a direction into a testable record with parameters, constraints, "
                "expected outcome and the lessons it was built from.",
     "inputs": ["directions", "stored lessons"],
     "outputs": ["hypothesis records"]},
    {"key": "experiment_engine", "label": "EXPERIMENT ENGINE",
     "agents": ("experiment_designer", "simulation_engine"),
     "modules": "lab.experiments · lab.simulation.registry · *_adapter",
     "purpose": "Picks a registered executable model, runs the seeded "
                "baseline-vs-proposal comparison and stores the measurements.",
     "inputs": ["hypothesis parameters", "registered adapters"],
     "outputs": ["simulation records with metrics and curves"]},
    {"key": "validation", "label": "VALIDATION", "agents": ("validation_agent",),
     "modules": "lab.discovery.status · lab.failures · lab.lessons",
     "purpose": "Classifies each direction from stored records only and keeps every "
                "failure plus the lesson grown from it.",
     "inputs": ["simulation outcomes", "knowledge items", "contradictions"],
     "outputs": ["direction status", "failure records", "lessons"]},
    {"key": "report", "label": "RESEARCH REPORT", "agents": ("synthesis_agent",),
     "modules": "lab.opportunity · lab.reporting.FinalReportGenerator",
     "purpose": "Assembles the report layer and the opportunity report: supported, "
                "speculative, failed, and what to do next.",
     "inputs": ["directions + statuses", "experiments", "failures", "lessons"],
     "outputs": ["research report", "next research directions"]},
    {"key": "memory", "label": "RESEARCH MEMORY", "agents": (),
     "modules": "lab.memory.ResearchMemory (append-only JSON store)",
     "purpose": "The single store every stage reads and writes: nothing is overwritten, "
                "revisions are kept, failed ideas are never deleted.",
     "inputs": ["records from every engine"],
     "outputs": ["revisions", "research history ledger", "derived record index"]},
)


def architecture(memory: Any, knowledge: Any, run: Optional[Dict[str, Any]],
                 records: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    """Interactive system map with the live state of each component."""
    records = records if records is not None else _records(memory)
    agent_payload = agents(memory, knowledge, run, records)
    by_key = {row["key"]: row for row in agent_payload["agents"]}
    items = _knowledge_items(knowledge)
    history_events = len(_CACHE.get("history", _memory_signature(memory, ()),
                                    lambda: list(memory.load_history())))
    components: List[Dict[str, Any]] = []
    for spec in ARCHITECTURE_COMPONENTS:
        related = [by_key[key] for key in spec["agents"] if key in by_key]
        counts: List[Dict[str, Any]] = [{"label": "agent engines", "value": len(related)}]
        for row in related:
            for count in row["counts"][:2]:
                counts.append({"label": "%s · %s" % (row["label"], count["label"]),
                               "value": count["value"]})
        if spec["key"] == "knowledge_engine":
            counts.append({"label": "knowledge items", "value": len(items)})
        if spec["key"] == "memory":
            counts.append({"label": "records stored",
                           "value": sum(len(records.get(c, [])) for c in _COLLECTIONS)})
            counts.append({"label": "history events", "value": history_events})
        if spec["key"] == "core":
            counts.append({"label": "exploration runs",
                           "value": len(records.get("explorations", []))})
        if any(row["state"] == "ACTIVE" for row in related):
            state, reason = "ACTIVE", "one of its engines owns the running backend stage"
        elif any(row["state"] == "FAULT" for row in related):
            state, reason = "FAULT", "a run failed while its engine was working"
        elif related and all(row["state"] == "NO DATA" for row in related):
            state, reason = "NO DATA", "no record of this component exists yet"
        elif related and any(row["state"] == "STANDBY" for row in related):
            state, reason = "STANDBY", "the active run has not reached this component"
        elif any(count["value"] for count in counts):
            state, reason = "READY", "records from this component exist in research memory"
        else:
            state, reason = "IDLE", "no run is active and no record exists for it yet"
        components.append({
            "key": spec["key"], "label": spec["label"], "modules": spec["modules"],
            "purpose": spec["purpose"], "inputs": list(spec["inputs"]),
            "outputs": list(spec["outputs"]), "state": state, "state_reason": reason,
            "counts": counts,
            "agents": [{"key": row["key"], "label": row["label"], "state": row["state"],
                        "role": row["role"], "records_in_run": row["records_in_run"]}
                       for row in related],
            "last_active": max([row["last_active"] for row in related] or [""]),
        })
    order = {"ACTIVE": 0, "FAULT": 1, "STANDBY": 2, "READY": 3, "IDLE": 4, "NO DATA": 5}
    return {
        "components": components,
        "flow": ["researcher", "core", "problem_engine", "knowledge_engine",
                 "research_agents", "hypothesis_engine", "experiment_engine",
                 "validation", "report", "memory"],
        "run_id": _stamp((run or {}).get("run_id")),
        "states": sorted({component["state"] for component in components},
                         key=lambda status: order.get(status, 9)),
        "generated_at": _now(),
        "note": ("Each component reports the state of the engines it wraps: the running "
                 "backend stage, and the records those engines stored. A component with no "
                 "record reports NO DATA rather than a fabricated status."),
    }


# ----------------------------------------------------------------------- overview
def overview(memory: Any, knowledge: Any, live_states: Sequence[Optional[Dict[str, Any]]] = (),
             run: Optional[Dict[str, Any]] = None, events_limit: int = 40) -> Dict[str, Any]:
    """Single bootstrap payload for the command centre (one round trip)."""
    records = _records(memory)
    run_list = runs(memory, live_states)
    selected = run
    if selected is None:
        target = run_list.get("active_run_id") or run_list.get("latest_run_id")
        selected = next((state for state in live_states
                         if isinstance(state, dict)
                         and _stamp(state.get("run_id")) == target), None)
        if selected is None and target:
            selected = latest_stored(memory, "explorations", target)
    scope = run_scope(memory, selected, records)
    items = _knowledge_items(knowledge)
    summary = knowledge.summary()
    agent_payload = agents(memory, knowledge, selected, records)
    history = _CACHE.get("history", _memory_signature(memory, ()),
                         lambda: list(memory.load_history()))
    return {
        "server_time": _now(),
        "knowledge": {
            "total_items": summary.get("total_items", 0),
            "by_section": summary.get("by_section", {}),
            "by_claim_type": summary.get("by_claim_type", {}),
            "contradiction_records": summary.get("contradiction_records", 0),
            "honesty_note": summary.get("honesty_note", ""),
        },
        "memory": {
            "root": memory.root,
            "collections": {collection: len(records.get(collection, []))
                            for collection in _COLLECTIONS},
            "history_events": len(history),
        },
        "runs": run_list,
        "run": {
            "run_id": _stamp(scope.get("run_id")),
            "question": _stamp(scope.get("question")),
            "state": live_status(selected),
            "counts": scope.get("counts", {}),
            "domains": list((selected or {}).get("domains") or []),
            "primary_domain": _stamp((selected or {}).get("primary_domain")),
            "subject": _stamp((selected or {}).get("subject") or
                              ((selected or {}).get("stats") or {}).get("subject_label")),
            "stats": (selected or {}).get("stats") or {},
        },
        "pipeline": pipeline(selected, scope),
        "agents": agent_payload,
        "architecture": architecture(memory, knowledge, selected, records),
        "events": events(memory, [selected], events_limit),
        "totals": {
            "papers": len(records.get("papers", [])),
            "knowledge_items": len(items),
            "sourced_items": len([item for item in items
                                  if _stamp(item.get("claim_type")) == "sourced"]),
            "directions": len(records.get("directions", [])),
            "hypotheses": len(records.get("hypotheses", [])),
            "experiments": len(records.get("simulations", [])),
            "contradictions": len(records.get("failures", [])),
            "lessons": len(records.get("lessons", [])),
            "runs": run_list.get("total", 0),
        },
        "generated_at": _now(),
    }
