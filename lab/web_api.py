"""AARL Web API - honest bridge between engines and GUI (Flask)."""
from __future__ import annotations
import datetime, json, os, re, threading, traceback, uuid
from typing import Any, Callable, Dict, List, Optional
from flask import Flask, jsonify, request, send_from_directory
from lab import observatory
from lab.config import LabConfig
from lab.cycle import ResearchCycle
from lab.discovery.directions import DirectionDiscovery
from lab.domains import relevant_domains, primary_domain
from lab.knowledge import KnowledgeManager
from lab.memory import ResearchMemory
from lab.papers import PaperIngestor
UI_DIR = os.path.join(os.path.dirname(__file__), "web_ui")
def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")
class ExplorationRegistry:
    def __init__(self, memory: ResearchMemory):
        self.memory = memory
        self._lock = threading.Lock()
        self._runs: Dict[str, Dict[str, Any]] = {}
    def create(self, problem: str) -> Dict[str, Any]:
        run_id = "RUN-%s" % uuid.uuid4().hex[:8].upper()
        state: Dict[str, Any] = {"run_id": run_id, "problem": problem, "status": "queued", "stages": [], "directions": [], "space": {}, "stats": {}, "error": "", "created_at": _now(), "updated_at": _now()}
        # The stage list the GUI must render is declared by the backend, so the UI
        # never invents pipeline steps that the engine does not actually perform.
        state["stage_template"] = [{"key": k, "label": l} for k, l in STAGE_LABELS]
        with self._lock:
            self._runs[run_id] = state
        self.memory.append_record("explorations", run_id, state)
        self.memory.append_history({"event": "exploration_created", "run_id": run_id, "problem": problem})
        return dict(state)
    def _latest_stored(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Newest stored version of a run: the base file or the highest ``.revN``.

        Progress updates are written as revisions (memory never overwrites), so a
        run reloaded after a restart would otherwise appear stuck at its very first
        state. This resolves to what was actually last recorded.
        """
        base = self.memory.get("explorations", run_id)
        if base is None:
            return None
        best, best_n = base, 1
        try:
            directory = self.memory.collection_dir("explorations")
            for name in os.listdir(directory):
                match = re.fullmatch(r"%s\.rev(\d+)\.json" % re.escape(str(run_id)), name)
                if not match:
                    continue
                n = int(match.group(1))
                if n > best_n:
                    body = self.memory.get("explorations", "%s.rev%d" % (run_id, n))
                    if isinstance(body, dict):
                        best, best_n = body, n
        except FileNotFoundError:  # pragma: no cover - directory exists
            pass
        return best

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._runs.get(run_id)
        if state is not None:
            return dict(state)
        return self._latest_stored(run_id)

    def live(self) -> List[Dict[str, Any]]:
        """In-memory run states only (cheap overlay for the observatory views).

        ``list()`` resolves every run from disk and carries the full possibility
        space of each one; the interface only needs the live ones on top of the
        stored states it loads itself.
        """
        with self._lock:
            return [dict(state) for state in self._runs.values()]

    def selected(self, run_id: str = "") -> Optional[Dict[str, Any]]:
        """A run by id, else the running one, else the newest recorded run."""
        if run_id:
            return self.get(run_id)
        with self._lock:
            states = list(self._runs.values())
        active = [s for s in states if str(s.get("status")) == "running"]
        if active:
            return dict(active[-1])
        if states:
            return dict(states[-1])
        stored = self.list()
        return stored[0] if stored else None
    def update(self, run_id: str, fn: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                state = self._latest_stored(run_id) or {}
                self._runs[run_id] = state
            fn(state)
            state["updated_at"] = _now()
            snapshot = json.loads(json.dumps(state, default=str))
        try:
            self.memory.append_record("explorations", run_id, snapshot, revision_of="progress update")
        except Exception:
            pass
        return snapshot
    def list(self) -> List[Dict[str, Any]]:
        records = []
        for record in self.memory.load_collection("explorations"):
            run_id = str(record.get("record_id", ""))
            records.append(self._latest_stored(run_id) or record)
        with self._lock:
            for run_id, state in self._runs.items():
                if not any(r.get("record_id") == run_id for r in records):
                    records.append(dict(state, record_id=run_id))
        records.sort(key=lambda r: str(r.get("created_at") or r.get("stored_at", "")), reverse=True)
        return records
STAGE_LABELS = [("understanding", "Understanding problem"), ("domains", "Detecting research domain"), ("literature", "Searching literature"), ("graph", "Building knowledge graph"), ("exploring", "Exploring alternative directions"), ("hypotheses", "Generating hypotheses"), ("mechanisms", "Testing mechanisms"), ("simulations", "Running simulations"), ("evaluating", "Evaluating research opportunities")]
def _set_stage(registry: ExplorationRegistry, run_id: str, key: str, state: str, detail: str = "") -> None:
    lbl = dict(STAGE_LABELS).get(key, key)
    def _fn(s: Dict[str, Any]) -> None:
        stages = s.setdefault("stages", [])
        for st in stages:
            if st.get("key") == key:
                st.update({"state": state, "label": lbl, "detail": detail, "at": _now()})
                break
        else:
            stages.append({"key": key, "label": lbl, "state": state, "detail": detail, "at": _now()})
    registry.update(run_id, _fn)
def _resolve_stack(memory_dir="research_memory"):
    memory = ResearchMemory(memory_dir)
    knowledge = KnowledgeManager(memory)
    return memory, knowledge
def _get_llm():
    """Return a usable LLM service, or ``None`` when none is configured.

    ``make_llm_service`` returns a service object even without an API key (it
    exposes ``available=False``), so it must be checked before being handed to
    the research engines — otherwise every engine pays a failing call and the
    GUI would report an LLM that is not actually usable.
    """
    try:
        from showcase.llm_service import make_llm_service
        svc = make_llm_service()
        return svc if getattr(svc, "available", True) else None
    except Exception:
        return None
def _related_for_direction(direction, memory):
    did = str(direction.get("direction_id", ""))
    hyps = [h for h in memory.load_collection("hypotheses") if str(h.get("direction_id", "")) == did]
    if not hyps:
        key = str(direction.get("mechanism_key", ""))
        if key:
            hyps = [h for h in memory.load_collection("hypotheses") if str(h.get("direction_mechanism_key", "")) == key]
    exp_ids = {str(h.get("experiment_id", "")) for h in hyps if h.get("experiment_id")}
    for hid in [str(h.get("hypothesis_id", "")) for h in hyps]:
        for s in memory.load_collection("simulations"):
            if str(s.get("hypothesis_id", "")) == hid:
                exp_ids.add(str(s.get("experiment_id", "")))
    sims = [s for s in memory.load_collection("simulations") if str(s.get("experiment_id", "")) in exp_ids or str(s.get("direction_id", "")) == did]
    fails = [f for f in memory.load_collection("failures") if str(f.get("direction_id", "")) == did or str(f.get("experiment_id", "")) in exp_ids]
    return hyps, sims, fails
def create_app(memory_dir="research_memory"):
    memory, knowledge = _resolve_stack(memory_dir)
    llm = _get_llm()
    registry = ExplorationRegistry(memory)

    def _latest_record(collection, record_id):
        """Newest version of a record: base file or the highest .revN (evaluated)."""
        if not record_id:
            return None
        base = memory.get(collection, record_id)
        if base is None:
            return None
        best, best_n = base, 1
        try:
            for name in os.listdir(memory.collection_dir(collection)):
                match = re.fullmatch(
                    r"%s\.rev(\d+)\.json" % re.escape(str(record_id)), name
                )
                if not match:
                    continue
                n = int(match.group(1))
                if n > best_n:
                    body = memory.get(collection, "%s.rev%d" % (record_id, n))
                    if isinstance(body, dict):
                        best, best_n = body, n
        except FileNotFoundError:  # pragma: no cover
            pass
        return best

    def _latest_collection(collection):
        out = []
        for record in memory.load_collection(collection):
            out.append(_latest_record(collection, str(record.get("record_id", ""))) or record)
        return out

    app = Flask(__name__, static_folder=UI_DIR, static_url_path="")
    try:
        app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    except Exception:
        pass
    def _run_exploration(run_id, problem):
        # The GUI shows live/done state from `stages`; `status` marks the run as
        # in progress so "live" is never displayed for a finished or idle run.
        registry.update(run_id, lambda s: s.update({"status": "running"}))
        _set_stage(registry, run_id, "understanding", "running", "Parsing problem.")
        try:
            cycle = ResearchCycle(config=LabConfig(research_question=problem, memory_dir=memory.root), llm_service=llm, memory=memory)
            discovery = DirectionDiscovery(memory, knowledge)
            _set_stage(registry, run_id, "understanding", "done", problem[:140])
            _set_stage(registry, run_id, "domains", "running", "Detecting domains.")
            domains = relevant_domains(problem)
            primary = primary_domain(problem)
            _set_stage(registry, run_id, "domains", "done", "Primary: %s." % primary)
            _set_stage(registry, run_id, "literature", "running", "Scanning knowledge.")
            lit = knowledge.search(problem, limit=8)
            hits = [i.to_dict() if hasattr(i, "to_dict") else dict(i) for i in lit]
            _set_stage(registry, run_id, "literature", "done", "%d overlapping items." % len(hits))
            _set_stage(registry, run_id, "graph", "running", "Building possibility space.")
            bundle = discovery.discover(problem, domains, primary, run_id=run_id)
            directions = bundle.get("directions", [])
            _set_stage(registry, run_id, "graph", "done", "%d directions." % len(directions))
            _set_stage(registry, run_id, "exploring", "done", "Proposals ready.")
            registry.update(run_id, lambda s: s.update({"space": bundle.get("space", {}), "stats": bundle.get("stats", {}), "rejected": bundle.get("rejected", []), "subject": bundle.get("subject", ""), "literature_overlap": hits, "domains": domains, "primary_domain": primary}))
            _set_stage(registry, run_id, "hypotheses", "running", "Generating hypotheses.")
            hyp_by = {}
            for idx, direction in enumerate(directions):
                q = "Direction %s: %s. %s" % (direction.get("direction_id", ""), direction.get("title", ""), direction.get("core_question", ""))
                hyps = cycle.hypotheses.generate(q, 1, primary)
                tagged = []
                for hyp in hyps:
                    hyp = dict(hyp)
                    hyp["direction_id"] = direction.get("direction_id", "")
                    hyp["direction_mechanism_key"] = str(direction.get("mechanism_key", ""))
                    hyp["exploration_run_id"] = run_id
                    cycle.memory.append_record("hypotheses", hyp.get("hypothesis_id", ""), hyp, revision_of="exploration")
                    stored = cycle.memory.get("hypotheses", hyp.get("hypothesis_id", "")) or hyp
                    knowledge.add(section="hypotheses", statement=str(stored.get("hypothesis", ""))[:600], claim_type="generated_hypothesis", provenance={"derived_from": direction.get("direction_id", ""), "generator": "aarl.hypothesis_engine", "run_id": run_id, "hypothesis_id": stored.get("hypothesis_id", "")}, source="aarl.hypothesis_engine", confidence=0.4, metadata={"hypothesis_id": stored.get("hypothesis_id", "")})
                    tagged.append(stored)
                hyp_by[direction["direction_id"]] = tagged
                _set_stage(registry, run_id, "hypotheses", "running", "Dir %d/%d done." % (idx + 1, len(directions)))
            _set_stage(registry, run_id, "hypotheses", "done", "Hypotheses stored.")
            _set_stage(registry, run_id, "mechanisms", "done", "PROPOSED mechanisms only.")
            _set_stage(registry, run_id, "simulations", "running", "Simulating.")
            sim_by = {}
            fail_by = {}
            lessons_all = memory.load_collection("lessons")
            total = len(directions)
            for i, direction in enumerate(directions):
                sims = []
                for hyp in hyp_by.get(direction["direction_id"], []):
                    sim = cycle.experiments.run(hyp, problem, 1)
                    sim["direction_mechanism_key"] = str(direction.get("mechanism_key", ""))
                    sim["direction_id"] = direction.get("direction_id", "")
                    memory.append_record("simulations", sim.get("experiment_id", ""), sim, revision_of="linked to direction")
                    sims.append(memory.get("simulations", sim.get("experiment_id", "")) or sim)
                    if cycle.failure_analyzer.is_failure(sim):
                        failure = cycle.failure_analyzer.analyze(sim)
                        failure["direction_mechanism_key"] = str(direction.get("mechanism_key", ""))
                        failure["direction_id"] = direction.get("direction_id", "")
                        memory.append_record("failures", failure.get("failure_id", ""), failure, revision_of="linked to direction")
                        lesson = cycle.lessons.from_failure(failure)
                        lesson["direction_mechanism_key"] = str(direction.get("mechanism_key", ""))
                        fail_by.setdefault(direction["direction_id"], []).append(memory.get("failures", failure.get("failure_id", "")) or failure)
                sim_by[direction["direction_id"]] = sims
                _set_stage(registry, run_id, "simulations", "running", "Direction %d/%d: %d simulations." % (i + 1, total, len(sims)))
            _set_stage(registry, run_id, "simulations", "done", "Simulations recorded honestly.")
            _set_stage(registry, run_id, "evaluating", "running", "Status, dimensions, evidence, gaps.")
            evaluated = []
            for direction in directions:
                did = direction["direction_id"]
                ev = discovery.evaluate(
                    direction,
                    hypotheses=hyp_by.get(did, []),
                    experiments=sim_by.get(did, []),
                    failures=fail_by.get(did, []),
                    lessons=lessons_all,
                    all_directions=directions,
                    persist=True,
                )
                evaluated.append(ev)
                _set_stage(registry, run_id, "evaluating", "running",
                           "Evaluated %d/%d." % (len(evaluated), len(directions)))

            def _summarise(d):
                st = d.get("status") or {}
                dims = d.get("dimensions") or {}
                evc = (d.get("evidence") or {}).get("counts") or {}
                gaps = d.get("gaps") or []
                unknowns = d.get("unknowns") or []
                main_unc = ""
                if gaps and isinstance(gaps[0], dict):
                    main_unc = str(gaps[0].get("question", gaps[0].get("gap", "")))
                elif unknowns:
                    main_unc = str(unknowns[0])
                return {
                    "direction_id": d.get("direction_id", ""),
                    "title": d.get("title", ""),
                    "core_question": d.get("core_question", ""),
                    "status": st.get("status", ""),
                    "status_label": st.get("label", ""),
                    "dot": st.get("dot", "#8b94a7"),
                    "css": st.get("css", ""),
                    "status_meaning": st.get("meaning", ""),
                    "status_reasons": st.get("reasons", []),
                    "rule_id": st.get("rule_id", ""),
                    "inspired_by": (d.get("inspiration") or {}).get("display", ""),
                    "potential": (dims.get("potential_novelty") or {}).get("level", ""),
                    "evidence_level": (dims.get("evidence_strength") or {}).get("level", ""),
                    "direct": evc.get("direct", 0),
                    "indirect": evc.get("indirect", 0),
                    "contradicting": evc.get("contradicting", 0),
                    "missing": evc.get("missing", 0),
                    "main_uncertainty": main_unc,
                    "already_failed": bool(d.get("already_failed")),
                    "refinement_title": ((d.get("refinement") or {}).get("title", "")
                                         if d.get("refinement") else ""),
                    "full": d,
                }
            summaries = [_summarise(d) for d in evaluated]
            counts = {
                "directions": len(summaries),
                "evidence_backed": sum(1 for s in summaries if s["status"] == "EVIDENCE_BACKED"),
                "promising": sum(1 for s in summaries if s["status"] == "PROMISING"),
                "speculative": sum(1 for s in summaries if s["status"] == "SPECULATIVE"),
                "inconclusive": sum(1 for s in summaries if s["status"] == "INCONCLUSIVE"),
                "failed_weak": sum(1 for s in summaries if s["status"] == "FAILED_WEAK"),
            }

            def _final(s):
                s["directions"] = summaries
                s["counts"] = counts
                s["status"] = "complete"
                s["completed_at"] = _now()
            registry.update(run_id, _final)
            _set_stage(registry, run_id, "evaluating", "done",
                       "%d directions evaluated." % len(evaluated))
            memory.append_history({"event": "exploration_completed", "run_id": run_id,
                                   "directions": len(evaluated)})
        except Exception as exc:  # noqa: BLE001 - surfaced honestly to the GUI
            traceback.print_exc()
            registry.update(run_id, lambda s: s.update({
                "status": "failed", "error": str(exc)[:500]}))
            memory.append_history({"event": "exploration_failed", "run_id": run_id,
                                   "error": str(exc)[:300]})
    # ------------------------------------------------------------------ pages
    @app.get("/")
    def _index():
        return send_from_directory(UI_DIR, "simple.html")

    @app.get("/classic")
    def _classic():
        return send_from_directory(UI_DIR, "index.html")

    # ------------------------------------------------------------------ system
    @app.get("/api/health")
    def _health():
        return jsonify({
            "ok": True, "time": _now(), "memory_dir": memory.root,
            "knowledge_items": knowledge.summary().get("total_items", 0),
            "llm_available": llm is not None,
        })

    @app.get("/api/memory/summary")
    def _memory_summary():
        cols = ("papers", "hypotheses", "simulations", "failures", "lessons",
                "directions", "explorations", "real_world_experiments")
        return jsonify({
            "knowledge": knowledge.summary(),
            "collections": {c: len(memory.load_collection(c)) for c in cols},
            "recent_events": memory.load_history()[-20:],
        })

    # ------------------------------------------------------------------ explorations
    @app.post("/api/explorations")
    def _create_exploration():
        body = request.get_json(force=True, silent=True) or {}
        problem = str(body.get("problem", "") or "").strip()
        if len(problem) < 4:
            return jsonify({"error": "Describe a research problem (a few words minimum)."}), 400
        state = registry.create(problem)
        threading.Thread(target=_run_exploration, args=(state["run_id"], problem),
                         daemon=True).start()
        return jsonify({"run_id": state["run_id"], "status": "running"})

    @app.get("/api/explorations")
    def _list_explorations():
        return jsonify({"explorations": registry.list()})

    @app.get("/api/explorations/<run_id>")
    def _get_exploration(run_id: str):
        state = registry.get(run_id)
        if not state:
            return jsonify({"error": "unknown run"}), 404
        return jsonify(state)

    @app.get("/api/explorations/<run_id>/space")
    def _exploration_space(run_id: str):
        state = registry.get(run_id)
        if not state:
            return jsonify({"error": "unknown run"}), 404
        return jsonify({
            "run_id": run_id,
            "problem": state.get("problem", ""),
            "space": state.get("space") or {},
            "stats": state.get("stats") or {},
            "status": state.get("status", ""),
        })

    # ------------------------------------------------------------------ directions
    @app.get("/api/directions")
    def _list_directions():
        out = []
        for d in _latest_collection("directions"):
            st = d.get("status") or {}
            out.append({
                "direction_id": d.get("direction_id", ""),
                "title": d.get("title", ""),
                "core_question": d.get("core_question", ""),
                "status": st.get("status", ""),
                "status_label": st.get("label", ""),
                "dot": st.get("dot", "#8b94a7"),
                "run_id": d.get("run_id", ""),
                "already_failed": bool(d.get("already_failed")),
            })
        return jsonify({"directions": out})

    @app.get("/api/directions/<direction_id>")
    def _get_direction(direction_id: str):
        d = _latest_record("directions", direction_id)
        if not d:
            return jsonify({"error": "unknown direction"}), 404
        hyps, sims, fails = _related_for_direction(d, memory)
        return jsonify({"direction": d, "hypotheses": hyps,
                        "simulations": sims, "failures": fails})

    @app.get("/api/directions/<direction_id>/report")
    def _direction_report(direction_id: str):
        d = _latest_record("directions", direction_id)
        if not d:
            return jsonify({"error": "unknown direction"}), 404
        from lab.opportunity import opportunity_report
        return jsonify(opportunity_report(d))

    # ------------------------------------------------------------------ literature
    @app.post("/api/papers/search")
    def _search_papers():
        body = request.get_json(force=True, silent=True) or {}
        query = str(body.get("query", "") or "").strip()
        if len(query) < 3:
            return jsonify({"error": "Enter a search query."}), 400
        limit = int(body.get("limit", 8) or 8)
        from lab.literature import search_literature
        try:
            results = search_literature(query, limit=limit)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": "Search failed: %s" % exc, "results": []}), 502
        return jsonify({"query": query, "results": results,
                        "note": ("Metadata and abstracts only, from public APIs "
                                 "(arXiv / Crossref). Full text stays at the source link.")})

    @app.post("/api/papers/ingest")
    def _ingest_paper():
        body = request.get_json(force=True, silent=True) or {}
        text = str(body.get("text", "") or "").strip()
        path = ""
        if body.get("path"):
            path = str(body.get("path", ""))
        structured = body.get("structured") if isinstance(body.get("structured"), dict) else None
        if not text and not path and not structured:
            return jsonify({"error": "Provide paper text, a file path, or structured metadata."}), 400
        ingestor = PaperIngestor(memory, knowledge, llm_service=llm)
        try:
            record = ingestor.ingest(path=path, text=text, structured=structured,
                                     source_url=str(body.get("source_url", "") or ""),
                                     provided_by="researcher")
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": "Ingestion failed: %s" % exc}), 400
        return jsonify({"paper": record})

    @app.get("/api/papers")
    def _list_papers():
        papers = memory.load_collection("papers")
        items = knowledge.items()
        by_paper: Dict[str, List[Any]] = {}
        for it in items:
            pid = str((it.provenance or {}).get("paper_id")
                      or it.metadata.get("paper_id") or "")
            if pid:
                by_paper.setdefault(pid, []).append(it)
        out = []
        for p in papers:
            pid = str(p.get("paper_id", ""))
            ex = p.get("extraction") or {}
            items_of = by_paper.get(pid, [])
            out.append({
                "paper_id": pid,
                "title": str(ex.get("title") or p.get("title") or "(untitled)"),
                "year": ex.get("year") or p.get("year") or "",
                "source_file": p.get("source_file") or "",
                # Legitimate source location, shown as a link-out (never a copy).
                "source_url": p.get("source_url") or (p.get("provenance") or {}).get("source_url", ""),
                "provided_by": (p.get("provenance") or {}).get("provided_by",
                                                               "researcher"),
                "stored_at": p.get("stored_at", ""),
                "counts": {
                    "claims": len(ex.get("claims") or []),
                    "limitations": len(ex.get("limitations") or []),
                    "results": len(ex.get("results") or []),
                    "concepts": len(set(ex.get("concepts") or [])),
                    "knowledge_items": len(items_of),
                },
                "citation": (p.get("provenance") or {}).get("citation", ""),
                # Derived, not stored: whether this paper currently contributes
                # knowledge. A withdrawn paper stays in the library and is shown
                # as contributing nothing rather than silently disappearing.
                "in_knowledge": bool(items_of),
                "extracted_total": len(p.get("integrated_items") or []),
            })
        out.sort(key=lambda r: str(r.get("stored_at", "")), reverse=True)
        return jsonify({"papers": out,
                        "note": ("Papers with extracted knowledge contribute to AARL's "
                                 "knowledge base. 'Remove from AARL knowledge' deletes "
                                 "only the claims this paper contributed.")})

    @app.get("/api/papers/<paper_id>")
    def _get_paper(paper_id: str):
        p = memory.get("papers", paper_id)
        if not p:
            return jsonify({"error": "unknown paper"}), 404
        contributed = [i.to_dict() for i in knowledge.items()
                       if str((i.provenance or {}).get("paper_id") or "") == paper_id
                       or str(i.metadata.get("paper_id") or "") == paper_id]
        return jsonify({"paper": p, "knowledge_items": contributed,
                        "note": ("Only claims extracted from the paper text itself are "
                                 "stored as 'sourced'; everything the extractor inferred "
                                 "is labelled 'inferred'.")})

    @app.delete("/api/papers/<paper_id>")
    def _remove_paper(paper_id: str):
        """Withdraw a paper's knowledge, or delete the paper record entirely.

        ``?knowledge_only=1`` (or ``{"knowledge_only": true}``) withdraws only the
        knowledge items the paper contributed and keeps the paper in the research
        library; without it the paper record is deleted as well. Both variants are
        recorded in the research history, and neither touches knowledge that came
        from other sources.
        """
        p = memory.get("papers", paper_id)
        if not p:
            return jsonify({"error": "unknown paper"}), 404
        body = request.get_json(silent=True) or {}
        flag = request.args.get("knowledge_only", body.get("knowledge_only", ""))
        knowledge_only = str(flag).strip().lower() in ("1", "true", "yes", "on")
        removed = knowledge.remove_by_paper(paper_id)
        if not knowledge_only:
            try:
                os.remove(memory.path_for("papers", paper_id))
            except OSError:
                pass
        memory.append_history({
            "event": ("paper_knowledge_withdrawn" if knowledge_only
                      else "paper_removed_from_knowledge"),
            "paper_id": paper_id,
            "knowledge_items_removed": removed,
            "paper_record_kept": knowledge_only,
        })
        return jsonify({
            "removed_paper": "" if knowledge_only else paper_id,
            "paper_kept": knowledge_only,
            "knowledge_items_removed": removed,
        })

    @app.post("/api/papers/upload")
    def _upload_paper():
        f = request.files.get("file")
        if f is None or not f.filename:
            return jsonify({"error": "No file provided."}), 400
        fname = os.path.basename(f.filename)
        if not fname.lower().endswith((".pdf", ".txt", ".md")):
            return jsonify({"error": "Supported formats: PDF, TXT, MD."}), 400
        upload_dir = os.path.join(memory.root, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        path = os.path.join(upload_dir, "%s_%s" % (uuid.uuid4().hex[:6], fname))
        f.save(path)
        ingestor = PaperIngestor(memory, knowledge, llm_service=llm)
        try:
            record = ingestor.ingest(path=path, provided_by="researcher")
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": "Ingestion failed: %s" % exc}), 400
        return jsonify({"paper": record})

    # ------------------------------------------------------------------ knowledge
    @app.get("/api/knowledge")
    def _knowledge_items():
        section = request.args.get("section", "")
        claim_type = request.args.get("claim_type", "")
        q = request.args.get("q", "").strip()
        limit = min(int(request.args.get("limit", 200) or 200), 1000)
        items = knowledge.items(section=section or None, claim_type=claim_type or None)
        if q:
            ql = q.lower()
            items = [i for i in items if ql in str(i.statement).lower()
                     or ql in str(i.source_paper).lower()]
        items = items[:limit]
        return jsonify({
            "items": [i.to_dict() for i in items],
            "summary": knowledge.summary(),
            "note": ("Every item carries provenance. 'sourced' items come from paper "
                     "text; 'inferred'/'generated_hypothesis' items are AARL's own "
                     "reasoning and are never presented as literature evidence."),
        })

    # ------------------------------------------------------------------ graph
    @app.get("/api/graph")
    def _knowledge_graph():
        """Typed knowledge graph built from stored records only."""
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        def node(nid: str, label: str, kind: str, **extra: Any) -> None:
            if nid not in nodes:
                nodes[nid] = {"id": nid, "label": label[:80], "kind": kind, **extra}

        def edge(src: str, dst: str, etype: str, note: str = "") -> None:
            if src in nodes and dst in nodes:
                edges.append({"source": src, "target": dst, "type": etype, "note": note})

        # knowledge items as claims
        hypotheses = memory.load_collection("hypotheses")
        by_direction: Dict[str, List[Dict[str, Any]]] = {}
        for h in hypotheses:
            did = str(h.get("direction_id", ""))
            if not did:
                # A stored hypothesis names its direction in its own text; that is a
                # declared reference, not a guessed match.
                did = observatory._direction_id_of(h)[0]
            if did:
                by_direction.setdefault(did, []).append(h)
        for it in knowledge.items():
            d = it.to_dict()
            nid = "claim:%s" % d["item_id"]
            node(nid, d["statement"][:80], "claim",
                 claim_type=d["claim_type"], section=d["section"],
                 confidence=d["confidence"], statement=d["statement"])
            if d.get("source_paper"):
                pid = "paper:%s" % d["source_paper"]
                node(pid, d["source_paper"][:80], "paper")
                edge(pid, nid, "supports",
                     "sourced from paper text" if d["claim_type"] == "sourced"
                     else "related wording only (NOT provenance-linked)")
            for rel in d.get("relationships") or []:
                other = "claim:%s" % rel.get("target", "")
                node(other, str(rel.get("target", ""))[:80], "claim")
                edge(nid, other, str(rel.get("type", "related_to")),
                     str(rel.get("note", "")))

        # papers
        for p in memory.load_collection("papers"):
            pid = "paper_rec:%s" % p.get("paper_id", "")
            ex = p.get("extraction") or {}
            node(pid, str(ex.get("title") or "(untitled)")[:80], "paper",
                 paper_id=p.get("paper_id", ""))

        # hypotheses
        for h in hypotheses:
            nid = "hyp:%s" % h.get("hypothesis_id", "")
            node(nid, str(h.get("hypothesis", ""))[:80], "hypothesis",
                 status=str(h.get("status", "")))
            for iid in h.get("based_on_knowledge") or []:
                edge("claim:%s" % iid, nid, "derived_from")
            for iid in h.get("supporting_evidence") or []:
                edge("claim:%s" % iid, nid, "supports")

        # simulations
        for s in memory.load_collection("simulations"):
            nid = "sim:%s" % s.get("experiment_id", "")
            node(nid, "%s (%s)" % (s.get("experiment_id", ""), s.get("status", "")),
                 "experiment", status=str(s.get("status", "")))
            edge("hyp:%s" % s.get("hypothesis_id", ""), nid, "tested_by")

        # failures (failure modes stay visible as research memory)
        for f in memory.load_collection("failures"):
            nid = "fail:%s" % f.get("failure_id", "")
            node(nid, str(f.get("observed_behavior") or f.get("failure_type", ""))[:80],
                 "failure_mode", failure_type=f.get("failure_type", ""),
                 severity=f.get("severity", ""))
            edge("sim:%s" % f.get("experiment_id", ""), nid, "causes",
                 "simulation ended in failure")

        # directions
        for d in memory.load_collection("directions"):
            nid = "dir:%s" % d.get("direction_id", "")
            st = d.get("status") or {}
            node(nid, str(d.get("title", ""))[:80], "hypothesis",
                 status=st.get("status", ""), dot=st.get("dot", ""))
            for h in by_direction.get(str(d.get("direction_id", "")), []):
                edge(nid, "hyp:%s" % h.get("hypothesis_id", ""),
                     "hypothesized_to_affect",
                     "link recorded by the engine or declared in the hypothesis text")

        # possibility-space nodes from stored explorations
        space_nodes: List[Dict[str, Any]] = []
        for run in memory.load_collection("explorations"):
            sp = run.get("space") or {}
            for n in sp.get("nodes") or []:
                space_nodes.append(n)

        return jsonify({
            "nodes": list(nodes.values()),
            "edges": edges,
            "space": space_nodes,
            "edge_types": sorted({e["type"] for e in edges}),
            "legend": {
                "claim": "Knowledge item (provenance-carrying)",
                "paper": "Research paper / source",
                "hypothesis": "Hypothesis or direction",
                "experiment": "Simulation record",
                "failure_mode": "Recorded failure (kept as research memory)",
            },
            "note": ("Edge types describe how records are linked in the store. An edge "
                     "is never called a scientific discovery because of a high score."),
        })

    # ------------------------------------------------------------------ failures
    @app.get("/api/failures")
    def _list_failures():
        out = []
        for f in memory.load_collection("failures"):
            out.append({
                "failure_id": f.get("failure_id", ""),
                "experiment_id": f.get("experiment_id", ""),
                "direction_id": f.get("direction_id", ""),
                "direction_mechanism_key": f.get("direction_mechanism_key", ""),
                "failure_type": f.get("failure_type", ""),
                "severity": f.get("severity", ""),
                "hypothesis": f.get("hypothesis", ""),
                "observed_behavior": f.get("observed_behavior", ""),
                "failed_assumption": f.get("failed_assumption", ""),
                "lesson": f.get("lesson", ""),
                "recommended_changes": list(f.get("recommended_changes") or []),
                "future_direction": f.get("future_direction", ""),
                "stored_at": f.get("stored_at", ""),
            })
        out.sort(key=lambda r: str(r.get("stored_at", "")), reverse=True)
        return jsonify({
            "failures": out,
            "note": ("Failed ideas are stored as research memory, never deleted. "
                     "Future explorations avoid repeating the same failed mechanism."),
        })

    # ------------------------------------------------------------------ observatory
    # Read-only instrumentation of the research loop (see lab/observatory.py). Every
    # number here is derived from stored records plus live run stage state; nothing
    # is fabricated for the interface.
    def _run_for(run_id: str) -> Optional[Dict[str, Any]]:
        return registry.selected(run_id)

    def _guarded(builder, label):
        try:
            return jsonify(builder())
        except Exception as exc:  # noqa: BLE001 - surfaced honestly to the interface
            traceback.print_exc()
            return jsonify({"error": "%s view failed: %s" % (label, exc)}), 500

    @app.get("/api/observatory/overview")
    def _observatory_overview():
        run_id = str(request.args.get("run_id", "") or "")
        return _guarded(lambda: observatory.overview(
            memory, knowledge, registry.live(), _run_for(run_id),
            events_limit=min(int(request.args.get("events", 40) or 40), 200)), "overview")

    @app.get("/api/observatory/runs")
    def _observatory_runs():
        return _guarded(lambda: observatory.runs(
            memory, registry.live(),
            limit=int(request.args.get("limit", 40) or 40)), "runs")

    @app.get("/api/observatory/pipeline")
    def _observatory_pipeline():
        run_id = str(request.args.get("run_id", "") or "")
        run = _run_for(run_id)
        return _guarded(lambda: observatory.pipeline(
            run, observatory.run_scope(memory, run)), "pipeline")

    @app.get("/api/observatory/agents")
    def _observatory_agents():
        run_id = str(request.args.get("run_id", "") or "")
        return _guarded(lambda: observatory.agents(
            memory, knowledge, _run_for(run_id)), "agents")

    @app.get("/api/observatory/events")
    def _observatory_events():
        run_id = str(request.args.get("run_id", "") or "")
        limit = int(request.args.get("limit", 60) or 60)
        trace = str(request.args.get("trace", "")).lower() in ("1", "true", "yes")
        return _guarded(lambda: observatory.events(
            memory, [_run_for(run_id)], limit, run_id=run_id, include_trace=trace), "events")

    @app.get("/api/observatory/hypotheses")
    def _observatory_hypotheses():
        run_id = str(request.args.get("run_id", "") or "")
        return _guarded(lambda: observatory.hypotheses_view(
            memory, _run_for(run_id),
            limit=int(request.args.get("limit", 60) or 60),
            state=str(request.args.get("state", "") or ""),
            direction_id=str(request.args.get("direction_id", "") or "")), "hypotheses")

    @app.get("/api/observatory/hypotheses/<hypothesis_id>")
    def _observatory_hypothesis(hypothesis_id: str):
        run_id = str(request.args.get("run_id", "") or "")
        payload = observatory.hypothesis_detail(memory, hypothesis_id, _run_for(run_id))
        if payload.get("error"):
            return jsonify(payload), 404
        return jsonify(payload)

    @app.get("/api/observatory/experiments")
    def _observatory_experiments():
        run_id = str(request.args.get("run_id", "") or "")
        return _guarded(lambda: observatory.experiments_view(
            memory, _run_for(run_id),
            limit=int(request.args.get("limit", 40) or 40),
            hypothesis_id=str(request.args.get("hypothesis_id", "") or ""),
            direction_id=str(request.args.get("direction_id", "") or ""),
            status=str(request.args.get("status", "") or "")), "experiments")

    @app.get("/api/observatory/experiments/<experiment_id>")
    def _observatory_experiment(experiment_id: str):
        run_id = str(request.args.get("run_id", "") or "")
        payload = observatory.experiment_detail(memory, experiment_id, _run_for(run_id))
        if payload.get("error"):
            return jsonify(payload), 404
        return jsonify(payload)

    @app.get("/api/observatory/thinking")
    def _observatory_thinking():
        run_id = str(request.args.get("run_id", "") or "")
        return _guarded(lambda: observatory.thinking(
            memory, knowledge, _run_for(run_id),
            layer_limit=int(request.args.get("layers", 28) or 28)), "thinking")

    @app.get("/api/observatory/graph")
    def _observatory_graph():
        run_id = str(request.args.get("run_id", "") or "")
        return _guarded(lambda: observatory.graph(
            memory, knowledge, _run_for(run_id)), "graph")

    @app.get("/api/observatory/report")
    def _observatory_report():
        run_id = str(request.args.get("run_id", "") or "")
        return _guarded(lambda: observatory.report(
            memory, knowledge, _run_for(run_id)), "report")

    @app.get("/api/observatory/architecture")
    def _observatory_architecture():
        run_id = str(request.args.get("run_id", "") or "")
        return _guarded(lambda: observatory.architecture(
            memory, knowledge, _run_for(run_id)), "architecture")

    # ------------------------------------------------------------------ static
    @app.get("/<path:filename>")
    def _static_files(filename: str):
        return send_from_directory(UI_DIR, filename)

    return app


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="AARL Web API")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--memory", default="research_memory")
    args = parser.parse_args()
    app = create_app(args.memory)
    print("AARL API on http://%s:%d" % (args.host, args.port))
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()




