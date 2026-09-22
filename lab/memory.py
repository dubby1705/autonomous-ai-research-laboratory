"""
AARL Lab — Persistent Research Memory
=====================================
Append-only, human-readable research memory. **Nothing is ever overwritten.**

Layout (all paths are relative to ``LabConfig.memory_dir``)::

    research_memory/
        knowledge.json               the living knowledge base (KnowledgeManager)
        research_history.json        append-only cycle / iteration / event ledger
        graph.json                   research-graph export (reporting layer)
        papers/                      PAPER-###   (reserved for the retrieval layer)
        hypotheses/                  HYP-###
        simulations/                 EXP-###     (Simulation JSON)
        failures/                    FAIL-###
        lessons/                     LESSON-###
        real_world_experiments/      RWE-### / RWR-###  (proposal + human result)
        custom_simulators/           user drop-in simulation adapters

Design rules
------------
* **Append-only.** Re-writing an existing id never destroys the old record: the
  new content is stored as ``<id>.revN.json`` and the revision is logged.
* **Idempotent.** Writing byte-identical content again is a no-op.
* **Atomic.** Every file is written to a temp file and ``os.replace``-d, so a
  crash can never leave a half-written record.
* **Provenance.** Each record keeps its own ``provenance`` block; the memory
  additionally exposes a derived index (id -> collection/file/links).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("aarl.lab.memory")

#: Record collections that live as one JSON file per record.
#:
#: ``papers`` … ``real_world_experiments`` are the original research-loop
#: collections. ``directions`` (discovered research directions) and
#: ``explorations`` (GUI exploration runs) were added by the discovery layer;
#: both are ordinary append-only collections, so nothing about the original
#: behaviour changes.
COLLECTIONS = (
    "papers",
    "hypotheses",
    "simulations",
    "failures",
    "lessons",
    "real_world_experiments",
    "directions",
    "explorations",
)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class ResearchMemory:
    """Append-only on-disk research memory."""

    def __init__(self, root: str = "research_memory") -> None:
        self.root = os.path.abspath(root)
        self.knowledge_path = os.path.join(self.root, "knowledge.json")
        self.history_path = os.path.join(self.root, "research_history.json")
        self.graph_path = os.path.join(self.root, "graph.json")
        self.custom_simulator_dir = os.path.join(self.root, "custom_simulators")
        self.ensure_layout()

    # ------------------------------------------------------------------ setup
    def ensure_layout(self) -> None:
        """Create the directory layout (idempotent)."""
        os.makedirs(self.root, exist_ok=True)
        for collection in COLLECTIONS:
            os.makedirs(self.collection_dir(collection), exist_ok=True)
        os.makedirs(self.custom_simulator_dir, exist_ok=True)

    def collection_dir(self, collection: str) -> str:
        if collection not in COLLECTIONS:
            raise ValueError(f"unknown memory collection '{collection}'")
        return os.path.join(self.root, collection)

    def path_for(self, collection: str, record_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(record_id))
        return os.path.join(self.collection_dir(collection), f"{safe}.json")

    # ------------------------------------------------------------------ ids
    def next_id(self, collection: str, prefix: str) -> str:
        """Next free ``PREFIX-NNN`` id for a collection (deterministic scan)."""
        directory = self.collection_dir(collection)
        pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)\.json$")
        highest = 0
        try:
            for name in os.listdir(directory):
                match = pattern.match(name)
                if match:
                    highest = max(highest, int(match.group(1)))
        except FileNotFoundError:  # pragma: no cover - created in __init__
            pass
        return f"{prefix}-{highest + 1:03d}"

    # ------------------------------------------------------------------ io
    @staticmethod
    def _atomic_write_json(path: str, data: Any, retries: int = 5) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, default=str)
        # On Windows os.replace can transiently fail with PermissionError when
        # another process (AV, indexer, a concurrent writer) holds the target
        # open. Retry with backoff before giving up.
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                os.replace(tmp, path)
                return path
            except PermissionError as exc:
                last_exc = exc
                time.sleep(0.1 * (2 ** attempt))
        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _read_json(path: str, default: Any = None) -> Any:
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:  # noqa: BLE001 - a corrupt record must not kill a run
            log.warning("Could not read %s (%s)", path, exc)
            return default
    # ------------------------------------------------------------------ records
    def append_record(
        self,
        collection: str,
        record_id: str,
        payload: Dict[str, Any],
        revision_of: Optional[str] = None,
    ) -> str:
        """Store one record without ever destroying an earlier version.

        Returns the path actually written. If the id already exists with
        different content the new version is written as ``<id>.revN.json`` and
        the revision is recorded in the research history.
        """
        path = self.path_for(collection, record_id)
        body = dict(payload)
        body.setdefault("record_id", record_id)
        body.setdefault("collection", collection)
        body.setdefault("stored_at", _now())

        if not os.path.exists(path):
            return self._atomic_write_json(path, body)

        existing = self._read_json(path, default=None)
        if existing == body:
            return path  # idempotent re-write

        revision = 2
        while os.path.exists(self.path_for(collection, f"{record_id}.rev{revision}")):
            revision += 1
        revision_id = f"{record_id}.rev{revision}"
        rev_path = self._atomic_write_json(self.path_for(collection, revision_id), body)
        self.append_history(
            {
                "event": "record_revision",
                "collection": collection,
                "record_id": record_id,
                "revision_id": revision_id,
                "supersedes": path,
                "revision_of": revision_of or "prior version",
                "file": rev_path,
            }
        )
        return rev_path

    def get(self, collection: str, record_id: str) -> Optional[Dict[str, Any]]:
        return self._read_json(self.path_for(collection, record_id), default=None)

    def load_collection(self, collection: str) -> List[Dict[str, Any]]:
        """Load every live (non-revision) record of a collection, ordered."""
        directory = self.collection_dir(collection)
        records: List[Dict[str, Any]] = []
        try:
            names = sorted(os.listdir(directory))
        except FileNotFoundError:  # pragma: no cover
            return records
        for name in names:
            if not name.endswith(".json") or ".rev" in name:
                continue
            body = self._read_json(os.path.join(directory, name), default=None)
            if isinstance(body, dict):
                records.append(body)
        records.sort(key=lambda r: str(r.get("record_id", "")))
        return records

    # ------------------------------------------------------------------ knowledge
    def load_knowledge(self) -> Dict[str, Any]:
        """Return the living knowledge store (empty skeleton when absent)."""
        store = self._read_json(self.knowledge_path, default=None)
        return store if isinstance(store, dict) else {}

    def save_knowledge(self, store: Dict[str, Any]) -> str:
        """Persist the knowledge store, snapshotted per write (never destructive)."""
        before = self._read_json(self.knowledge_path, default=None)
        if isinstance(before, dict) and before:
            revision = 2
            while os.path.exists(os.path.join(self.root, f"knowledge.rev{revision}.json")):
                revision += 1
            self._atomic_write_json(
                os.path.join(self.root, f"knowledge.rev{revision}.json"), before
            )
        return self._atomic_write_json(self.knowledge_path, store)

    # ------------------------------------------------------------------ history
    def append_history(self, event: Dict[str, Any]) -> str:
        """Append one event to the research history ledger."""
        ledger = self._read_json(self.history_path, default=None)
        if not isinstance(ledger, dict):
            ledger = {"created_at": _now(), "events": []}
        entry = dict(event)
        entry.setdefault("timestamp", _now())
        ledger.setdefault("events", [])
        ledger["events"].append(entry)
        ledger["last_updated"] = entry["timestamp"]
        return self._atomic_write_json(self.history_path, ledger)

    def load_history(self) -> List[Dict[str, Any]]:
        ledger = self._read_json(self.history_path, default=None)
        if isinstance(ledger, dict):
            events = ledger.get("events")
            return list(events) if isinstance(events, list) else []
        return []


# ------------------------------------------------------------------ derived
    def link_ids(self, payload: Dict[str, Any]) -> List[str]:
        """Extract the AARL record ids referenced by a stored record."""
        found: List[str] = []
        pattern = re.compile(
            r"\b(?:PAPER|HYP|EXP|FAIL|LESSON|RWE|RWR|DIR|RUN)-[0-9]+\b"
        )
        for value in payload.values():
            if isinstance(value, str):
                found.extend(pattern.findall(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        found.extend(pattern.findall(item))
                    elif isinstance(item, dict):
                        found.extend(self.link_ids(item))
            elif isinstance(value, dict):
                found.extend(self.link_ids(value))
        return sorted(set(found))

    def record_index(self) -> Dict[str, Any]:
        """Derived index: id -> {collection, file, links} for retrieval."""
        index: Dict[str, Any] = {}
        for collection in COLLECTIONS:
            for record in self.load_collection(collection):
                record_id = str(record.get("record_id", ""))
                if not record_id:
                    continue
                index[record_id] = {
                    "collection": collection,
                    "file": self.path_for(collection, record_id),
                    "links": self.link_ids(record),
                }
        return index

    def provenance(self, record_id: str) -> Dict[str, Any]:
        """Where a record lives, what it links to, and what links back to it."""
        index = self.record_index()
        entry = index.get(record_id)
        if entry is None:
            return {
                "record_id": record_id,
                "found": False,
                "collection": None,
                "file": None,
            }
        return {
            "record_id": record_id,
            "found": True,
            "collection": entry["collection"],
            "file": entry["file"],
            "links_to": [rid for rid in entry["links"] if rid != record_id],
            "linked_from": sorted(
                rid
                for rid, meta in index.items()
                if record_id in meta.get("links", []) and rid != record_id
            ),
        }

    def stats(self) -> Dict[str, int]:
        """Counts per collection plus the knowledge-section sizes."""
        counts = {
            collection: len(self.load_collection(collection)) for collection in COLLECTIONS
        }
        store = self.load_knowledge()
        items = store.get("items")
        counts["knowledge_items"] = len(items) if isinstance(items, list) else 0
        counts["history_events"] = len(self.load_history())
        return counts