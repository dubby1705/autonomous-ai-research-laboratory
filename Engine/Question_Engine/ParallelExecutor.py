"""
ParallelExecutor.py — Shared Parallel Processing Layer for AARL
==============================================================
One small, dependency-free abstraction so *every* stage of the AARL pipeline
can fan work out over all available CPU cores / server vCPUs.

Design goals
------------
* **Safe by default** — never crashes a research run. Any failure inside the
  parallel layer transparently degrades to the equivalent sequential result.
* **Deterministic** — results are returned in input order, so runs are still
  reproducible and identical to the sequential pipeline.
* **Nested-safe** — if a parallel section is invoked from inside a worker
  thread, the nested call executes inline. This avoids thread-pool deadlocks
  and unbounded thread growth when engines call each other.
* **Zero dependencies** — only the Python standard library
  (`concurrent.futures`), so it works on any server without new wheels.
* **Server friendly** — worker count auto-scales to the machine
  (`os.cpu_count()`) and can be pinned/disabled through environment variables
  or CLI flags.

Environment variables
---------------------
    AARL_PARALLEL=0|1          master on/off switch (default: 1 = enabled)
    AARL_PROCESSING=parallel|sequential   processing style: parallel fan-out
                               or one-by-one sequential (default: parallel)
    AARL_MAX_WORKERS=<int>     worker cap (default: cpu_count)
    AARL_PARALLEL_MODE=thread|process   execution backend (default: thread)
    AARL_PARALLEL_BATCH=<int>  optional batch size for huge job lists

Typical use
-----------
    from ParallelExecutor import parallel_map, run_parallel, tprint

    results  = parallel_map(score_claim, claims)           # ordered list
    outputs  = run_parallel({"evidence": f1, "maths": f2})  # dict of results
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# =========================================================
# SINGLE-INSTANCE ALIASING ACROSS IMPORT PATHS
# =========================================================
# Some modules import this file as a top-level module (``ParallelExecutor``)
# while others import it through the package path
# (``Engine.Question_Engine.ParallelExecutor``). Without aliasing, Python
# would create TWO module instances, each with its own config object and
# worker pools, so a runtime switch (e.g. the interactive prompt in AARL_run)
# would not reach importers that resolved the other name. Registering this
# module under both names guarantees one shared instance process-wide,
# whichever import path resolves first. (Skipped under ``python
# ParallelExecutor.py`` — a script has no second import path to unify.)
if __name__ != "__main__":
    for _alias in ("ParallelExecutor", "Engine.Question_Engine.ParallelExecutor"):
        if _alias != __name__:
            sys.modules.setdefault(_alias, sys.modules[__name__])

__all__ = [
    "ParallelConfig",
    "get_parallel_config",
    "set_parallel_config",
    "reset_parallel_config",
    "describe_parallelism",
    "parallel_map",
    "parallel_starmap",
    "run_parallel",
    "parallel_json_load",
    "parallel_copy_files",
    "tprint",
    "is_parallel_enabled",
    "resolve_workers",
    "llm_workers",
    "processing_style",
    "use_parallel",
    "is_failed",
    "failed_error",
]

# =========================================================
# THREAD-SAFE PRINTING
# =========================================================
_PRINT_LOCK = threading.Lock()


def tprint(*args: Any, **kwargs: Any) -> None:
    """
    Thread-safe ``print`` — prevents garbled interleaved console output.

    Also degrades gracefully when the active stdout code page cannot encode
    the message (common on Windows/servers running a cp1252 locale), so a
    worker thread can never crash the run just because it logged an emoji.
    """
    with _PRINT_LOCK:
        try:
            print(*args, **kwargs)
            sys.stdout.flush()
            return
        except UnicodeEncodeError:
            pass
        except Exception:
            return
        # ---- ASCII-safe fallback ---------------------------------------
        try:
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            safe = " ".join(str(a) for a in args)
            safe = safe.encode(encoding, "replace").decode(encoding, "replace")
            print(safe, **kwargs)
            sys.stdout.flush()
        except Exception:
            pass


# =========================================================
# CONFIGURATION
# =========================================================
_ENV_ENABLED = "AARL_PARALLEL"
_ENV_WORKERS = "AARL_MAX_WORKERS"
_ENV_MODE = "AARL_PARALLEL_MODE"
_ENV_BATCH = "AARL_PARALLEL_BATCH"
_ENV_STYLE = "AARL_PROCESSING"

# Accepted spellings for the two processing styles. Anything else resolves to
# the default ("parallel"), so a typo can never silently slow a run down.
_SEQUENTIAL_STYLES = {
    "sequential", "seq", "serial", "normal", "one", "one-by-one", "onebyone",
    "1by1", "single", "single-thread", "single-threaded", "sync", "off",
}
_PARALLEL_STYLES = {
    "parallel", "par", "multi", "concurrent", "fanout", "fan-out", "on",
}

_TRUTHY = {"1", "true", "yes", "on", "y", "enabled"}
_FALSY = {"0", "false", "no", "off", "n", "disabled", ""}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    token = raw.strip().lower()
    if token in _TRUTHY:
        return True
    if token in _FALSY:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


@dataclass
class ParallelConfig:
    """Resolved parallelism settings for the current process."""

    enabled: bool = True
    max_workers: int = 0        # 0 => auto (os.cpu_count())
    mode: str = "thread"        # "thread" (I/O bound) or "process" (CPU bound)
    batch_size: int = 0         # 0 => no batching
    min_items: int = 2          # do not spin up a pool for 0/1 item
    style: str = "parallel"     # "parallel" fan-out or "sequential" one-by-one

    @property
    def sequential(self) -> bool:
        """True when the operator chose the one-by-one processing style."""
        return str(self.style).strip().lower() in _SEQUENTIAL_STYLES

    @property
    def cpu_workers(self) -> int:
        """Resolved worker count (never below 1)."""
        if self.max_workers and self.max_workers > 0:
            return max(1, int(self.max_workers))
        return max(1, os.cpu_count() or 4)

    def workers_for(self, n_items: int) -> int:
        """Workers actually used for a job of ``n_items`` tasks."""
        if self.sequential or not self.enabled or n_items < self.min_items:
            return 1
        return max(1, min(int(n_items), self.cpu_workers))

    def describe(self) -> str:
        if self.sequential:
            return (
                "sequential (one-by-one) | parallel fan-out disabled "
                "(AARL_PROCESSING=sequential; set to 'parallel' to enable, "
                f"cpu_count={os.cpu_count() or 'unknown'})"
            )
        if not self.enabled:
            return "disabled (sequential)"
        return (
            f"parallel | mode={self.mode} | workers={self.cpu_workers} "
            f"(cpu_count={os.cpu_count() or 'unknown'})"
            + (f" | batch={self.batch_size}" if self.batch_size else "")
        )
# Module-level active configuration. Lazily built from the environment so CLI
# flags (which export env vars before the pipeline imports) are respected.
_config: Optional[ParallelConfig] = None
_config_lock = threading.Lock()


def _build_config_from_env() -> ParallelConfig:
    mode = os.environ.get(_ENV_MODE, "thread").strip().lower()
    if mode not in ("thread", "process"):
        mode = "thread"
    style = os.environ.get(_ENV_STYLE, "parallel").strip().lower()
    if style in _SEQUENTIAL_STYLES:
        style = "sequential"
    elif style not in _PARALLEL_STYLES:
        style = "parallel"
    return ParallelConfig(
        enabled=_env_flag(_ENV_ENABLED, True),
        max_workers=max(0, _env_int(_ENV_WORKERS, 0)),
        mode=mode,
        batch_size=max(0, _env_int(_ENV_BATCH, 0)),
        style=style,
    )


def get_parallel_config() -> ParallelConfig:
    """Return the process-wide :class:`ParallelConfig` (env-derived once)."""
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = _build_config_from_env()
    return _config


def reset_parallel_config() -> None:
    """Force a re-read of the environment on the next access (used by tests)."""
    global _config
    with _config_lock:
        _config = None
    _shutdown_pools()


def set_parallel_config(
    enabled: Optional[bool] = None,
    max_workers: Optional[int] = None,
    mode: Optional[str] = None,
    batch_size: Optional[int] = None,
    style: Optional[str] = None,
) -> ParallelConfig:
    """Explicitly override parallelism settings at runtime."""
    global _config
    current = get_parallel_config()
    with _config_lock:
        _config = ParallelConfig(
            enabled=current.enabled if enabled is None else bool(enabled),
            max_workers=(current.max_workers if max_workers is None
                         else max(0, int(max_workers))),
            mode=(current.mode if mode is None else str(mode).lower()),
            batch_size=(current.batch_size if batch_size is None
                        else max(0, int(batch_size))),
            style=(current.style if style is None else str(style).strip().lower()),
            min_items=current.min_items,
        )
        if _config.mode not in ("thread", "process"):
            _config.mode = "thread"
        if _config.style in _SEQUENTIAL_STYLES:
            _config.style = "sequential"
        elif _config.style not in _PARALLEL_STYLES:
            _config.style = "parallel"
    _shutdown_pools()
    return _config


def is_parallel_enabled() -> bool:
    cfg = get_parallel_config()
    return cfg.enabled and not cfg.sequential


def processing_style() -> str:
    """Return the active processing style: ``"parallel"`` or ``"sequential"``."""
    return "sequential" if get_parallel_config().sequential else "parallel"


def use_parallel(n_items: Optional[int] = None) -> bool:
    """
    True when fan-out is currently allowed.

    This is the single gate every engine consults before dispatching work, so
    the operator's parallel / one-by-one choice applies to the whole pipeline:

    * ``AARL_PROCESSING=sequential`` (or ``--normal``) forces one-by-one;
    * fewer than ``min_items`` items never spins up a pool;
    * ``AARL_PARALLEL=0`` disables fan-out entirely.
    """
    cfg = get_parallel_config()
    if cfg.sequential or not cfg.enabled:
        return False
    if n_items is not None and n_items < cfg.min_items:
        return False
    return True


def resolve_workers(n_items: int, max_workers: Optional[int] = None) -> int:
    """Public helper mirroring :meth:`ParallelConfig.workers_for`."""
    cfg = get_parallel_config()
    if cfg.sequential:
        return 1
    if max_workers is not None and max_workers > 0:
        if not cfg.enabled or n_items < cfg.min_items:
            return 1
        return max(1, min(int(n_items), int(max_workers)))
    return cfg.workers_for(n_items)


def describe_parallelism() -> str:
    """One-line human readable summary, used in the pipeline banner."""
    return get_parallel_config().describe()


def is_failed(value: Any) -> bool:
    """
    True when ``value`` is the failure sentinel produced by :func:`run_parallel`
    or :func:`parallel_map`. Lets callers substitute a safe fallback instead of
    propagating an exception object.
    """
    return isinstance(value, _Failed)


def failed_error(value: Any) -> Optional[str]:
    """Return the captured error string of a sentinel, or ``None``."""
    error = getattr(value, "error", None)
    return error if isinstance(error, str) else None


# Concurrency cap for LLM-backed fan-out. Providers (Groq/OpenAI/Gemini) and
# local Ollama rate-limit aggressively, so we deliberately keep LLM fan-out
# below the raw CPU count unless the operator overrides it.
_ENV_LLM_WORKERS = "AARL_LLM_WORKERS"
DEFAULT_LLM_WORKERS = 8


def llm_workers(default: int = DEFAULT_LLM_WORKERS) -> int:
    """Worker cap for LLM-bound jobs (rate-limit friendly; env: AARL_LLM_WORKERS)."""
    cfg = get_parallel_config()
    if not cfg.enabled or cfg.sequential:
        return 1
    cap = _env_int(_ENV_LLM_WORKERS, default)
    if cap <= 0:
        cap = default
    return max(1, min(cap, cfg.cpu_workers))
# =========================================================
# SHARED POOL MANAGEMENT
# =========================================================
# Reusing one pool across the whole pipeline avoids paying pool-creation cost
# for every phase (there are dozens of fan-out points) and keeps the total
# thread count bounded on a server instead of spawning a pool per phase.
_pools: Dict[Tuple[str, int], Any] = {}
_pools_lock = threading.Lock()

# Tracks whether the current thread is already a pool worker. Nested fan-out
# runs inline instead of queueing behind itself (deadlock / starvation guard).
_local = threading.local()


def _in_worker() -> bool:
    return bool(getattr(_local, "depth", 0) > 0)


def _mark_worker_enter() -> None:
    _local.depth = getattr(_local, "depth", 0) + 1


def _mark_worker_exit() -> None:
    _local.depth = max(0, getattr(_local, "depth", 1) - 1)


def _get_pool(mode: str, workers: int):
    """Return (or lazily create) the shared executor for mode/worker count."""
    key = (mode, int(workers))
    with _pools_lock:
        pool = _pools.get(key)
        if pool is None:
            if mode == "process":
                pool = ProcessPoolExecutor(max_workers=int(workers))
            else:
                pool = ThreadPoolExecutor(
                    max_workers=int(workers),
                    thread_name_prefix="aarl",
                )
            _pools[key] = pool
        return pool


def _shutdown_pools() -> None:
    """Shut every shared pool down (called on config change and at exit)."""
    with _pools_lock:
        pools = list(_pools.values())
        _pools.clear()
    for pool in pools:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # Python < 3.9 has no cancel_futures
            try:
                pool.shutdown(wait=False)
            except Exception:
                pass
        except Exception:
            pass


atexit.register(_shutdown_pools)


class _Failed:
    """Internal sentinel: a worker raised but errors are being swallowed.

    The error is stored as a *string* so the sentinel stays picklable and can
    safely travel back from a process-pool worker.
    """

    __slots__ = ("error",)

    def __init__(self, error: Any):
        self.error = (
            error if isinstance(error, str)
            else f"{type(error).__name__}: {error}"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<_Failed {self.error}>"


def _process_call(payload: Tuple[int, Callable, Any]) -> Tuple[int, Any, Any]:
    """Module-level worker body so process-mode tasks stay picklable."""
    idx, fn, item = payload
    try:
        return idx, fn(item), None
    except Exception as exc:  # noqa: BLE001
        return idx, None, _Failed(exc)


def _thread_call(pair: Tuple[int, Any], fn: Callable, swallow: bool):
    """Worker body for thread mode (closures are fine inside one process)."""
    idx, item = pair
    _mark_worker_enter()
    try:
        return idx, fn(item), None
    except Exception as exc:  # noqa: BLE001
        if not swallow:
            raise
        return idx, None, _Failed(exc)
    finally:
        _mark_worker_exit()
# =========================================================
# CORE: ORDERED PARALLEL MAP
# =========================================================
def parallel_map(
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    max_workers: Optional[int] = None,
    mode: Optional[str] = None,
    ordered: bool = True,
    default: Any = None,
    swallow_errors: bool = True,
    label: Optional[str] = None,
) -> List[Any]:
    """
    Apply ``fn`` to every item of ``items`` using all available workers.

    Always returns a list with **one entry per input item** (same order by
    default). Exceptions are captured and replaced by ``default`` when
    ``swallow_errors`` is True so one bad item can never kill a research run.

    Falls back to a plain sequential loop when parallelism is disabled, when
    there are fewer than ``min_items`` items, or when the executor cannot be
    created / a worker raises unexpectedly.
    """
    if not isinstance(items, (list, tuple)):
        items = list(items)
    n = len(items)
    if n == 0:
        return []

    cfg = get_parallel_config()
    resolved_mode = (mode or cfg.mode or "thread").lower()
    if resolved_mode not in ("thread", "process"):
        resolved_mode = "thread"
    workers = resolve_workers(n, max_workers)

    # ---- sequential fast path (also the nested-call guard) ---------------
    if workers <= 1 or _in_worker():
        out: List[Any] = []
        for item in items:
            try:
                out.append(fn(item))
            except Exception as exc:  # noqa: BLE001
                if not swallow_errors:
                    raise
                if label:
                    tprint(f"   [Parallel:{label}] item failed: {exc}")
                out.append(default)
        return out

    # ---- parallel path ---------------------------------------------------
    indexed = list(enumerate(items))
    batch = max(0, cfg.batch_size)

    try:
        if batch and n > batch:
            # Streaming/batched mode keeps memory flat for very large jobs.
            results: List[Any] = [default] * n
            for start in range(0, n, batch):
                chunk = indexed[start:start + batch]
                results[start:start + batch] = _execute_ordered(
                    fn, chunk, resolved_mode, workers, ordered, default,
                    swallow_errors, label,
                )
            return results
        return _execute_ordered(
            fn, indexed, resolved_mode, workers, ordered, default,
            swallow_errors, label,
        )
    except Exception as exc:  # noqa: BLE001 — never let the pool kill a run
        tprint(f"   [Parallel] pool unavailable ({exc}); running sequentially")
        out = []
        for item in items:
            try:
                out.append(fn(item))
            except Exception as inner:  # noqa: BLE001
                if not swallow_errors:
                    raise
                out.append(default)
        return out


def _execute_ordered(
    fn: Callable[[Any], Any],
    indexed: Sequence[Tuple[int, Any]],
    mode: str,
    workers: int,
    ordered: bool,
    default: Any,
    swallow_errors: bool,
    label: Optional[str],
) -> List[Any]:
    """Run one pool batch and return results in input order."""
    n = len(indexed)
    pool = _get_pool(mode, workers)
    results: List[Any] = [default] * n

    # Materialise immediately — generators can only be walked once.
    if mode == "process":
        # Process workers must be module-level + picklable.
        payloads = [(idx, fn, item) for idx, item in indexed]
        collected = list(
            pool.map(_process_call, payloads) if ordered else (
                future.result() for future in as_completed(
                    [pool.submit(_process_call, p) for p in payloads]
                )
            )
        )
    elif ordered:
        collected = list(
            pool.map(lambda pair: _thread_call(pair, fn, swallow_errors), indexed)
        )
    else:
        collected = [
            future.result()
            for future in as_completed([
                pool.submit(_thread_call, pair, fn, swallow_errors)
                for pair in indexed
            ])
        ]

    failures = 0
    first_error: Optional[str] = None
    for _, _value, failure in collected:
        if failure is None:
            continue
        failures += 1
        if first_error is None:
            first_error = failure.error

    if failures and not swallow_errors:
        raise RuntimeError(f"[Parallel] {failures} worker(s) failed: {first_error}")

    if not ordered:
        collected = sorted(collected, key=lambda row: row[0])

    for idx, value, failure in collected:
        if idx < n:
            results[idx] = default if failure is not None else value

    if failures and label:
        tprint(
            f"   [Parallel:{label}] {failures}/{n} item(s) failed; "
            f"using fallback values. First error: {first_error}"
        )
    return results
# =========================================================
# STARMAP — multi-argument fan-out
# =========================================================
def _unpack_call(payload: Tuple[Callable, Sequence[Any], Dict[str, Any]]) -> Any:
    """Module-level helper so starmap jobs stay picklable in process mode."""
    fn, args, kwargs = payload
    return fn(*args, **kwargs)


def parallel_starmap(
    fn: Callable[..., Any],
    args_list: Sequence[Sequence[Any]],
    max_workers: Optional[int] = None,
    mode: Optional[str] = None,
    default: Any = None,
    swallow_errors: bool = True,
    label: Optional[str] = None,
) -> List[Any]:
    """Like :func:`parallel_map` but each job receives several arguments."""
    jobs: List[Tuple[Callable, Sequence[Any], Dict[str, Any]]] = []
    for entry in args_list:
        if isinstance(entry, tuple):
            jobs.append((fn, entry, {}))
        elif isinstance(entry, list):
            jobs.append((fn, tuple(entry), {}))
        elif isinstance(entry, dict):
            jobs.append((fn, (), entry))
        else:
            jobs.append((fn, (entry,), {}))
    return parallel_map(
        _unpack_call, jobs,
        max_workers=max_workers, mode=mode,
        default=default, swallow_errors=swallow_errors, label=label,
    )


# =========================================================
# PHASE-LEVEL CONCURRENCY
# =========================================================
def run_parallel(
    tasks: Dict[str, Callable[[], Any]],
    max_workers: Optional[int] = None,
    swallow_errors: bool = True,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run several *independent* callables at the same time and return
    ``{name: result}``.

    Use this for genuinely independent pipeline stages (e.g. "write the
    markdown report" and "render the plots", or "score evidence" and "derive
    equations"). Failures are captured per task — a task that raises yields a
    ``_Failed`` sentinel (and is reported to the console) instead of aborting
    the run.

    Threads are always used here: these stages are I/O bound (disk, network)
    and threads share the loaded knowledge base / numpy arrays without the
    pickling cost of separate processes. Only directly-executed callables are
    supported (they are not picklable), which is exactly the phase-level case.
    """
    names = list(tasks.keys())
    if not names:
        return {}

    workers = resolve_workers(len(names), max_workers)
    results: Dict[str, Any] = {}

    # ---- sequential fast path (also the nested-call guard) ---------------
    if workers <= 1 or _in_worker():
        for name in names:
            try:
                results[name] = tasks[name]()
            except Exception as exc:  # noqa: BLE001
                if not swallow_errors:
                    raise
                tprint(f"   [Parallel:{label or 'stage'}] '{name}' failed: {exc}")
                results[name] = _Failed(exc)
        return results

    pool = _get_pool("thread", workers)
    futures = {pool.submit(tasks[name]): name for name in names}
    for future in as_completed(futures):
        name = futures[future]
        try:
            results[name] = future.result()
        except Exception as exc:  # noqa: BLE001
            if not swallow_errors:
                raise
            tprint(f"   [Parallel:{label or 'stage'}] '{name}' failed: {exc}")
            results[name] = _Failed(exc)

    # Return in the caller-declared order so downstream code is deterministic.
    return {name: results[name] for name in names}


# =========================================================
# BULK FILE I/O HELPERS
# =========================================================
def parallel_json_load(
    paths: Sequence[str],
    default: Any = None,
) -> List[Any]:
    """Load many JSON files concurrently; missing/broken files yield ``default``."""
    import json as _json

    def _load_one(path: str) -> Any:
        if not path or not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return _json.load(fh)
        except Exception:
            return default

    return parallel_map(_load_one, list(paths), default=default, label="json-load")


def parallel_copy_files(
    pairs: Sequence[Tuple[str, str]],
) -> int:
    """Copy many files at once. Returns the number copied successfully."""
    import shutil

    def _copy_one(pair: Tuple[str, str]) -> bool:
        src, dst = pair
        if not src or not os.path.exists(src):
            return False
        try:
            dst_dir = os.path.dirname(dst)
            if dst_dir:
                os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except Exception:
            return False

    outcomes = parallel_map(_copy_one, list(pairs), default=False, label="file-copy")
    return sum(1 for ok in outcomes if ok)
# =========================================================
# SELF-TEST (python ParallelExecutor.py)
# =========================================================
if __name__ == "__main__":
    import json
    import time as _time

    print("=" * 70)
    print("AARL ParallelExecutor self-test")
    print("=" * 70)
    print(f"Configuration: {describe_parallelism()}")
    print("-" * 70)

    # 1. Correctness + ordering
    def _square(x: int) -> int:
        return x * x

    expect = [i * i for i in range(12)]
    got = parallel_map(_square, list(range(12)))
    assert got == expect, f"parallel_map ordering broken: {got}"
    print(f"[1/5] parallel_map ordered .......... OK ({len(got)} items)")

    # 2. Error isolation
    def _flaky(x: int) -> int:
        if x == 3:
            raise ValueError("boom")
        return x

    got = parallel_map(_flaky, list(range(6)), default=-1, label="flaky")
    assert got == [0, 1, 2, -1, 4, 5], got
    print("[2/5] error isolation ............... OK (bad item -> default)")

    # 3. Speed-up on a simulated I/O bound job
    def _sleep_ms(_: int) -> int:
        _time.sleep(0.05)
        return 1

    jobs = list(range(16))
    t0 = _time.time()
    parallel_map(_sleep_ms, jobs)
    par = _time.time() - t0
    t0 = _time.time()
    [_sleep_ms(j) for j in jobs]
    seq = _time.time() - t0
    print(f"[3/5] speed-up ...................... {seq:.2f}s -> {par:.2f}s "
          f"({seq / max(par, 1e-9):.1f}x)")

    # 4. Stage-level concurrency
    def _stage_a() -> str:
        _time.sleep(0.05)
        return "A"

    def _stage_b() -> str:
        _time.sleep(0.05)
        return "B"

    out = run_parallel({"a": _stage_a, "b": _stage_b}, label="selftest")
    assert out == {"a": "A", "b": "B"}, out
    print(f"[4/5] run_parallel stages ........... OK {out}")

    # 5. Bulk JSON loading
    tmp = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_parallel_selftest.json"
    )
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"ok": True}, fh)
    loaded = parallel_json_load([tmp, tmp, "does_not_exist.json"])
    os.remove(tmp)
    assert loaded[0] == {"ok": True} and loaded[2] is None, loaded
    print("[5/5] parallel_json_load ............ OK")

    print("-" * 70)
    print("All ParallelExecutor self-tests passed.")