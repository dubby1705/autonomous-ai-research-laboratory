"""
OllamaManager.py — Robust Ollama Integration for AARL
=====================================================
Complete Ollama lifecycle management:
  - Detect whether Ollama is installed
  - Detect whether the Ollama service is running
  - Auto-start Ollama if possible
  - Discover every installed model
  - Allow user to select any installed model
  - Support streaming responses
  - Handle timeouts
  - Handle connection failures
  - Retry gracefully
  - Never crash the pipeline because Ollama is unavailable

This module is a drop-in replacement for the old OllamaClient.py
and is used by GroqClient.py for the Ollama-first strategy.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
import threading
from typing import Optional, Dict, Any, List, Generator, Tuple

# =========================================================
# CONSTANTS
# =========================================================
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_LOCALHOST = "http://localhost:11434"
OLLAMA_API_TAGS = "/api/tags"
OLLAMA_API_GENERATE = "/api/generate"
OLLAMA_API_CHAT = "/api/chat"
OLLAMA_API_SHOW = "/api/show"
OLLAMA_API_PULL = "/api/pull"

DEFAULT_TIMEOUT = 30          # Connection timeout (seconds)
DEFAULT_GENERATE_TIMEOUT = 120  # Generation timeout (seconds)
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5           # Exponential backoff multiplier

# Default model preference order (fallback if no models found)
DEFAULT_MODEL_PREFERENCES = [
    "llama3.2", "llama3.1", "llama3", "phi3:mini", "phi3",
    "mistral", "gemma2", "qwen2.5", "deepseek-r1", "codellama",
]

# =========================================================
# STATE
# =========================================================
_ollama_checked = False
_ollama_available = False
_ollama_installed = None  # None = unknown, True/False
_ollama_models: List[str] = []
_ollama_models_loaded = False
_selected_model: Optional[str] = None
_start_attempted = False
_lock = threading.Lock()


# =========================================================
# INSTALLATION DETECTION
# =========================================================
def is_ollama_installed() -> bool:
    """Check whether the Ollama binary is installed on this system."""
    global _ollama_installed
    if _ollama_installed is not None:
        return _ollama_installed

    # Check PATH for ollama executable
    ollama_bin = shutil.which("ollama")
    if ollama_bin:
        _ollama_installed = True
        return True

    # Check common install locations
    common_paths = []
    if sys.platform == "win32":
        common_paths = [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
            os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe"),
            r"C:\Program Files\Ollama\ollama.exe",
        ]
    elif sys.platform == "darwin":
        common_paths = [
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            "/Applications/Ollama.app/Contents/Resources/ollama",
        ]
    else:  # Linux
        common_paths = [
            "/usr/local/bin/ollama",
            "/usr/bin/ollama",
            "/snap/bin/ollama",
        ]

    for path in common_paths:
        if os.path.exists(path):
            _ollama_installed = True
            return True

    _ollama_installed = False
    return False


def get_ollama_binary() -> Optional[str]:
    """Return the path to the ollama binary, or None if not installed."""
    ollama_bin = shutil.which("ollama")
    if ollama_bin:
        return ollama_bin

    if sys.platform == "win32":
        common_paths = [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
            os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe"),
            r"C:\Program Files\Ollama\ollama.exe",
        ]
    elif sys.platform == "darwin":
        common_paths = [
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            "/Applications/Ollama.app/Contents/Resources/ollama",
        ]
    else:
        common_paths = [
            "/usr/local/bin/ollama",
            "/usr/bin/ollama",
            "/snap/bin/ollama",
        ]

    for path in common_paths:
        if os.path.exists(path):
            return path
    return None


# =========================================================
# SERVICE DETECTION
# =========================================================
def _http_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[Dict[str, Any]]:
    """Perform a GET request, returning parsed JSON or None on failure."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError):
        pass
    return None


def _http_post(url: str, payload: Dict[str, Any], timeout: int = DEFAULT_GENERATE_TIMEOUT,
               stream: bool = False) -> Optional[Any]:
    """Perform a POST request. Returns parsed JSON or None on failure."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError):
        pass
    return None


def is_ollama_running() -> bool:
    """Check whether the Ollama service is currently reachable."""
    global _ollama_available, _ollama_checked
    with _lock:
        if _ollama_checked:
            return _ollama_available

        # Try 127.0.0.1 first, then localhost
        for base in (OLLAMA_HOST, OLLAMA_LOCALHOST):
            result = _http_get(f"{base}{OLLAMA_API_TAGS}", timeout=5)
            if result is not None:
                _ollama_available = True
                _ollama_checked = True
                return True

        _ollama_available = False
        _ollama_checked = True
        return False


def reset_ollama_check() -> None:
    """Reset the cached availability check (useful after starting Ollama)."""
    global _ollama_checked, _ollama_available, _ollama_models_loaded
    _ollama_checked = False
    _ollama_available = False
    _ollama_models_loaded = False


# =========================================================
# AUTO-START
# =========================================================
def start_ollama() -> bool:
    """Attempt to start the Ollama service automatically."""
    global _start_attempted
    if _start_attempted:
        return is_ollama_running()

    _start_attempted = True

    if not is_ollama_installed():
        return False

    ollama_bin = get_ollama_binary()
    if not ollama_bin:
        return False

    print("\n[OllamaManager] Ollama is installed but not running. Attempting to start...")

    try:
        if sys.platform == "win32":
            # On Windows, start ollama serve in a new process
            subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            # On Unix, start in background
            subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        # Wait for the service to become available (up to 15 seconds)
        for attempt in range(15):
            time.sleep(1)
            reset_ollama_check()
            if is_ollama_running():
                print("[OllamaManager] [OK] Ollama service started successfully.")
                return True

        print("[OllamaManager] [WARN] Ollama service did not become ready within 15 seconds.")
        return False

    except Exception as e:
        print(f"[OllamaManager] [ERROR] Failed to start Ollama: {e}")
        return False


# =========================================================
# MODEL DISCOVERY
# =========================================================
def discover_models(force_refresh: bool = False) -> List[str]:
    """Discover all installed Ollama models."""
    global _ollama_models, _ollama_models_loaded

    if _ollama_models_loaded and not force_refresh:
        return _ollama_models

    if not is_ollama_running():
        return []

    for base in (OLLAMA_HOST, OLLAMA_LOCALHOST):
        result = _http_get(f"{base}{OLLAMA_API_TAGS}", timeout=10)
        if result is not None:
            models = []
            for model_info in result.get("models", []):
                name = model_info.get("name", "")
                if name:
                    models.append(name)
            if models:
                _ollama_models = sorted(models)
                _ollama_models_loaded = True
                return _ollama_models

    return []


def get_available_models() -> List[str]:
    """Get the list of available models (cached)."""
    return discover_models()


def select_model_interactive() -> Optional[str]:
    """
    Allow the user to select any installed Ollama model.
    Returns the selected model name, or None if no models available.
    """
    global _selected_model

    models = discover_models()
    if not models:
        return None

    # If only one model, use it directly
    if len(models) == 1:
        _selected_model = models[0]
        print(f"[OllamaManager] Using only available model: {_selected_model}")
        return _selected_model

    print("\n[OllamaManager] Available Ollama models:")
    for i, model in enumerate(models, 1):
        print(f"  [{i}] {model}")

    try:
        choice = input(f"\nSelect model [1-{len(models)}] (default: 1): ").strip()
        if not choice:
            idx = 0
        else:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(models):
                idx = 0
        _selected_model = models[idx]
        print(f"[OllamaManager] Selected model: {_selected_model}")
        return _selected_model
    except (EOFError, ValueError):
        # Non-interactive mode — pick the first model
        _selected_model = models[0]
        print(f"[OllamaManager] Non-interactive mode. Using model: {_selected_model}")
        return _selected_model


def auto_select_model() -> Optional[str]:
    """
    Automatically select the best available model without user interaction.
    Uses preference order, falling back to the first available model.
    """
    global _selected_model

    if _selected_model:
        return _selected_model

    models = discover_models()
    if not models:
        return None

    # Try preference order first
    for preferred in DEFAULT_MODEL_PREFERENCES:
        for model in models:
            if model.startswith(preferred):
                _selected_model = model
                return _selected_model

    # Fallback: first model
    _selected_model = models[0]
    return _selected_model


def get_selected_model() -> Optional[str]:
    """Get the currently selected model, auto-selecting if needed."""
    if _selected_model:
        return _selected_model
    return auto_select_model()


def set_selected_model(model: str) -> None:
    """Manually set the selected model."""
    global _selected_model
    _selected_model = model


# =========================================================
# CORE QUERY FUNCTIONS
# =========================================================
def query_ollama(
    prompt: str,
    system: str = "",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    output_json: bool = False,
    timeout: int = DEFAULT_GENERATE_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    stream: bool = False,
) -> Optional[str]:
    """
    Send a prompt to Ollama and get the response.
    Returns None if Ollama is unavailable or the request fails.

    Features:
      - Automatic model selection if not specified
      - Exponential backoff retry
      - Timeout handling
      - Never raises exceptions — always returns None on failure
    """
    if not is_ollama_running():
        # Try to start Ollama once
        if is_ollama_installed() and start_ollama():
            pass
        else:
            return None

    if not model:
        model = get_selected_model()
    if not model:
        return None

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if output_json:
        payload["format"] = "json"

    # Try 127.0.0.1 first, then localhost
    bases = [OLLAMA_HOST, OLLAMA_LOCALHOST]

    for attempt in range(max_retries + 1):
        for base in bases:
            url = f"{base}{OLLAMA_API_GENERATE}"
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        result = json.loads(resp.read().decode("utf-8"))
                        return result.get("response", "")
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError):
                continue

        if attempt < max_retries:
            backoff = RETRY_BACKOFF ** attempt
            time.sleep(backoff)

    return None


def query_ollama_stream(
    prompt: str,
    system: str = "",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    timeout: int = DEFAULT_GENERATE_TIMEOUT,
) -> Generator[str, None, None]:
    """
    Send a prompt to Ollama and stream the response token by token.
    Yields response chunks. Never raises — silently stops on failure.
    """
    if not is_ollama_running():
        if is_ollama_installed() and start_ollama():
            pass
        else:
            return

    if not model:
        model = get_selected_model()
    if not model:
        return

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    for base in (OLLAMA_HOST, OLLAMA_LOCALHOST):
        url = f"{base}{OLLAMA_API_GENERATE}"
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    for line in resp:
                        try:
                            chunk = json.loads(line.decode("utf-8"))
                            piece = chunk.get("response", "")
                            if piece:
                                yield piece
                            if chunk.get("done", False):
                                return
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError, OSError):
            continue
    return


def query_ollama_json(
    prompt: str,
    system: str = "",
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    timeout: int = DEFAULT_GENERATE_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> Optional[Dict[str, Any]]:
    """Send a prompt and expect JSON response from Ollama."""
    raw = query_ollama(
        prompt, system, model, temperature,
        max_tokens=max_tokens, output_json=True,
        timeout=timeout, max_retries=max_retries,
    )
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None


def batch_ollama(
    prompts: List[str],
    system: str = "",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_workers: int = 4,
) -> List[Optional[str]]:
    """Run multiple prompts through Ollama, optionally in parallel."""
    if not prompts:
        return []

    # Sequential batch (safer for local models)
    results = []
    for prompt in prompts:
        result = query_ollama(prompt, system, model, temperature)
        results.append(result)
    return results


# =========================================================
# STATUS & DIAGNOSTICS
# =========================================================
def get_ollama_status() -> Dict[str, Any]:
    """Get a comprehensive status report of the Ollama installation."""
    installed = is_ollama_installed()
    running = is_ollama_running()
    models = discover_models() if running else []

    status = {
        "installed": installed,
        "running": running,
        "models": models,
        "model_count": len(models),
        "selected_model": get_selected_model() if running else None,
        "binary_path": get_ollama_binary() if installed else None,
        "host": OLLAMA_HOST,
    }

    if not installed:
        status["error"] = (
            "Ollama is not installed. Please install it from https://ollama.com/download\n"
            "After installation, run 'ollama pull llama3.2' to download a model."
        )
    elif not running:
        status["error"] = (
            "Ollama is installed but not running. "
            "Run 'ollama serve' in a terminal, or start the Ollama desktop app."
        )
    elif not models:
        status["error"] = (
            "Ollama is running but no models are installed. "
            "Run 'ollama pull llama3.2' to download a model."
        )

    return status


def print_ollama_status() -> None:
    """Print a human-readable status report."""
    status = get_ollama_status()

    print("\n" + "=" * 60)
    print("  OLLAMA STATUS REPORT")
    print("=" * 60)
    print(f"  Installed:      {'[OK] Yes' if status['installed'] else '[NO] No'}")
    print(f"  Running:        {'[OK] Yes' if status['running'] else '[NO] No'}")
    if status.get("binary_path"):
        print(f"  Binary:         {status['binary_path']}")
    if status["running"]:
        print(f"  Models:         {status['model_count']}")
        for model in status["models"]:
            marker = " <- selected" if model == status["selected_model"] else ""
            print(f"    - {model}{marker}")
    if status.get("error"):
        print(f"\n  [WARN] {status['error']}")
    print("=" * 60)


# =========================================================
# COMPATIBILITY LAYER (drop-in for old OllamaClient.py)
# =========================================================
OLLAMA_MODEL = "phi3:mini"  # Kept for backward compatibility
OLLAMA_URL = f"{OLLAMA_HOST}{OLLAMA_API_GENERATE}"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}{OLLAMA_API_TAGS}"

def check_ollama_available() -> bool:
    """Backward-compatible availability check."""
    return is_ollama_running()


def ollama_complete(system_prompt: str, user_prompt: str,
                    model: str = "",
                    temperature: float = 0.1) -> Optional[str]:
    """Backward-compatible completion function."""
    return query_ollama(
        user_prompt,
        system=system_prompt,
        model=model or None,
        temperature=temperature,
    )


def ollama_complete_json(system_prompt: str, user_prompt: str,
                         model: str = "",
                         temperature: float = 0.1) -> Optional[Dict[str, Any]]:
    """Backward-compatible JSON completion function."""
    return query_ollama_json(
        user_prompt,
        system=system_prompt,
        model=model or None,
        temperature=temperature,
    )


# =========================================================
# MAIN ENTRY POINT
# =========================================================
if __name__ == "__main__":
    print_ollama_status()

    if is_ollama_running():
        models = discover_models()
        if models:
            selected = select_model_interactive()
            if selected:
                print(f"\nTesting model '{selected}' with a sample prompt...")
                response = query_ollama(
                    "Say hello in exactly 5 words.",
                    model=selected,
                    temperature=0.1,
                    max_tokens=50,
                )
                if response:
                    print(f"  Response: {response}")
                else:
                    print("  [ERROR] No response received.")
