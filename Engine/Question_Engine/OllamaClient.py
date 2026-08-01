"""
Shared Ollama client for local LLM inference.
Uses phi3:mini (2.2GB) — runs on i5/16GB/integrated GPU.

Used for: problem analysis, knowledge extraction, hypothesis generation,
DOSCAN concept scoring, mathematics derivation, evidence scoring.

PREFERRED PROVIDER: Ollama is preferred for all analysis/hypothesis/derivation
stages. Groq is reserved ONLY for the simulation comparison (Research/compare.py).

Includes a robust availability check that tries both `localhost` and `127.0.0.1`
to avoid false "Ollama not running" detections.
"""

import json
import urllib.request
from typing import Optional, Dict, Any, List

# Try 127.0.0.1 first, then localhost — avoids IPv6/IPv4 resolution pitfalls
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"
OLLAMA_MODEL = "phi3:mini"  # 2.2GB, runs on i5/16GB

# Cache for availability check
_ollama_checked = False
_ollama_available = False


def check_ollama_available() -> bool:
    """Check whether Ollama is reachable.
    Tries 127.0.0.1 first, then localhost. Caches the result.
    """
    global _ollama_checked, _ollama_available
    if _ollama_checked:
        return _ollama_available

    candidates = [
        OLLAMA_TAGS_URL,
        "http://localhost:11434/api/tags",
        "http://127.0.0.1:11434/api/tags",
    ]
    for url in candidates:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    _ollama_available = True
                    break
        except Exception:
            continue

    _ollama_checked = True
    return _ollama_available


def query_ollama(prompt: str, system: str = "",
                 model: str = OLLAMA_MODEL,
                 temperature: float = 0.7,
                 max_tokens: int = 512,
                 output_json: bool = False) -> Optional[str]:
    """
    Send a prompt to Ollama and get the response.
    Returns None if Ollama is not running or the request fails.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    if output_json:
        payload["format"] = "json"

    # Try 127.0.0.1 first, then localhost
    for base in ("http://127.0.0.1:11434", "http://localhost:11434"):
        url = f"{base}/api/generate"
        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                return result.get("response", "")
        except Exception:
            continue
    return None


def query_ollama_json(prompt: str, system: str = "",
                      model: str = OLLAMA_MODEL,
                      temperature: float = 0.3) -> Optional[Dict[str, Any]]:
    """Send a prompt and expect JSON response from Ollama."""
    raw = query_ollama(prompt, system, model, temperature, output_json=True)
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


def batch_ollama(prompts: List[str], system: str = "",
                 model: str = OLLAMA_MODEL,
                 temperature: float = 0.7) -> List[Optional[str]]:
    """Run multiple prompts sequentially through Ollama."""
    results = []
    for prompt in prompts:
        result = query_ollama(prompt, system, model, temperature)
        results.append(result)
    return results