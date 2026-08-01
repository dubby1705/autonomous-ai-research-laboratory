"""
SHARED LLM CLIENT — Ollama-First with Groq Fallback
===================================================
Hybrid provider strategy:
  - Ollama (local, free) is PREFERRED for all analysis/hypothesis/derivation.
  - Groq is used ONLY for simulation comparison + as fallback.
"""

import os
import json
import time
from typing import Optional, Dict, Any
from groq import Groq

from Engine.Question_Engine.OllamaClient import (
    query_ollama_json as _ollama_json,
    query_ollama as _ollama_text,
    check_ollama_available as _ollama_health,
    OLLAMA_MODEL,
)

_groq_bad_key = False
_ollama_checked_flag = False
_ollama_ok = False


def check_ollama_available() -> bool:
    """Robust check whether Ollama is running (localhost + 127.0.0.1)."""
    global _ollama_checked_flag, _ollama_ok
    if _ollama_checked_flag:
        return _ollama_ok
    _ollama_ok = _ollama_health()
    _ollama_checked_flag = True
    return _ollama_ok


def _print_unauthorized_error():
    """Print a clear diagnostic when the Groq API key is invalid."""
    print()
    print("   ============================================================")
    print("   ERROR: Groq API key is invalid or missing (HTTP 401).")
    if check_ollama_available():
        print("   -> Ollama is available. Continuing with local fallback.")
    else:
        print("   -> Ollama is NOT reachable (tried 127.0.0.1 and localhost:11434).")
        print("      Start 'ollama serve' to enable local inference.")
    print("   ============================================================")
    print()


def ollama_complete(system_prompt: str, user_prompt: str,
                    model: str = OLLAMA_MODEL,
                    temperature: float = 0.1) -> Optional[str]:
    """Call Ollama directly. Returns text response or None."""
    if not check_ollama_available():
        print("   ❌ Ollama is not reachable (tried 127.0.0.1 and localhost:11434).")
        return None
    try:
        return _ollama_text(user_prompt, system=system_prompt,
                            model=model, temperature=temperature)
    except Exception as e:
        print(f"   ❌ Ollama call failed: {e}")
        return None


def ollama_complete_json(system_prompt: str, user_prompt: str,
                         model: str = OLLAMA_MODEL,
                         temperature: float = 0.1) -> Optional[Dict[str, Any]]:
    """Call Ollama and parse JSON response."""
    if not check_ollama_available():
        return None
    try:
        return _ollama_json(user_prompt, system=system_prompt,
                            model=model, temperature=temperature)
    except Exception as e:
        print(f"   ❌ Ollama JSON call failed: {e}")
        return None


def groq_complete(system_prompt: str, user_prompt: str,
                  model: str = "llama-3.3-70b-versatile",
                  temperature: float = 0.1,
                  response_format: Optional[Dict] = None,
                  max_retries: int = 2) -> Optional[str]:
    """Call Groq only. Returns None if unavailable/failed."""
    global _groq_bad_key

    if _groq_bad_key:
        return None
    if not os.environ.get("GROQ_API_KEY", "").strip():
        print("   ⚠️  GROQ_API_KEY not set. Skipping Groq.")
        return None

    for attempt in range(max_retries + 1):
        try:
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
            }
            if response_format:
                kwargs["response_format"] = response_format
            completion = client.chat.completions.create(**kwargs)
            _groq_bad_key = False
            return completion.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if "401" in error_str or "invalid" in error_str.lower() or "unauthorized" in error_str.lower():
                _groq_bad_key = True
                _print_unauthorized_error()
                return None
            if "429" in error_str or "rate_limit" in error_str.lower() or "rate limit" in error_str.lower():
                _groq_bad_key = True
                print("   ⚠️  Groq rate limited. Aborting Groq.")
                return None
            if attempt < max_retries:
                print(f"   ⚠️  Groq error, retrying... ({attempt + 1}/{max_retries})")
                time.sleep(1)
                continue
            print(f"   ⚠️  Groq error: {str(e)[:80]}")
            return None
    return None


def groq_complete_json(system_prompt: str, user_prompt: str,
                       model: str = "llama-3.3-70b-versatile",
                       temperature: float = 0.1) -> Optional[Dict[str, Any]]:
    """Call Groq and parse JSON response."""
    response = groq_complete(
        system_prompt, user_prompt, model=model, temperature=temperature,
        response_format={"type": "json_object"},
    )
    if response:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None
    return None


def llm_complete(system_prompt: str, user_prompt: str,
                 model: str = "",
                 temperature: float = 0.1,
                 response_format: Optional[Dict] = None,
                 max_retries: int = 2) -> Optional[str]:
    """Ollama-first with Groq fallback — used by analysis/hypothesis stages."""
    if check_ollama_available():
        result = ollama_complete(system_prompt, user_prompt,
                                 model=model or OLLAMA_MODEL,
                                 temperature=temperature)
        if result:
            return result
        print("   ⚠️  Ollama returned empty. Trying Groq fallback...")
    return groq_complete(system_prompt, user_prompt,
                         model=model or "llama-3.3-70b-versatile",
                         temperature=temperature,
                         response_format=response_format,
                         max_retries=max_retries)


def llm_complete_json(system_prompt: str, user_prompt: str,
                      model: str = "",
                      temperature: float = 0.1) -> Optional[Dict[str, Any]]:
    """Ollama-first JSON response, Groq fallback."""
    if check_ollama_available():
        result = ollama_complete_json(system_prompt, user_prompt,
                                      model=model or OLLAMA_MODEL,
                                      temperature=temperature)
        if result:
            return result
        print("   ⚠️  Ollama returned empty. Trying Groq fallback...")
    return groq_complete_json(system_prompt, user_prompt, model=model or "llama-3.3-70b-versatile",
                              temperature=temperature)


def llm_complete_prefer_groq(system_prompt: str, user_prompt: str,
                             model: str = "llama-3.3-70b-versatile",
                             temperature: float = 0.1,
                             response_format: Optional[Dict] = None,
                             max_retries: int = 2) -> Optional[str]:
    """Groq-first with Ollama fallback — used by simulation comparison."""
    result = groq_complete(system_prompt, user_prompt, model=model,
                           temperature=temperature,
                           response_format=response_format,
                           max_retries=max_retries)
    if result:
        return result
    if check_ollama_available():
        print("   ⚠️  Groq unavailable. Falling back to Ollama...")
        return ollama_complete(system_prompt, user_prompt,
                               model=OLLAMA_MODEL, temperature=temperature)
    return None


def llm_complete_prefer_groq_json(system_prompt: str, user_prompt: str,
                                  model: str = "llama-3.3-70b-versatile",
                                  temperature: float = 0.1) -> Optional[Dict[str, Any]]:
    """Groq-first JSON response, Ollama fallback."""
    result = groq_complete_json(system_prompt, user_prompt, model=model,
                                temperature=temperature)
    if result:
        return result
    if check_ollama_available():
        print("   ⚠️  Groq unavailable. Falling back to Ollama...")
        return ollama_complete_json(system_prompt, user_prompt,
                                    model=OLLAMA_MODEL, temperature=temperature)
    return None