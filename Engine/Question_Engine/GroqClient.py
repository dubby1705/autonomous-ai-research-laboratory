"""
SHARED GROQ CLIENT with Automatic Ollama Fallback
- Uses Groq by default (reads API key from environment)
- If Groq fails with 429 (rate limit) or any error → automatically falls back to Ollama
- All pipeline files should use this instead of raw Groq()
"""

import os
import json
import time
from typing import Optional, Dict, Any, Callable
from groq import Groq

# Import Ollama fallback
from Engine.Question_Engine.OllamaClient import query_ollama_json as _ollama_json

# Track rate limit state to avoid hammering Groq
_groq_rate_limited = False
_groq_rate_limit_until = 0

def get_groq_client() -> Groq:
    """Get a Groq client using the environment API key."""
    return Groq()

def groq_complete(system_prompt: str, user_prompt: str, 
                  model: str = "llama-3.3-70b-versatile",
                  temperature: float = 0.1,
                  response_format: Optional[Dict] = None,
                  max_retries: int = 2) -> Optional[str]:
    """
    Call Groq with automatic Ollama fallback on failure.
    
    If Groq returns 429 (rate limit) or any other error:
    - Falls back to Ollama phi3:mini
    - Waits 5 seconds before retrying Groq (once)
    - If Groq still fails, stays on Ollama for the rest of the session
    
    Returns:
        The response text, or None if both fail
    """
    global _groq_rate_limited, _groq_rate_limit_until
    
    current_time = time.time()
    
    # If we're in rate limit cooldown, skip Groq and go straight to Ollama
    if _groq_rate_limited and current_time < _groq_rate_limit_until:
        return _fallback_to_ollama(system_prompt, user_prompt, temperature)
    
    # Try Groq
    for attempt in range(max_retries + 1):
        try:
            client = get_groq_client()
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
            }
            if response_format:
                kwargs["response_format"] = response_format
            
            completion = client.chat.completions.create(**kwargs)
            
            # Success — reset rate limit state
            _groq_rate_limited = False
            return completion.choices[0].message.content
            
        except Exception as e:
            error_str = str(e)
            
            # Check for rate limit (429)
            if "429" in error_str or "rate_limit" in error_str.lower() or "rate limit" in error_str.lower():
                print(f"   ⚠️ Groq rate limited. Falling back to Ollama...")
                _groq_rate_limited = True
                _groq_rate_limit_until = current_time + 60  # 60 second cooldown
                return _fallback_to_ollama(system_prompt, user_prompt, temperature)
            
            # Check for JSON validation errors (prompt issue, not rate limit)
            if "json_validate_failed" in error_str or "Failed to generate JSON" in error_str:
                if attempt < max_retries:
                    print(f"   ⚠️ Groq JSON error, retrying... ({attempt + 1}/{max_retries})")
                    time.sleep(1)
                    continue
                else:
                    print(f"   ⚠️ Groq JSON error after retries. Falling back to Ollama...")
                    return _fallback_to_ollama(system_prompt, user_prompt, temperature)
            
            # Other errors — fall back to Ollama
            print(f"   ⚠️ Groq error: {str(e)[:80]}. Falling back to Ollama...")
            return _fallback_to_ollama(system_prompt, user_prompt, temperature)
    
    return None

def _fallback_to_ollama(system_prompt: str, user_prompt: str, 
                        temperature: float) -> Optional[str]:
    """Fall back to Ollama when Groq is unavailable."""
    try:
        result = _ollama_json(user_prompt, system_prompt, temperature=temperature)
        if result:
            return json.dumps(result)
        return None
    except Exception as e:
        print(f"   ❌ Ollama fallback also failed: {e}")
        return None

def groq_complete_json(system_prompt: str, user_prompt: str,
                       model: str = "llama-3.3-70b-versatile",
                       temperature: float = 0.1) -> Optional[Dict[str, Any]]:
    """
    Call Groq with JSON response format, with Ollama fallback.
    """
    response = groq_complete(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    if response:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None
    return None