"""
Shared Ollama client for local LLM inference.
Uses phi3:mini (2.2GB) — runs on i5/16GB/integrated GPU.
Used for: DOSCAN concept scoring, hypothesis generation, knowledge extraction.
Groq is reserved for: final research comparison, critical validation.
"""

import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"  # 2.2GB, runs on i5/16GB

def query_ollama(prompt: str, system: str = "", 
                 model: str = OLLAMA_MODEL,
                 temperature: float = 0.7,
                 max_tokens: int = 512,
                 output_json: bool = False) -> Optional[str]:
    """
    Send a prompt to Ollama and get the response.
    Falls back gracefully if Ollama is not running.
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
    
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(OLLAMA_URL, data, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result.get("response", "")
    except Exception as e:
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