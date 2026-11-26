"""
Model adapters: OpenAI-compatible, local HTTP (vLLM), and Ollama.
All return plain text (model's raw output).
"""
from __future__ import annotations
import os, httpx, json, time
from typing import Dict, Any, Optional, List

class BaseAdapter:
    def generate(self, prompt: str, **gen_kwargs) -> str:
        raise NotImplementedError

class OpenAIAdapter(BaseAdapter):
    def __init__(self, model: str, base_url: Optional[str]=None, api_key: Optional[str]=None):
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY", "") if api_key is None else api_key

    def generate(self, prompt: str, **gen_kwargs) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "messages": [{"role":"user","content":prompt}],
            "temperature": gen_kwargs.get("temperature", 0.7),
            "max_tokens": gen_kwargs.get("max_new_tokens", 800),
            "top_p": gen_kwargs.get("top_p", 1.0),
        }
        with httpx.Client(timeout=60) as client:
            r = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]

class HTTPAdapter(BaseAdapter):
    """Generic text generation server (vLLM-style) expecting a simple /generate endpoint."""
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, **gen_kwargs) -> str:
        payload = {
            "prompt": prompt,
            "temperature": gen_kwargs.get("temperature", 0.7),
            "max_tokens": gen_kwargs.get("max_new_tokens", 800),
            "top_p": gen_kwargs.get("top_p", 1.0),
        }
        with httpx.Client(timeout=120) as client:
            r = client.post(f"{self.base_url}/generate", json=payload)
            r.raise_for_status()
            return r.json().get("text","")

class OllamaAdapter(BaseAdapter):
    def __init__(self, model: str, base_url: str="http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, **gen_kwargs) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": gen_kwargs.get("temperature", 0.7),
                "top_p": gen_kwargs.get("top_p", 1.0),
                "num_predict": gen_kwargs.get("max_new_tokens", 800),
            }
        }
        with httpx.Client(timeout=120) as client:
            r = client.post(f"{self.base_url}/api/generate", json=payload)
            r.raise_for_status()
            out = []
            for line in r.iter_lines():
                if not line: continue
                obj = json.loads(line)
                if "response" in obj:
                    out.append(obj["response"])
                if obj.get("done"):
                    break
            return "".join(out)
