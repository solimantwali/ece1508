# lmstudio_client.py
from __future__ import annotations
from typing import Optional, Dict, Any, List

import requests
import json


class LMStudioClient:
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model_name: str = "llama-3-8b-instruct",
        default_temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.max_tokens = max_tokens

    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        extra_gen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.default_temperature,
            "max_tokens": self.max_tokens,
        }
        if extra_gen_kwargs:
            payload.update(extra_gen_kwargs)

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
