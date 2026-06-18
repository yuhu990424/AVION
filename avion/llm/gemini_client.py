from __future__ import annotations

import os
import time

from avion.llm.parser import parse_candidate_response
from avion.llm.prompt_templates import build_remote_sensing_prompt
from avion.llm.schemas import CandidateResponse


class GeminiCandidateGenerator:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
        temperature: float = 0.3,
        top_p: float = 0.95,
        max_retries: int = 5,
        retry_sleep: float = 5.0,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {api_key_env}; export it in the shell.")
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise RuntimeError("google-genai is required for Gemini generation.") from exc

        self._genai = genai
        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

    def generate(self, class_name: str, n: int = 30) -> CandidateResponse:
        prompt = build_remote_sensing_prompt(class_name=class_name, n=n)
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self._types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=self.temperature,
                        top_p=self.top_p,
                    ),
                )
                return parse_candidate_response(response.text, expected_count=n)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep * attempt)
        raise RuntimeError(f"Gemini generation failed for class={class_name}") from last_error

