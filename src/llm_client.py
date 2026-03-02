# src/llm_client.py
import time
from openai import OpenAI

class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float, max_tokens: int, timeout_sec: int, max_retries: int):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def chat(self, messages, force_json_object: bool = True) -> str:
        """
        If force_json_object=True, try OpenAI response_format json_object first.
        If the backend doesn't support it, gracefully fall back to normal chat output.
        """
        last_err = None
        tried_json_mode = False

        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs = {}
                if force_json_object:
                    tried_json_mode = True
                    kwargs["response_format"] = {"type": "json_object"}

                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs
                )
                return resp.choices[0].message.content

            except Exception as e:
                last_err = e
                msg = str(e).lower()

                # If backend doesn't support response_format, fall back once.
                if force_json_object and (
                    "response_format" in msg or
                    "unknown parameter" in msg or
                    "unexpected keyword" in msg or
                    "not supported" in msg
                ):
                    force_json_object = False  # downgrade and retry
                time.sleep(1.2 * attempt)

        raise RuntimeError(f"LLM call failed after retries (json_mode_tried={tried_json_mode}): {last_err}")