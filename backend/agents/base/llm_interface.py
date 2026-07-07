import os
import json
import time
import random
from groq import Groq
from dotenv import load_dotenv
from backend.config import settings

load_dotenv()

_raw_key = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEY = _raw_key.strip().strip('"').strip("'")

_TIMEOUT = getattr(settings, "REQUEST_TIMEOUT_SECONDS", 30)

# FIX: no explicit timeout previously meant a slow/agentic model response
# could hang far longer than expected before the SDK's own (very long)
# default kicked in. An explicit per-request timeout means a bad request
# fails fast and predictably, instead of stalling and surfacing as a
# "random" failure much later.
client = Groq(api_key=GROQ_API_KEY, timeout=_TIMEOUT) if GROQ_API_KEY else None

SYSTEM_PROMPT = (
    "You are an executive in a startup board meeting. "
    "Use only the supplied live state and debate context. "
    "Return ONLY valid JSON. No markdown, no code fences, no extra text. "
    "If the prompt provides exact numbers, treat them as ground truth. "
    "Never invent launch-state numbers or stale examples."
)

def call_llm(prompt: str) -> str:
    if client is None:
        return json.dumps({
            "error": "LLM call failed",
            "details": "GROQ_API_KEY is missing"
        })

    last_error = None

    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )

            content = completion.choices[0].message.content or ""
            return content.strip()

        except Exception as exc:
            last_error = exc
            if attempt < 2:
                backoff = (0.75 * (2 ** attempt)) + random.uniform(0, 0.5)
                time.sleep(backoff)

    return json.dumps({
        "error": "LLM call failed",
        "details": repr(last_error)
    })
