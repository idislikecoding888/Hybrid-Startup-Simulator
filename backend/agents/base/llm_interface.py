# backend/agents/base/llm_interface.py

import os
from groq import Groq
from dotenv import load_dotenv
from backend.config import settings

load_dotenv()

# Strip any accidental surrounding quotes from the API key in .env
_raw_key = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEY = _raw_key.strip().strip('"').strip("'")

client = Groq(api_key=GROQ_API_KEY)


def call_llm(prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI agent in a startup simulation. "
                        "Always return ONLY valid JSON. No markdown, no explanation, no code fences."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )
        return completion.choices[0].message.content

    except Exception as e:
        # Return structured error so deliberation_engine can surface it
        return f'{{"error": "LLM call failed", "details": "{str(e)}"}}'
