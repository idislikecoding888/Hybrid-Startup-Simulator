# backend/config.py

class Settings:
    # Simulation controls
    MAX_STEPS = 50
    INITIAL_BUDGET = 10000
    INITIAL_PRICE = 500
    INITIAL_CUSTOMERS = 100

    # Agent configs
    AGENTS = [
        "founder",
        "marketing",
        "investor",
        "customer"
    ]

    # Deliberation settings
    MAX_CRITIQUE_ROUNDS = 2
    CONSENSUS_THRESHOLD = 0.7

    # Logging
    ENABLE_LOGGING = True

    # Groq LLM config
    LLM_PROVIDER = "groq"
    # FIX: "groq/compound" is Groq's agentic/tool-using *system* model — a
    # single chat.completions.create() call to it can internally trigger many
    # sub-calls (web search, code execution, secondary reasoning passes).
    # That is the actual source of the 80-100+ calls per step and of the
    # intermittent "LLM call failed" errors (extra internal hops = more
    # surface area for timeouts/rate limits). Swapping to a plain, non-agentic
    # chat model makes each agent.run() correspond to exactly one real API
    # call, restoring the expected ~4 calls/step (max 12 with retries).
    MODEL_NAME = "llama-3.1-8b-instant"
    TEMPERATURE = 0.7
    MAX_TOKENS = 1024
    REQUEST_TIMEOUT_SECONDS = 30

    # Hybrid LLM + PPO weighting layer
    HYBRID_PPO_ENABLED = True          # if False, weighting.py always uses equal weights
    HYBRID_WEIGHTING_TEMPERATURE = 0.5  # softmax temperature over agent alignment scores


settings = Settings()