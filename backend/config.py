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
    MODEL_NAME = "groq/compound"   # ✅ active Groq model
    TEMPERATURE = 0.7
    MAX_TOKENS = 1024

    # Hybrid LLM + PPO weighting layer
    HYBRID_PPO_ENABLED = True          # if False, weighting.py always uses equal weights
    HYBRID_WEIGHTING_TEMPERATURE = 0.5  # softmax temperature over agent alignment scores


settings = Settings()