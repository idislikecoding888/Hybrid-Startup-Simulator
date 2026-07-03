# backend/hybrid/__init__.py
#
# Hybrid LLM + PPO coordination layer.
#
# The LLM agents (founder / marketing / investor / customer) still perform
# all business reasoning. PPO does NOT make business decisions itself -
# it observes the current state + every agent's proposal and produces a
# dynamic weight per agent. Those weights are used to fuse the agents'
# proposals into one final decision. See hybrid_engine.py for the
# orchestration of this flow.
