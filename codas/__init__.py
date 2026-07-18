"""CoDaS: AI Co-Data-Scientist for wearable biomarker discovery.

Subpackages:
  codas.core     deterministic discovery engine (numpy/pandas/scipy/scikit-learn, no LLM)
  codas.agents   google-adk + Gemini agent graph over the engine's deterministic tools
  codas.service  FastAPI service exposing the engine and the agent pipeline
"""
