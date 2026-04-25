"""UI layer — Gradio app + provider routing for BYOK model selection.

The Gradio app lives in ``app.py`` at the repo root (HF Space convention);
this package contains the supporting modules:

- ``providers.py`` — Provider protocol + concrete implementations
  (HF Inference, Anthropic SDK, OpenAI-compatible)
- ``router.py``    — model-to-provider mapping + per-tier curated lists
- ``policies.py``  — adapt a chat-completion provider into the
  ``policy(observation) -> action_dict`` shape the runners want
"""
