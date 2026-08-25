"""Base URLs + default models for the free-tier API providers (BYO key).

NB: model ids and free-tier limits DRIFT (monthly) — re-verify before relying on
them (docs/stack.md). Cerebras ~1M tok/day; Groq fast, ~14.4k req/day.
"""

CEREBRAS = {"base_url": "https://api.cerebras.ai/v1", "default_model": "qwen-3-32b"}
GROQ = {"base_url": "https://api.groq.com/openai/v1", "default_model": "llama-3.3-70b-versatile"}
