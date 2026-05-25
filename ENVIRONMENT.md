# Environment

Recommended runtime:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9 or newer is supported. Python 3.10 or newer is recommended.

API-based steps require an OpenAI-compatible chat-completions endpoint:

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4.1-mini-2025-04-14"
```

Local embedding steps use `sentence-transformers`. The default model is `BAAI/bge-m3`:

```bash
export SENTENCE_TRANSFORMERS_HOME=/path/to/cache
```

The released scripts do not include private API keys, local absolute data paths, or generated experiment outputs.
