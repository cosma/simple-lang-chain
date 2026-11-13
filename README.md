# simple-lang-chain

Small LangChain project to learn the primitives. One numbered file per
primitive, in order.

| File | Primitive |
|---|---|
| `01_chat_model.py` | Chat model |
| `02_messages.py` | Messages (system / human / ai) |
| `03_prompt_template.py` | Prompt templates |
| `04_output_parser.py` | Output parsers |
| `05_chain.py` | Chains with `\|` (LCEL) |
| `06_tools.py` | Tool calling |

`llm.py` builds the model. It uses Claude Haiku by default; set `CHAT_MODEL`
in `.env` to change it.

## Setup

```bash
cp .env.example .env   # then add your ANTHROPIC_API_KEY
uv sync
```

## Run

```bash
uv run python 01_chat_model.py
uv run python 02_messages.py
uv run python 03_prompt_template.py   # no API key needed
uv run python 04_output_parser.py     # no API key needed
uv run python 05_chain.py
uv run python 06_tools.py
```
