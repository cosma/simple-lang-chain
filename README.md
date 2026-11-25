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
| `07_structured_output.py` | Structured output (typed Pydantic result) |
| `08_agent.py` | Agents (`create_agent` runs the tool loop) |
| `09_review_pipeline.py` | 8-component chain: lambda, assign, parallel, branch, batch |
| `10_retrieval.py` | Retrieval (RAG): embeddings, vector store, answer from docs (needs `VOYAGE_API_KEY`) |
| `11_memory.py` | Memory: per-conversation history with a checkpointer |
| `12_streaming.py` | Streaming tokens as they are generated |
| `13_human_in_the_loop.py` | Human approval before a sensitive tool runs |
| `14_langgraph.py` | LangGraph: state, nodes, conditional edges |

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
uv run python 07_structured_output.py
uv run python 08_agent.py
uv run python 09_review_pipeline.py
uv run python 10_retrieval.py        # needs VOYAGE_API_KEY
uv run python 11_memory.py
uv run python 12_streaming.py
uv run python 13_human_in_the_loop.py
uv run python 14_langgraph.py
```
