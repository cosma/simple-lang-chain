# Primitive 12: streaming.
# Users don't want to wait for the full answer. stream_mode="messages" yields tokens as the model writes them.
from langchain.agents import create_agent
from langchain.tools import tool

from llm import get_model


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny in {city}."  ## fake data


agent = create_agent(model=get_model(), tools=[get_weather])

pieces = []  ## we keep the text pieces so the self-check can look at them
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather in Lisbon? Answer in one sentence."}]},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] != "messages":
        continue
    token, metadata = chunk["data"]
    if metadata.get("langgraph_node") == "model" and token.text:  ## only the model's words, not tool plumbing
        print(token.text, end="", flush=True)  ## flush so each token appears immediately
        pieces.append(token.text)
print()

assert len(pieces) > 1, "got the whole answer in one chunk; nothing was streamed"  ## self-check
