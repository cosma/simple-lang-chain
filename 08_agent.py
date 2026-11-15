# Primitive 8: agents.
# Lesson 06 ran the tool loop by hand. create_agent runs that loop for you:
# model asks for tools -> agent runs them -> feeds results back -> repeats until the model answers.
from langchain.agents import create_agent  ## builds the loop around a model and its tools
from langchain.messages import ToolMessage  ## the message type that holds a tool's result
from langchain.tools import tool  ## same decorator as lesson 06

from llm import get_model


import datetime, pytz



@tool
def get_weather(location: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny in {location}."  ## fake data


ZONES = {"tokyo": "Asia/Tokyo", "berlin": "Europe/Berlin", "bucharest": "Europe/Bucharest",
         "paris": "Europe/Paris", "barcelona": "Europe/Madrid", "toronto": "Canada/Toronto"}  ## pytz wants IANA names, not city names


@tool
def get_time(location: str) -> str:
    """Get the current local time in a city."""
    zone = ZONES.get(location.strip().lower())
    if zone is None:  ## return the miss as text so the model can retry, instead of crashing the run
        return f"Unknown city '{location}'. Known cities: {', '.join(ZONES)}."
    now = datetime.datetime.now(pytz.timezone(zone))
    return f"It's {now:%H:%M} in {location}."  ## fake data


agent = create_agent(
    model=get_model(),  ## our Haiku model object
    tools=[get_weather, get_time],  ## two tools, so the agent has to pick the right one
    system_prompt="You are a concise travel assistant.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "Weather in Paris,  Barcelona and Berlin? How is  time in Tokyo, Torinto, Berlin(Germany) and Bucharest?"}]})


for msg in result["messages"]:  ## the full trail: question, tool requests, tool results, answer
    print(f"\n---\n{msg.type:>6}: {msg.text or msg.tool_calls}")  ## tool requests have no text, so show tool_calls

print("\n-------\nFinal:\n-------\n", result["messages"][-1].text)  ## the last message is the answer

assert any(isinstance(m, ToolMessage) for m in result["messages"]), "agent never called a tool"