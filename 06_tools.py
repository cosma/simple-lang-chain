# Primitive 6: tools.
# The model asks for a tool call; your code runs it and sends the result back for the final answer.
# Flow: question -> model asks for tool(s) -> our code runs them -> model writes final answer.
from langchain.messages import HumanMessage  ## the user's question, wrapped in the message format the model expects
from langchain.tools import tool  ## decorator that turns a function into a tool the model can see

from llm import get_model  ## model factory from llm.py (Haiku by default, model name comes from .env)


# --- Define the tool -------------------------------------------------------

@tool  ## registers the function as a tool; LangChain sends its name, docstring and type hints to the model
def get_weather(location: str) -> str:  ## the model fills in `location` when it calls the tool
    """Get the current weather for a city."""  ## the docstring is the description the model reads to decide when to use this
    return f"It's sunny in {location}."  ## fake data; a real tool would call a weather API here


tools = [get_weather]  ## list of every tool we offer the model (add more here)
tools_by_name = {t.name: t for t in tools}  ## lookup table: tool name -> tool object, to run whatever the model names


# --- Attach tools to the model ---------------------------------------------

model = get_model().bind_tools(tools)  ## every later call on `model` includes these tool descriptions


# --- Round 1: the model asks for tools -------------------------------------

messages = [HumanMessage("What's the weather in Boston and Tokyo?")]  ## start the conversation with the question
ai_msg = model.invoke(messages)  ## the model may reply with tool_calls (no text) instead of an answer
# print(ai_msg)  ## uncomment to see the raw tool-request message
messages.append(ai_msg)  ## keep the model's tool request in the history; it must see its own request to match results

assert ai_msg.tool_calls, "model did not request any tool call"  ## self-check: fail loudly if the model skipped the tool

# --- Run the tools our code was asked to run -------------------------------

for call in ai_msg.tool_calls:  ## one entry per requested call, each with name, args and id (two here: Boston and Tokyo)
    # tools_by_name[...] finds the Python function; .invoke(call) runs it with call["args"]
    # and returns a ToolMessage that carries call["id"] so the model can match result to request
    messages.append(tools_by_name[call["name"]].invoke(call))  ## add each result to the history


# --- Round 2: the model writes the final answer ----------------------------

final = model.invoke(messages)  ## the model now sees the question, its tool requests, and the tool results
print(final.text)  ## the answer that uses the tool output, e.g. "It's sunny in Boston and Tokyo."
