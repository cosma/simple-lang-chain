# Primitive 11: memory (short-term, per conversation).
# An LLM forgets everything between calls. A checkpointer saves the message history per thread_id,
# so a second invoke on the same thread sees the first one.
from langchain_core.globals import set_debug
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  ## keeps history in RAM; use a Postgres saver in production

from llm import get_model

set_debug(False)

agent = create_agent(
    model=get_model(),
    tools=[],
    system_prompt="You are a friendly assistant. Keep answers to one sentence.",
    checkpointer=InMemorySaver(),
)

thread = {"configurable": {"thread_id": "lesson-11"}}  ## same id = same conversation; new id = fresh memory

agent.invoke({"messages": [{"role": "user", "content": "Hi! My name is Bob."}]}, thread)
result = agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, thread)
print(result["messages"][-1].text)

other = agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]},
                     {"configurable": {"thread_id": "lesson-11-new"}})  ## different thread: no memory of Bob
print(other["messages"][-1].text)

assert "Bob" in result["messages"][-1].text, "agent forgot the name on the same thread"  ## self-check
