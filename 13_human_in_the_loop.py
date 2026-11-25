# Primitive 13: human in the loop.
# Some tools must not run without a person saying yes (sending email, deleting data, spending money).
# HumanInTheLoopMiddleware pauses the agent before those tools; you resume with approve or reject.
# Needs a checkpointer and a thread_id so the paused run can be picked up again.
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from llm import get_model

SENT = []  ## stands in for a real mail server, so we can see when the tool actually ran


@tool
def search_docs(query: str) -> str:
    """Search the internal docs."""
    return f"Found 1 page about '{query}'."  ## fake data, safe to run without asking


@tool
def send_email(to: str, body: str) -> str:
    """Send an email to a person."""
    SENT.append((to, body))  ## the side effect we want a human to approve
    return f"Email sent to {to}."


agent = create_agent(
    model=get_model(),
    tools=[search_docs, send_email],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"send_email": True, "search_docs": False})],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "lesson-13"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Email anna@example.com that the report is ready."}]},
    config, version="v2",
)

print("paused:", bool(result.interrupts))
assert result.interrupts and not SENT, "send_email ran without approval"  ## nothing sent yet

result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config, version="v2")  ## a human said yes
print("sent to:", SENT)
assert len(SENT) == 1, "approval did not run the tool"  ## self-check
