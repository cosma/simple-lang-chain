# Primitive 14: LangGraph (the graph under agents).
# create_agent is a ready-made graph. When you need your own steps and routing, you write the graph:
#   state (a dict) -> nodes (functions that return updates) -> edges (fixed or conditional) -> compile -> invoke
# Job: classify a message, then send it to the matching handler.
from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from llm import get_model

model = get_model()


class State(TypedDict):
    message: str
    kind: str   ## "question" or "complaint", filled in by classify
    reply: str  ## the final answer, filled in by a handler


def classify(state: State):
    word = model.invoke(
        f"Is this a 'question' or a 'complaint'? Answer with one word.\n\n{state['message']}"
    ).text.strip().lower()
    return {"kind": "complaint" if "complaint" in word else "question"}  ## a node returns only the keys it changes


def answer_question(state: State):
    return {"reply": model.invoke(f"Answer briefly: {state['message']}").text}


def handle_complaint(state: State):
    return {"reply": model.invoke(f"Write a short, polite apology reply to: {state['message']}").text}


def route(state: State) -> Literal["answer_question", "handle_complaint"]:  ## the conditional edge: pick the next node
    return "handle_complaint" if state["kind"] == "complaint" else "answer_question"


graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("answer_question", answer_question)
graph.add_node("handle_complaint", handle_complaint)
graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route)
graph.add_edge("answer_question", END)
graph.add_edge("handle_complaint", END)
app = graph.compile()  ## must compile before invoke

out = app.invoke({"message": "My kettle arrived broken and nobody answers my emails!"})
print(out["kind"], "->", out["reply"])

assert out["kind"] == "complaint", "classifier missed an obvious complaint"  ## self-check
