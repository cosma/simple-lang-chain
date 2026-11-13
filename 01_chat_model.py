# Primitive 1: chat model.
# A chat model is the object that talks to the LLM. You call .invoke() on it.
from llm import get_model  ## our factory from llm.py that builds the Haiku model

model = get_model()  ## create the chat model object (no API call happens yet)

reply = model.invoke("What is LangChain in one sentence?")  ## send a plain string; returns an AIMessage

print(reply.text)  ## .text gives just the answer string, without metadata
print(reply.usage_metadata)
