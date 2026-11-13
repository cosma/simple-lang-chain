# Primitive 2: messages.
# A conversation is a list of messages. Each one has a role: system, human, or ai.
from langchain.messages import AIMessage, HumanMessage, SystemMessage  ## the three message classes

from llm import get_model  ## model factory

model = get_model()  ## build the chat model

# Message objects: SystemMessage sets behaviour, HumanMessage is the user, AIMessage is the model's past reply.
conversation = [  ## a list of messages is the full chat history we send
    SystemMessage("You translate English to French. You always return one phrase"),  ## instructions for the model
    HumanMessage("Translate: I love programming."),  ## a user turn
    AIMessage("J'adore la programmation."),  ## a previous model turn, so the model sees the pattern
    HumanMessage("Translate: I love building applications."),  ## the new question
]

reply = model.invoke(conversation)  ## the model answers the last message in light of the history
print(reply.text)
print(reply.usage_metadata)
# print(reply.tool_calls)
# print(reply.model_dump_json)

# The same conversation as plain dicts also works (OpenAI-style role/content).
dict_conversation = [  ## same content as above, different syntax
    {"role": "system", "content": "You translate English to French. You always return one phrase"},  ## "role" picks the message type
    {"role": "user", "content": "Translate: I love programming."},  ## "user" == HumanMessage
]
#print(model.invoke(dict_conversation).text)  ## identical kind of result
