# Primitive 3: prompt templates.
# A template is a message list with {placeholders}. Fill it in with a dict of values.
from langchain_core.prompts import ChatPromptTemplate  ## the template class

prompt = ChatPromptTemplate.from_messages([  ## build the template from (role, text) pairs
    ("system", "You translate English to {language}. Reply with the translation only."),  ## {language} is a placeholder
    ("human", "{text}"),  ## {text} is filled in at call time
])

filled = prompt.invoke({"language": "French", "text": "I love building applications."})  ## substitute the placeholders
print(filled.to_messages())  ## show the resulting messages; no model is called here

# from llm import get_model

# model = get_model()
# reply = model.invoke(filled)  ## filled is a PromptValue; invoke accepts it directly
# print(reply.text)
