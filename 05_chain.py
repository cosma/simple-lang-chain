# Primitive 5: chains (LCEL).
# The | operator wires steps together: prompt -> model -> parser. Each step's output feeds the next.
from langchain_core.output_parsers import StrOutputParser  ## step 3: keep only the text
from langchain_core.prompts import ChatPromptTemplate  ## step 1: fill the template


from llm import get_model  ## step 2: the model

prompt = ChatPromptTemplate.from_messages([  ## same template as lesson 3
    ("system", "You translate English to {language}. Reply with the translation only."),
    ("human", "{text}"),
])

chain = prompt | get_model() | StrOutputParser()  ## chain of three runnables joined by |

print(chain.invoke({"language": "Romanian", "text": "I love building applications."}))  ## input dict goes into step 1
