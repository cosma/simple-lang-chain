# Primitive 10: retrieval (RAG).
# The model only knows what you put in its prompt. RAG = find the relevant docs first, then ask.
#   docs -> embed (vectors) -> vector store
#   question -> embed -> nearest docs -> prompt with those docs -> model answers from them
# Embeddings come from Voyage AI (Anthropic's recommended partner; Anthropic has no embedding model).
# Needs VOYAGE_API_KEY in .env.
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_voyageai import VoyageAIEmbeddings

from llm import get_model

# A tiny fake knowledge base. Real life: split your PDFs/pages into chunks first.
docs = [
    "Returns: items can be returned within 30 days with the original receipt.",
    "Shipping: standard delivery takes 3-5 working days; express takes 1 day.",
    "Warranty: all mugs and kettles have a 2-year warranty against manufacturing defects.",
    "Support: our team answers emails Monday to Friday, 9:00 to 17:00 Berlin time.",
]

embeddings = VoyageAIEmbeddings(model="voyage-3-lite")  ## text -> vector; no chat, no Haiku here
store = InMemoryVectorStore(embeddings)  ## keeps vectors in memory; fine for lessons and small data
store.add_texts(docs)  ## embeds all docs once

question = "How long do I have to send something back?"
hits = store.similarity_search(question, k=2)  ## the 2 docs whose vectors are closest to the question's vector
context = "\n".join(h.page_content for h in hits)

# The model answers only from the retrieved context, so it can't invent a policy.
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the context. If the answer isn't there, say you don't know.\n\nContext:\n{context}"),
    ("human", "{question}"),
])
chain = prompt | get_model() | StrOutputParser()
answer = chain.invoke({"context": context, "question": question})
print(answer)

assert any("30 days" in h.page_content for h in hits), "retrieval missed the returns policy"  ## self-check
