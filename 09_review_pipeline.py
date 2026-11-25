# Lesson 9: a bigger chain with 8 components, built only from pieces you have already seen.
# Job: take a customer review and write a reply. Unhappy and urgent reviews get an apology,
# all others get a thank-you.
#
#   {review}
#     -> 1 clean           RunnableLambda             plain Python, no API
#     -> 2+3 classify      prompt | structured model  adds "analysis" (a Review)
#     -> 4 assign          RunnablePassthrough        keeps "review", adds "analysis"
#     -> 5 fan-out         RunnableParallel           summary + topic, side by side
#     -> 6 route           RunnableBranch             apology or thank-you
#     -> 7 reply           prompt | model
#     -> 8 parse           StrOutputParser            plain string out
from typing import Literal

from langchain_core.globals import set_debug
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field

from llm import get_model

set_debug(True)  # prints input/output of every runnable: prompts, model replies, parsers, branch


class Review(BaseModel):  ## same schema as lesson 07 (files starting with a digit can't be imported)
    """Analysis of a customer review."""
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Overall tone of the review")
    topic: str = Field(description="What the review is mainly about, 1-3 words")
    urgent: bool = Field(description="True if the customer needs help soon (broken item, missing order, refund)")


def needs_apology(x: dict) -> bool:  ## the routing rule, kept as a plain function so we can test it for free
    a = x["analysis"]
    return a.sentiment == "negative" and a.urgent


# Self-check before any API call: a logic bug fails here, without spending tokens.
assert needs_apology({"analysis": Review(sentiment="negative", topic="refund", urgent=True)})
assert not needs_apology({"analysis": Review(sentiment="negative", topic="color", urgent=False)})

model = get_model()
parser = StrOutputParser()

# 1. clean: strip spaces and cap length (stops a huge input from costing a lot)
clean = RunnableLambda(lambda x: {"review": x["review"].strip()[:75]}) # 200 

# 2 + 3. classify: prompt -> model that returns a Review object
classify = ChatPromptTemplate.from_messages([
    ("system", "Classify the customer review."),
    ("human", "{review}"),
]) | model.with_structured_output(Review)

# 5a. a side job that runs alongside the topic step: a one-line summary
summarize = ChatPromptTemplate.from_messages([
    ("system", "Summarize the review in one short sentence."),
    ("human", "{review}"),
]) | model | parser

# 7. two reply prompts, one per branch. Both read {review}, {summary} and {topic}.
apology = ChatPromptTemplate.from_messages([
    ("system", "You are customer support. Write a short, sincere apology (max 3 sentences) "
               "and promise a fix within 24h. Issue: {topic}. Summary: {summary}"),
    ("human", "{review}"),
])
thanks = ChatPromptTemplate.from_messages([
    ("system", "You are customer support. Write a short, warm thank-you (max 2 sentences). "
               "Topic: {topic}. Summary: {summary}"),
    ("human", "{review}"),
])


###
### { "review": "My order arrived smashed and support hasn't answered in a week. I want a refund!...."}
###

###
### { "review": "My order arrived smashed and support hasn't answered in a week. I want a refund!...."}
###

chain = (
    clean  ## 1
    | RunnablePassthrough.assign(analysis=classify)  ## 2, 3, 4: dict now has review + analysis
    | RunnableParallel(  ## 5: every key is computed at the same time from the same input dict
        review=lambda x: x["review"],
        analysis=lambda x: x["analysis"],
        topic=lambda x: x["analysis"].topic,  ## prompts can't read x.analysis.topic, so flatten it here
        summary=summarize,
    )
    | RunnableBranch(  ## 6: first (condition, runnable) pair that matches wins; last item is the default
        (needs_apology, apology | model | parser),  ## 7 + 8
        thanks | model | parser,  ## 7 + 8
    )
)




reviews = [
    {"review": "  My order arrived smashed and support hasn't answered in a week. I want a refund!  "},
    # {"review": "Love the mug, the colour is even nicer than in the photos."},
    # {"review": "The blue is a bit darker than I expected, but it's fine."},
]

replies = chain.batch(reviews)  ## runs all three reviews concurrently, returns results in the same order

for r, reply in zip(reviews, replies):
    print(f"REVIEW: {r['review'].strip()}\nREPLY:  {reply}\n")

def draw_graph(path="chain.mmd"):  ## writes Mermaid text; paste into mermaid.live or a GitHub .md to view
    with open(path, "w") as f:
        f.write(chain.get_graph().draw_mermaid())


draw_graph()

chain.get_graph().print_ascii()

chain.get_graph().draw_mermaid_png()