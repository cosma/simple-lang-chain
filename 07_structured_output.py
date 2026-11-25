# Primitive 7: structured output.
# Instead of free text, the model fills in a typed object you define. Use it when you need fields, not prose.
from typing import Literal  ## restricts a field to a fixed set of values

from pydantic import BaseModel, Field  ## pydantic describes the shape; it ships with langchain

from llm import get_model  ## model factory from llm.py (Haiku by default)


class Review(BaseModel):  ## the shape we want back; LangChain sends this schema to the model
    """Analysis of a customer review."""  ## the docstring also goes to the model as the schema description
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Overall tone of the review")
    topic: str = Field(description="What the review is mainly about, 1-3 words")
    urgent: bool = Field(description="True if the customer needs help soon (broken item, missing order, refund)")


model = get_model().with_structured_output(Review)  ## wraps the model: output is a Review, not an AIMessage

result = model.invoke("My order arrived smashed and support hasn't answered in a week. I want a refund.")
print(result)  ## e.g. sentiment='negative' topic='damaged order' urgent=True
print(result.sentiment, result.urgent)  ## normal attribute access, no string parsing

assert isinstance(result, Review), "expected a Review object"  ## self-check: we got the typed object
