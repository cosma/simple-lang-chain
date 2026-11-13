# Primitive 4: output parsers.
# The model returns an AIMessage object. A parser turns it into the plain value you want.
from langchain_core.output_parsers import StrOutputParser  ## parser that keeps just the text
from langchain_core.messages import AIMessage  ## a hand-made model reply, so we can test the parser alone

parser = StrOutputParser()  ## create the parser (no model needed)

fake_reply = AIMessage("J'adore la programmation.")  ## pretend the model said this
#print(fake_reply)
text = parser.invoke(fake_reply)  ## parser pulls out the string
print(repr(text))  ## repr shows it is a plain str now
assert isinstance(text, str), "parser should return a str"  ## self-check: the output type is str
