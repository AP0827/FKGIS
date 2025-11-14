# test_llm.py
import spacy
from spacy_llm import make_llm

nlp = spacy.blank("en")

llm = make_llm(
    model="openai/gpt-4o-mini",
    task="completion",
    config={"max_tokens": 20}
)

nlp.add_pipe("llm", config={"llm": llm})

doc = nlp("Say hello to Aayush.")
print(doc._.llm_answer)
