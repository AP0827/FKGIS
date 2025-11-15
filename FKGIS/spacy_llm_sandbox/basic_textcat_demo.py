import sys
import spacy

def make_textcat_nlp():
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        config={
            "task": {
                "@llm_tasks": "spacy.TextCat.v1",
                "labels": "INSULT, COMPLIMENT",
            },
            "model": {
                # Use an adapter name from the error list, e.g. spacy.GPT-3-5.v3
                "@llm_models": "spacy.GPT-3-5.v3",
                "name": "gpt-3.5-turbo",
            },
        },
    )
    return nlp

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python basic_textcat_demo.py 'Your text here'")
        sys.exit(1)

    text = sys.argv[1]
    nlp = make_textcat_nlp()
    doc = nlp(text)
    print("Text:", text)
    print("Categories:", doc.cats)
