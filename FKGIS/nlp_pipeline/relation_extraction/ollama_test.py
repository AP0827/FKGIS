import spacy
import sys

def build_textcat_with_gemma():
    nlp = spacy.blank("en")

    nlp.add_pipe(
        "llm",
        config={
            "task": {
                # FIX 3.1: Use the more modern task version
                "@llm_tasks": "spacy.TextCat.v3", 
                "labels": "INSULT, COMPLIMENT, THREAT, PRAISE",
                "exclusive_classes": True,
            },
            "model": {
                # FIX 1: Use the NEW wrapper for langchain-ollama
                "@llm_models": "langchain.OllamaLLM.v1", 
                
                "name": "gemma:7b",
                "config": {
                    "base_url": "http://localhost:11434",
                    "temperature": 0.0,
                    # FIX 3.2: Add context length to remove the sharding warning
                    "context_length": 2048 
                }
            }
        },
    )

    return nlp

if __name__ == "__main__":
    # FIX 2: Read from command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python ollama_test.py \"<your text here>\"")
        sys.exit(1)
        
    text = sys.argv[1]
    
    print(f"Processing text: '{text}'")
    
    # Make sure your Ollama server is running!
    try:
        nlp = build_textcat_with_gemma()
        doc = nlp(text)
        print("\n--- Results ---")
        # Sort results for readability
        sorted_cats = sorted(doc.cats.items(), key=lambda item: item[1], reverse=True)
        print(dict(sorted_cats))
        
    except Exception as e:
        print(f"\n--- An Error Occurred ---")
        print(f"Make sure your Ollama server is running and 'gemma:7b' is pulled.")
        print(f"Error details: {e}")