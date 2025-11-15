import os
from openai import OpenAI
from openai import APIError, AuthenticationError

def check_env():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("❌ OPENAI_API_KEY is NOT set in environment.")
        return None
    print("✅ OPENAI_API_KEY found in environment.")
    return key

def check_key_validity(key):
    try:
        client = OpenAI(api_key=key)
        models = client.models.list()
        print("✅ API key is active. Model count:", len(models.data))
        return True
    except AuthenticationError as e:
        print("❌ API key is INVALID.")
        print("Reason:", e)
        return False
    except APIError as e:
        print("⚠️ API key present but request failed (rate limits / network).")
        print("Reason:", e)
        return False
    except Exception as e:
        print("⚠️ Unexpected error:", e)
        return False

if __name__ == "__main__":
    key = check_env()
    if key:
        check_key_validity(key)
