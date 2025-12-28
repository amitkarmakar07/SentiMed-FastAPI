import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(f"API Key present: {bool(GEMINI_API_KEY)}")

if not GEMINI_API_KEY:
    print("ERROR: No API key found. Please set GEMINI_API_KEY in .env")
    exit(1)

# Configure GenAI
try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("GenAI Configured successfully.")
except Exception as e:
    print(f"Failed to configure GenAI: {e}")
    exit(1)

# List available models
print("\n--- Available Models ---")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Name: {m.name}")
except Exception as e:
    print(f"Failed to list models: {e}")

# Models to test - Matching main.py
models_to_try = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-001",
]

prompt = "Hello, are you working?"

print("\n--- Testing Generation ---")
for model_id in models_to_try:
    print(f"\nTesting model: {model_id}")
    try:
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(prompt)
        
        print("Success!")
        if hasattr(response, "text"):
            print(f"Response: {response.text}")
        else:
            print(f"Response object: {response}")
            
    except Exception as e:
        print(f"FAILED: {e}")
        # Only print stack trace if it's not a simple NotFound error
        if "NotFound" not in str(type(e)):
            import traceback
            traceback.print_exc()

print("\n--- Done ---")
