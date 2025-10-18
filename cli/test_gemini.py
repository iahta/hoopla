import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")

from google import genai

client = genai.Client(api_key=api_key)

content = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
)

print(content.text)
metadata = content.usage_metadata
print(f"Prompt Tokens: {metadata.prompt_token_count}\nResponse Tokens: {metadata.candidates_token_count}")  
