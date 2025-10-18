import os
from dotenv import load_dotenv
from google import genai

from lib.search_utils import GEMINI_API_KEY




def enhance_prompt(query, method):
    api_key = GEMINI_API_KEY
    client = genai.Client(api_key=api_key)
    print(f"Using key {api_key[:6]}...")

    match method:
        case "spell":  
            content = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=f"""Fix any spelling errors in this movie query.
                        Only correct obvious typos. Don't change correctly spelled words.
                        Query: "{query}" 
                        If no errors, return the original query.
                        Corrected:"""
            )

        case _:
            print("no method provided")
            return


    metadata = content.usage_metadata
    print(f"Prompt Tokens: {metadata.prompt_token_count}\nResponse Tokens: {metadata.candidates_token_count}")  
    return content.text
    