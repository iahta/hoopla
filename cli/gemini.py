import os
from dotenv import load_dotenv
from google import genai
import time

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
        case "rewrite":
            content = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
            )
        case "expand":
            content = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""
            )
        case _:
            print("no method provided")
            return


    metadata = content.usage_metadata
    print(f"Prompt Tokens: {metadata.prompt_token_count}\nResponse Tokens: {metadata.candidates_token_count}")  
    return content.text
    

def rerank_docs(query: str, rrf_results: dict, method: str):
    api_key = GEMINI_API_KEY
    client = genai.Client(api_key=api_key)
    print(f"Reranking Using key {api_key[:6]}...")
    match method:
        case "individual":
            for result in rrf_results.keys():
                doc = rrf_results[result]["document"]
                content = client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents=f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
                )
                
                rrf_results[result]["rerank"] = float(str(content.text).strip())
                time.sleep(3)
        
        case _:
            print("no rerank method provided")
            return rrf_results
    
    sorted_doc = dict(sorted(rrf_results.items(), key=lambda item: item[1]['rerank'], reverse=True))
    return sorted_doc

