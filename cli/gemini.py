
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
import json
import re

from lib.search_utils import GEMINI_API_KEY, formatted_results
from sentence_transformers import CrossEncoder




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
    

def rerank_docs(query: str, rrf_results: dict, method: str, limit: int):
    api_key = GEMINI_API_KEY
    client = genai.Client(api_key=api_key)
    print(f"Reranking Using {method} method...")

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
            sorted_doc = dict(sorted(rrf_results.items(), key=lambda item: item[1]['rerank'], reverse=True)[:limit])
        case "batch": 
            doc_list = []
            for result in rrf_results.keys():
                doc_list.append(rrf_results[result]["document"])
            doc_list_str = str(doc_list)
            content = client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents=f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
            )
            cleaned = re.sub(r'```json|```', '', content.text).strip()
            results = json.loads(cleaned)
            for i, result in enumerate(results):
                rrf_results[result]["rerank"] = i + 1
            sorted_doc = dict(sorted(rrf_results.items(), key=lambda item: item[1]['rerank'])[:limit])
        case "cross_encoder":
            pairs = []
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            for result in rrf_results.keys():
                doc = rrf_results[result].get('document', '')
                pairs.append([query, f"{doc.get('title', '')} - {doc}"])
            scores = cross_encoder.predict(pairs)
            for i, result in enumerate(rrf_results.keys()):
                rrf_results[result]["rerank"] = scores[i]
            sorted_doc = dict(sorted(rrf_results.items(), key=lambda item: item[1]['rerank'], reverse=True)[:limit])
        case _:
            print("no rerank method provided")
            return rrf_results
    
    return sorted_doc


def evaluate_results(query: str, results: dict, rerank_method: str = ""):
    api_key = GEMINI_API_KEY
    client = genai.Client(api_key=api_key)
    formatted_result = formatted_results(results, rerank_method)
    content = client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents=f"""Rate how relevant each result is to this query on a 0-3 scale. The current rankings are numbered.:

Query: "{query}"

Results:
{formatted_result}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order as the they are numbered. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
                )
    cleaned = re.sub(r'```json|```', '', content.text).strip()
    llm_results = json.loads(cleaned)
    for i, res in enumerate(results.keys()):
        results[res]["eval"] = llm_results[i]
    sorted_doc = dict(sorted(results.items(), key=lambda item: item[1]['eval'], reverse=True))
    return sorted_doc


def augmented_results(query: str, rrf_results: dict):
    api_key = GEMINI_API_KEY
    client = genai.Client(api_key=api_key)
    formatted_result = formatted_results(rrf_results)
    content = client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{formatted_result}

Provide a comprehensive answer that addresses the query:"""
                )
    return content.text


def summarize_results(query: str, rrf_results: dict):
    api_key = GEMINI_API_KEY
    client = genai.Client(api_key=api_key)
    formatted_result = formatted_results(rrf_results)
    content = client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{formatted_result}
Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:
"""
                )
    return content.text

def cite_results(query: str, rrf_results: dict):
    api_key = GEMINI_API_KEY
    client = genai.Client(api_key=api_key)
    formatted_result = formatted_results(rrf_results)
    content = client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{formatted_result}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
                )
    return content.text

def question_results(query: str, rrf_results: dict):
    api_key = GEMINI_API_KEY
    client = genai.Client(api_key=api_key)
    formatted_result = formatted_results(rrf_results)
    content = client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {query}

Documents:
{formatted_result}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
                )
    return content.text


def multimodal_results(query: str, img: bytes, mime: str):
    api_key = GEMINI_API_KEY
    client = genai.Client(api_key=api_key)
    prompt = """
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""
    parts = [
        prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        query.strip()
    ]
    content = client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents=parts
                )
    print(f"Rewritten query: {content.text.strip()}")
    if content.usage_metadata is not None:
        print(f"Total tokens:    {content.usage_metadata.total_token_count}")