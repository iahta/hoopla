from .search_utils import (
    load_movies,
    DEFAULT_SEARCH_LIMIT,
    K_CONSTANT_RRF,
)
from .hybrid_search import (
    HybridSearch,
)

from gemini import (
    augmented_results,
    summarize_results,
    cite_results,
    question_results,
)

def rag_command(query: str):
    movies = load_movies()
    rrf_search = HybridSearch(movies)
    rrf_results = rrf_search.rrf_search(query, K_CONSTANT_RRF, DEFAULT_SEARCH_LIMIT)
    rag_response = augmented_results(query, rrf_results)
    print("Search Results:")
    for result in rrf_results.keys():
        doc = rrf_results[result]["document"]
        print(f" - {doc["title"]}")
    print()
    print("RAG Response:")
    print(rag_response)

def summarize_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    rrf_search = HybridSearch(movies)
    rrf_results = rrf_search.rrf_search(query, K_CONSTANT_RRF, DEFAULT_SEARCH_LIMIT)
    summary = summarize_results(query, rrf_results)
    print("Search Results:")
    for result in rrf_results.keys():
        doc = rrf_results[result]["document"]
        print(f" - {doc["title"]}")
    print()
    print("LLM Summary:")
    print(summary)

def citation_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    rrf_search = HybridSearch(movies)
    rrf_results = rrf_search.rrf_search(query, K_CONSTANT_RRF, DEFAULT_SEARCH_LIMIT)
    cite_result = cite_results(query, rrf_results)
    print("Search Results:")
    for result in rrf_results.keys():
        doc = rrf_results[result]["document"]
        print(f" - {doc["title"]}")
    print()
    print("LLM Answer:")
    print(cite_result)

def question_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    rrf_search = HybridSearch(movies)
    rrf_results = rrf_search.rrf_search(query, K_CONSTANT_RRF, DEFAULT_SEARCH_LIMIT)
    answer = question_results(query, rrf_results)
    print("Search Results:")
    for result in rrf_results.keys():
        doc = rrf_results[result]["document"]
        print(f" - {doc["title"]}")
    print()
    print("Answer:")
    print(answer)
