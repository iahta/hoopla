from .search_utils import (
    load_movies,
    DEFAULT_SEARCH_LIMIT,
    K_CONSTANT_RRF,
)
from .hybrid_search import (
    HybridSearch,
)

from gemini import augmented_results

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