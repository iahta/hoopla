from rapidfuzz import fuzz
from lib.hybrid_search import HybridSearch
from lib.search_utils import (
    load_movies,
    load_golden_dataset,
)
from lib.semantic_search import SemanticSearch

def is_close_match(a: str, b: str, threshold: int = 80) -> bool:
    return fuzz.token_set_ratio(a, b) >= threshold

def precision_at_k(retrieved_docs: list[str], relevant_docs: set[str], k: int = 5) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / k


def recall_at_k(retrieved_docs: list[str], relevant_docs: set[str], k: int = 5) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / len(relevant_docs)

"""
def precision_at_k(retrieved_docs: list[str], relevant_docs: set[str], k: int = 5) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        for relevant_doc in relevant_docs:
            if is_close_match(doc, relevant_doc, threshold=80):
                relevant_count += 1
    return relevant_count / k

def recall_at_k(retrieved_docs: list[str], relevant_docs: set[str], k: int = 5, threshold: int = 80) -> float:
    top_k = retrieved_docs[:k]
    matched_relevant = set() 
    for retrieved_doc in top_k:
        for relevant_doc in relevant_docs:
            if is_close_match(retrieved_doc, relevant_doc, threshold):
                matched_relevant.add(retrieved_doc)
                  # Stop after first match to avoid double-counting
    return len(matched_relevant) / len(relevant_docs) if relevant_docs else 0.0
"""
def evaluate_command(limit: int = 5) -> dict:
    movies = load_movies()
    golden_data = load_golden_dataset()
    test_cases = golden_data["test_cases"]

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    total_precision = 0
    results_by_query = {}
    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = set(test_case["relevant_docs"])
        search_results = hybrid_search.rrf_search(query, k=60, limit=limit)
        retrieved_docs = []
        for result in search_results:
            title = search_results[result].get("document", "").get("title", "")
            if title:
                retrieved_docs.append(title)
        precision = precision_at_k(retrieved_docs, relevant_docs, limit)
        recall = recall_at_k(retrieved_docs, relevant_docs, limit)

        results_by_query[query] = {
            "precision": precision,
            "recall": recall,
            "retrieved": retrieved_docs[:limit],
            "relevant": list(relevant_docs),
        }

        total_precision += precision

    return {
        "test_cases_count": len(test_cases),
        "limit": limit,
        "results": results_by_query,
    }
