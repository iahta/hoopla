import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from lib.search_utils import load_movies


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, (limit * 500))
        semantic_results = self.semantic_search.search_chunks(query, (limit * 500))
        
        norm_bm25 = normalize(bm25_results)
        norm_semantic = normalize(semantic_results)

        bm_25_dict = {doc["id"]: doc for doc in norm_bm25}
        semantic_dict = {doc["id"]: doc for doc in norm_semantic}

        all_ids = set(bm_25_dict.keys()) | set(semantic_dict.keys())
        weighted_results = []
        for doc_id in all_ids:
            bm25_doc = bm_25_dict.get(doc_id, {})
            semantic_doc = semantic_dict.get(doc_id, {})

            bm25_score = bm25_doc.get("norm_score", 0)
            semantic_score = semantic_doc.get("norm_score", 0)

            hybrid = hybrid_score(bm25_score, semantic_score, alpha)
            title = bm25_doc.get("title") or semantic_doc.get("title") or ""
            description = bm25_doc.get("document") or semantic_doc.get("description") or ""
            
            weighted_results.append({
                "doc_id": doc_id,
                "bm25": bm25_score,
                "semantic": semantic_score,
                "hybrid_score": hybrid,
                "title": title,
                "description": description[:100]
            })
        weighted_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return weighted_results[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
def normalize(scores: list[dict]) -> list[dict]:
    if not scores:
        return []
    
    values = [s["score"] for s in scores]
    max_score = max(values)
    min_score = min(values)
    norm_scores = []
    if min_score == max_score:
        return [1.0] * len(scores)
    for score in scores:
        score["norm_score"] = (score["score"] - min_score) / (max_score - min_score)
        norm_scores.append(score)
    return norm_scores

def hybrid_score(bm25_score, semantic_score, alpha: float = 0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search_command(query: str, alpha: float = 0.5, limit: int = 5):
    movies = load_movies()
    search = HybridSearch(movies)
    results = search.weighted_search(query, alpha, limit)
    for i, result in enumerate(results):
        print(f"{i + 1}. {result["title"]}\nHybrid Score: {result["hybrid_score"]:.3f}\nBM25: {result["bm25"]:.3f}, Semantic: {result["semantic"]:.3f}\n{result["description"]}")

 
