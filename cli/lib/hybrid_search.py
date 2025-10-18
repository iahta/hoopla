import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from gemini import (
    enhance_prompt,
    rerank_docs,
)
from .search_utils import (
    load_movies,
    K_CONSTANT_RRF,
    DEFAULT_SEARCH_LIMIT,
)


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

    def rrf_search(self, query, k, limit: int = 10, rerank_method: str = ""):
        bm25_results = self._bm25_search(query, (limit * 500))
        semantic_results = self.semantic_search.search_chunks(query, (limit * 500))
        
        doc_map = {}

        for rank, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            score = rrf_score(rank, k)
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "document": doc,
                    "bm25_rank": rank,
                    "semantic_rank": 0,
                    "bm25_rrf": score,
                    "semantic_rrf": 0,
                    "rrf_sum": 0,
                }
            else:
                sum_score = doc_map[doc_id]["semantic_rrf"] + score
                doc_map[doc_id]["bm25_rank"] = rank
                doc_map[doc_id]["rrf_sum"] = sum_score

        for rank, doc in enumerate(semantic_results):
            doc_id = doc["id"]
            score = rrf_score(rank, k)
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "document": doc,
                    "bm25_rank": 0,
                    "semantic_rank": rank,
                    "bm25_rrf": 0,
                    "semantic_rrf": score,
                    "rrf_sum": 0,
                }
            else:
                sum_score = doc_map[doc_id]["bm25_rrf"] + score
                doc_map[doc_id]["semantic_rank"] = rank
                doc_map[doc_id]["rrf_sum"] = sum_score

        if rerank_method:
            limit = limit * 5
            
        sorted_doc = dict(sorted(doc_map.items(), key=lambda item: item[1]['rrf_sum'], reverse=True)[:limit])
        return sorted_doc

def rrf_score(rank, k=60):
    return 1 / (k + rank)

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
    
 
def rrf_search_command(query: str, k: int = K_CONSTANT_RRF, limit: int = DEFAULT_SEARCH_LIMIT, method: str = "", rerank_method: str = ""):
    movies = load_movies()
    search = HybridSearch(movies)

    if method:
        enhanced_query = enhance_prompt(query, method)
        if query == enhanced_query:
            print("No enhancement, using original query")
        else:
            print( f"Enhanced query ({method}): '{query}' -> {enhanced_query}\n")
            query = enhanced_query
    
    rrf_results = search.rrf_search(query, k, limit, rerank_method)
    if rerank_method:
        rrf_results = rerank_docs(query, rrf_results, rerank_method, limit)
    for i, result in enumerate(rrf_results.keys()):
        score = rrf_results[result]
        doc = rrf_results[result]["document"]
        print(f"{i + 1}. {doc["title"]}")
        if "rerank" in score:
            match rerank_method:
                case "individual":
                    print(f"Rerank Score: {score['rerank']:.3f}/10")
                case "batch":
                    print(f"Rerank Rank: {score['rerank']}")
        print(f"RRF Score: {score["rrf_sum"]:.3f}")
        print(f"BM25 Rank: {score["bm25_rank"]}, Semantic Rank: {score["semantic_rank"]}")
        print(f"{doc["description"][:100]}")