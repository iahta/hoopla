import json
import os
from dotenv import load_dotenv
from typing import Any

load_dotenv()
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_SEMANTIC_CHUNK_OVERLAP = 1
ALPHA_CONSTANT_HYBRID = 0.5
K_CONSTANT_RRF = 60
MAX_CHUNK_SIZE = 4
SCORE_PRECISION = 3
BM25_K1 = 1.5 #k1 - tunable saturation parameter
BM25_B = 0.75 #B - normalization strength

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
GEMINI_API_KEY = api_key = os.environ.get("GEMINI_API_KEY")
GOLDEN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATA_PATH, "r") as f:
        return json.load(f)

def load_stop_words() -> list[str]:
    with open(STOP_WORDS_PATH, "r") as f: 
        return f.read().splitlines()

def format_search_result (
        doc_id: str, title: str, desription: str, score: float, **metadata: Any
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "title": title,
        "description": desription,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }

def format_embedded_search_result(
        score: float, doc_id: int, title: str, description: str
) -> dict[int, str, str]:
    return {
        "score": score,
        "id": doc_id,
        "title": title,
        "description": description
    }

def formatted_results(results: dict, rerank_method: str = ""):
    lines = []
    for i, result in enumerate(results.keys()):
        score = results[result]
        doc = results[result]["document"]

        lines.append(f"{i + 1}. {doc["title"]}")
        if "rerank" in score:
            match rerank_method:
                case "individual":
                    lines.append(f"Rerank Score: {score['rerank']:.3f}/10")
                case "batch":
                    lines.append(f"Rerank Rank: {score['rerank']}")
                case "cross_encoder":
                    lines.append(f"Cross Encoder Score: {score['rerank']:.3f}")
        lines.append(f"RRF Score: {score["rrf_sum"]:.3f}")
        lines.append(f"BM25 Rank: {score["bm25_rank"]}, Semantic Rank: {score["semantic_rank"]}")
        lines.append(f"{doc["description"]}")
    return "\n".join(lines)