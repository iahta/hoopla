import argparse
import json

from lib.hybrid_search import (
    normalize,
    weighted_search_command,
    rrf_search_command,
)
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    ALPHA_CONSTANT_HYBRID,
    K_CONSTANT_RRF,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores to match semantic scores")
    normalize_parser.add_argument("scores", type=json.loads, nargs="*", help="Numbers to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Weighted Search")
    weighted_search_parser.add_argument("query", type=str, help="The query to search the documents")
    weighted_search_parser.add_argument("--alpha", type=float, nargs="?", default=ALPHA_CONSTANT_HYBRID, help="The alpha constant to use in the search")
    weighted_search_parser.add_argument("--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT, help="Limit the number of results returned")
    
    rrf_search_parser = subparsers.add_parser("rrf-search", help="Seach using Reciprical Rank Fusion")
    rrf_search_parser.add_argument("query", type=str, help="The query to search the documents")
    rrf_search_parser.add_argument("--k", type=int, nargs="?", default=K_CONSTANT_RRF, help="The k constant is how much weight we give results in the search")
    rrf_search_parser.add_argument("--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT, help="Limit the number of results returned")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual"], help="Rerank method")
    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize(args.scores)
            for score in scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit, args.enhance, args.rerank_method)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()