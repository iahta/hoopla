#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    verify_embeddings,
    embed_text,
    embed_query_text,
    search_command,
)
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify Language Model")
    subparsers.add_parser("verify_embeddings", help="Verify embedded text")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate Text Embedding")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    embed_query_parser = subparsers.add_parser("embedquery", help="Generate vectors for query")
    embed_query_parser.add_argument("query", type=str, help="Query to be embedded")

    search_parser = subparsers.add_parser("search", help="Search the movie database")
    search_parser.add_argument("query", type=str, help="Query to search movie database")
    search_parser.add_argument("--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT, help="Set the limit of search results")

    args = parser.parse_args()



    

    match args.command:
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "search":
            results = search_command(args.query, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} (score: {res['score']:.4f})\n{res['description']}") 
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
    