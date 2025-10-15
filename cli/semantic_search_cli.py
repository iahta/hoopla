#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    verify_embeddings,
    embed_text,
    embed_query_text,
    semantic_search,
    chunk_command,
    semantic_chunks_command,
    embed_chunks_command,
    search_chunked,
)
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MAX_CHUNK_SIZE,
    DEFAULT_SEMANTIC_CHUNK_OVERLAP, 
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    subparsers.add_parser("verify", help="Verify Language Model")
    subparsers.add_parser("verify_embeddings", help="Verify embedded text")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate Text Embedding")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    embed_query_parser = subparsers.add_parser("embedquery", help="Generate vectors for query")
    embed_query_parser.add_argument("query", type=str, help="Query to be embedded")

    search_parser = subparsers.add_parser("search", help="Search the movie database")
    search_parser.add_argument("query", type=str, help="Query to search movie database")
    search_parser.add_argument("--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT, help="Set the limit of search results")

    chunk_parser = subparsers.add_parser("chunk", help="Split text into smaller pierces")
    chunk_parser.add_argument("text", type=str, help="Text to be split")
    chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=DEFAULT_CHUNK_SIZE, help="Size of chunk of text")
    chunk_parser.add_argument("--overlap", type=int, nargs="?", default=DEFAULT_CHUNK_OVERLAP, help="Number of words to overlap in a chunk")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Split texts on natural breaks")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs="?", default=DEFAULT_CHUNK_SIZE, help="Maximum size of a chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs="?", default=DEFAULT_CHUNK_OVERLAP)

    subparsers.add_parser("embed_chunks", help="Embed chunks of data")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search chunked scores of movies to find the most similiar result")
    search_chunked_parser.add_argument("query", type=str, help="Query to be chunk searched")
    search_chunked_parser.add_argument("--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT, help="Limit the size of your search results")

    args = parser.parse_args()



    

    match args.command:
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "embed_chunks":
            embed_chunks_command()
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "search":
            results = semantic_search(args.query, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} (score: {res['score']:.4f})\n{res['description']}") 
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunks_command(args.text, args.max_chunk_size, args.overlap)
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
    