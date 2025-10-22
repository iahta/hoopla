import argparse

from lib.augmented_generation import (
    rag_command,
    summarize_command,
)
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
)

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")


    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument("--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT, help="Limit the search results default 5")
    


    args = parser.parse_args()

    match args.command:
        case "rag":
            rag_command(args.query)
        case "summarize":
            summarize_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()