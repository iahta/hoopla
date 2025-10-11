#!/usr/bin/env python3
from lib.keyword_search import (
    search_command, 
    build_command,
    tf_command,
    idf_command,
    tf_idf_command,
    bm25_idf_command,
    bm25_tf_command,
    bm25search_command,
)

from lib.search_utils import BM25_K1, BM25_B, DEFAULT_SEARCH_LIMIT, load_movies

import argparse



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build Inverted Index of Movies")
    
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Show frequncey of a term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID number")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="Show rarity score for term")
    idf_parser.add_argument("term", type=str, help="Term to get rarity score for")

    tf_idf_parser = subparsers.add_parser("tfidf", help="Show TF-IDF for a term")
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID number")
    tf_idf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")

    bm25_idf_parser = subparsers.add_parser(
        'bm25idf', help=("Get BM25 IDF score for a given term")
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get Bm25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 B parameter for length normalization")
    
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT, help="Set limit of Documents Scores: Default 5")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print(f'Searching for: {args.query}')
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. Title: {res['title']} ID: {res['id']}")    
        case "tf":
            print(f"Retrieving frequency of {args.term}...")
            result = tf_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {result}")
        case "idf":
            print("Calculating the Inverse Document Frequency...")
            result = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {result:.2f}")
        case "tfidf":
            result = tf_idf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {result:.2f}")
        case "bm25idf":
            result = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {result:.2f}")
        case "bm25tf":
            result = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {result:.2f}")
        case "bm25search":
            results = bm25search_command(args.query, args.limit)
            movies = load_movies()
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res}) Title: {movies[res - 1]["title"]} Score: {results[res]:.2f} ")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()