import argparse
import json
from lib.search_utils import GOLDEN_DATA_PATH, load_movies
from lib.hybrid_search import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    movies = load_movies()
    search = HybridSearch(movies)

    args = parser.parse_args()
    limit = args.limit

    with open(GOLDEN_DATA_PATH, "r") as f:
        golden_data = json.load(f)
    for data in golden_data["test_cases"]:
        rrf_search = search.rrf_search(data["query"], 60, limit)
        precision = len(data["relevant_docs"]) / len(rrf_search)
        result_titles = []
        relevant_titles = []
        for result in rrf_search.keys():
            title = rrf_search[result]["document"]["title"]
            result_titles.append(title)
        for title in data["relevant_docs"]:
            relevant_titles.append(title)
        joined_result_titles = ", ".join(result_titles)
        joined_titles = ", ".join(relevant_titles)
        print(f"- Query: {data["query"]}")
        print(f"    - Precision@{limit}: {precision:.4f}")
        print(f"    - Retrieved: {joined_result_titles}")
        print(f"    - Relevant: {joined_titles}")


if __name__ == "__main__":
    main()
    