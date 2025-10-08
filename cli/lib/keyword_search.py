from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
)
def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        processed_query = process_text(query)
        processed_title = process_text(movie["title"])
        if processed_query in processed_title:
            results.append(movie)
            if len(results) >= limit:
                break
    return results

def process_text(text: str) -> str:
    text = text.lower()
    return text
