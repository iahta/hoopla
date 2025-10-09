from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
)
import string
def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        processed_query = process_text(query)
        processed_title = process_text(movie["title"])
        if any(word in processed_title for word in processed_query):
            results.append(movie)
            if len(results) >= limit:
                break
    return results

def process_text(text: str) -> list[str]:
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = [word for word in text.split() if word.strip()]
    return text
