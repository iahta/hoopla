from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stop_words,
)
from nltk.stem import PorterStemmer
import os
import string
import pickle
def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results

def process_text(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def has_matching_token(query_tokens: list[str],title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def tokenize_text(text: str) -> list[str]:
    text = process_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stop_words()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

class InvertedIndex:
    def __init__(self):
        self.index = {} #token (strings) -set of doc ids
        self.docmap = {} #doc ID (int) - full document object
    def __add_document(self, doc_id: int, text: str):
        tokenized_text = tokenize_text(text)
        for token in tokenized_text:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
    def get_documents(self, term: str):
        term = term.lower()
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)
    def build(self, movies: list[dict]):
        for movie in movies:
            self.__add_document(movie['id'], f"{movie['title']}{movie['description']}")
            self.docmap[movie["id"]] = movie
    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f_index:
            pickle.dump(self.index, f_index)
        with open("cache/docmap.pkl", "wb") as f_docmap:
            pickle.dump(self.docmap, f_docmap)