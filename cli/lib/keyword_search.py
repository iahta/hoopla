import os
import string
import pickle
from nltk.stem import PorterStemmer

from collections import defaultdict

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stop_words,
)

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                return results
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
        self.index = defaultdict(set) #token (strings) -set of doc ids
        self.docmap: dict[int, dict] = {} #doc ID (int) - full document object
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str):
        tokenized_text = tokenize_text(text)
        for token in set(tokenized_text):
            self.index[token].add(doc_id)

    def get_documents(self, term: str):
        term = term.lower()
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            doc_description = f"{movie['title']}{movie['description']}"
            self.docmap[movie["id"]] = movie
            self.__add_document(doc_id, doc_description)
            
    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f_index:
            pickle.dump(self.index, f_index)
        with open(self.docmap_path, "wb") as f_docmap:
            pickle.dump(self.docmap, f_docmap)

def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")