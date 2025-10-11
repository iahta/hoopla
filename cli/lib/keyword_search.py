import os
import sys
import string
import pickle
from nltk.stem import PorterStemmer

from collections import defaultdict, Counter

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stop_words,
)

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    inverted_index = InvertedIndex()
    inverted_index.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []   
    for token in query_tokens:
        doc_ids = inverted_index.get_documents(token)
        for id in doc_ids:
            if id in seen:
                continue
            seen.add(id)
            doc = inverted_index.docmap[id]
            if not doc:
                continue
            results.append(doc)
            if len(results) >= limit:
                return results
    return results

def process_text(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
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
        self.term_frequencies: dict[int, Counter] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def __add_document(self, doc_id: int, text: str):
        tokenized_text = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokenized_text)
        for token in set(tokenized_text):
            self.index[token].add(doc_id)

    def get_documents(self, term: str):
        term = term.lower()
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokenized_text = tokenize_text(term)
        if len(tokenized_text) > 1:
            raise ValueError("More than one token")
        return self.term_frequencies[doc_id][tokenized_text[0]]
    
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
        with open(self.term_frequencies_path, "wb") as f_term_frequencies:
            pickle.dump(self.term_frequencies, f_term_frequencies)

    def load(self):
        with open(self.index_path, "rb") as f_index:
            self.index = pickle.load(f_index)
        with open(self.docmap_path, "rb") as f_docmap:
            self.docmap = pickle.load(f_docmap)
        with open(self.term_frequencies_path, "rb") as f_term_termfrequencies:
            self.term_frequencies = pickle.load(f_term_termfrequencies)

def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()

def tf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


