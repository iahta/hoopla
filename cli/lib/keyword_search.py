import os
import string
import pickle
import math
from nltk.stem import PorterStemmer

from collections import defaultdict, Counter

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    BM25_K1,
    BM25_B,
    load_movies,
    load_stop_words,
    format_search_result, 
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
        self.term_frequencies = defaultdict(Counter) #doc id (int) - counter object (term: frequency)
        self.doc_lengths = defaultdict(int)
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
    
    def __get_avg_doc_length(self) -> float:
        total_length = 0
        for doc_length in self.doc_lengths.values():
            total_length += doc_length
        if total_length == 0:
            return 0.0
        return total_length / len(self.doc_lengths)

    def get_documents(self, term: str):
        term = term.lower()
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokenized_text = tokenize_text(term)
        if len(tokenized_text) != 1:
            raise ValueError("More than one token")
        return self.term_frequencies[doc_id][tokenized_text[0]]
    
    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        #OkapiBM25
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap) #N
        term_doc_count = len(self.index[token]) #DF
        #log((N - df + 0.5) / (df + 0.5) + 1)
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
    
    def get_bm25_tf (self, doc_id : int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        #term frequency saturation
        raw_tf = self.get_tf(doc_id, term)
        #Length normalization factor
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)
        tf_component = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
        return tf_component

    def bm25(self, doc_id: int, term: str) -> float:
        bm25_idf = self.get_bm25_idf(term)
        bm25_tf = self.get_bm25_tf(doc_id, term)
        return bm25_tf * bm25_idf
    
    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        tokens = tokenize_text(query)
        scores = defaultdict(float)
        for doc_id in self.docmap:
            score = 0.0
            for token in tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_scores[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                desription=doc["description"],
                score=score,
            )
            results.append(formatted_result)
        return results

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            doc_description = f"{movie['title']} {movie['description']}"
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
        with open(self.doc_lengths_path, "wb") as f_doc_lengths:
            pickle.dump(self.doc_lengths, f_doc_lengths)

    def load(self):
        with open(self.index_path, "rb") as f_index:
            self.index = pickle.load(f_index)
        with open(self.docmap_path, "rb") as f_docmap:
            self.docmap = pickle.load(f_docmap)
        with open(self.term_frequencies_path, "rb") as f_term_termfrequencies:
            self.term_frequencies = pickle.load(f_term_termfrequencies)
        with open(self.doc_lengths_path, "rb") as f_doc_lengths:
            self.doc_lengths = pickle.load(f_doc_lengths)

def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)

def tf_idf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    idx = InvertedIndex()
    idx.load()
    scores = idx.bm25_search(query, limit)
    return scores