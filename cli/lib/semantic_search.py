import os
import numpy as np
from sentence_transformers import SentenceTransformer
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    CACHE_DIR,
    load_movies,
    format_embedded_search_result,
)

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text: str):
        if text == "" or text.isspace():
            raise ValueError("No text to embed")
        encode = self.model.encode([text])
        return encode[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        doc_list = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        doc_list = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
        if os.path.exists(self.embeddings_path):
                self.embeddings = np.load(self.embeddings_path)
                if len(self.embeddings) == len(doc_list):
                    return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit):
        score = []
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        for i, doc_embedding in enumerate(self.embeddings):
            cosigne = cosine_similarity(query_embedding, doc_embedding)
            score.append((cosigne, self.documents[i]))
        sorted_scores = sorted(score, key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in sorted_scores[:limit]:
            formatted_result = format_embedded_search_result(
                score,
                doc["title"],
                doc["description"]
            )
            results.append(formatted_result)
        return results
    
def verify_model():
    search = SemanticSearch()
    model = search.model
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")

def embed_text(text: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search = SemanticSearch()
    documents = load_movies()
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1) #magnitude
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    search = SemanticSearch()
    movies = load_movies()
    search.load_or_create_embeddings(movies)
    results = search.search(query, limit)
    return results
    
def fixed_size_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = 0) -> list[str]:
    words = text.split()
    chunks = []
    n_words = len(words)
    i = 0
    while i < n_words - overlap:
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks

def chunk_command(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP):
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, res in enumerate(chunks, 1):
        print(f"{i}. {res}")
    


