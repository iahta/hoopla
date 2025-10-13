import os
import numpy as np
from sentence_transformers import SentenceTransformer
from .search_utils import (
    CACHE_DIR,
    load_movies,
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