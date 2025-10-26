from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity
from .search_utils import load_movies

class MultimodalSearch:
    def __init__(self, documents: list[dict], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)


    def embed_image(self, path: str):
        img = Image.open(path)
        encode = self.model.encode([img])
        return encode[0]
    
    def search_with_image(self, path: str):
        image_embedding = self.embed_image(path)
        result = []
        for i, text_embedding in enumerate(self.text_embeddings):
            cos_sim = cosine_similarity(image_embedding, text_embedding)
            result.append({
                "doc_id": self.documents[i]["id"],
                "title": self.documents[i]["title"],
                "description": self.documents[i]["description"],
                "score": cos_sim
            })
        sorted_doc = sorted(result, key=lambda item: item['score'], reverse=True)[:5]
        return sorted_doc

    
def verify_image_embedding(path: str):
    search = MultimodalSearch()
    embedding = search.embed_image(path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(path: str):
    movies = load_movies()
    search = MultimodalSearch(movies)
    result = search.search_with_image(path)
    return result
