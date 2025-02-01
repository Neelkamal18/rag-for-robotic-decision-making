import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class FAISSIndexer:
    def __init__(self, data_path, index_path):
        self.data_path = data_path
        self.index_path = index_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.documents = []

    def load_data(self):
        """Loads knowledge base from a JSON file."""
        with open(self.data_path, "r") as file:
            data = json.load(file)
            self.documents = data["documents"]
        return self.documents

    def create_faiss_index(self):
        """Creates a FAISS index from document embeddings."""
        if not self.documents:
            self.load_data()
        
        texts = [doc["text"] for doc in self.documents]
        embeddings = self.model.encode(texts)  # Generate embeddings

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings, dtype="float32"))

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        print(f"✅ FAISS index saved at {self.index_path}")

if __name__ == "__main__":
    FAISSIndexer("data/robotic_manuals.json", "data/embeddings/robotic_manuals.faiss").create_faiss_index()
    FAISSIndexer("data/troubleshooting_logs.json", "data/embeddings/troubleshooting_logs.faiss").create_faiss_index()
