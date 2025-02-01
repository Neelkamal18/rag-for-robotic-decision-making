import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

INDEX_DIR = "models/faiss_index/"

# Ensure the FAISS index directory exists
os.makedirs(INDEX_DIR, exist_ok=True)


class FAISSIndexer:
    def __init__(self, data_path, index_name):
        """Initializes FAISS indexer with data source and index storage."""
        self.data_path = data_path
        self.index_path = os.path.join(INDEX_DIR, index_name)  # Store in `models/faiss_index/`
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # Sentence Transformer for embeddings
        self.index = None
        self.documents = []

    def load_data(self):
        """Loads knowledge base from a JSON file and validates format."""
        if not os.path.exists(self.data_path):
            print(f"Error: Data file {self.data_path} not found.")
            return False

        try:
            with open(self.data_path, "r") as file:
                data = json.load(file)
                if "documents" not in data:
                    print(f"Error: Invalid JSON format in {self.data_path}. Missing 'documents' key.")
                    return False
                self.documents = data["documents"]
                return True
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {self.data_path}.")
            return False

    def create_faiss_index(self):
        """Creates a FAISS index from document embeddings and saves it."""
        if not self.load_data() or not self.documents:
            print(f"⚠️ Warning: No documents found in {self.data_path}. Skipping FAISS indexing.")
            return

        texts = [doc["text"] for doc in self.documents]
        embeddings = self.model.encode(texts)  # Generate sentence embeddings

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)

        # Add embeddings to FAISS index
        self.index.add(np.array(embeddings, dtype="float32"))

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index successfully saved at {self.index_path}")

    def index_exists(self):
        """Checks if FAISS index already exists."""
        return os.path.exists(self.index_path)


if __name__ == "__main__":
    # Index robotic manuals
    robotic_manuals_indexer = FAISSIndexer("data/robotic_manuals.json", "robotic_manuals.faiss")
    if not robotic_manuals_indexer.index_exists():
        robotic_manuals_indexer.create_faiss_index()
    else:
        print(f"FAISS index already exists for robotic manuals at {robotic_manuals_indexer.index_path}")

    # Index troubleshooting logs
    troubleshooting_logs_indexer = FAISSIndexer("data/troubleshooting_logs.json", "troubleshooting_logs.faiss")
    if not troubleshooting_logs_indexer.index_exists():
        troubleshooting_logs_indexer.create_faiss_index()
    else:
        print(f"FAISS index already exists for troubleshooting logs at {troubleshooting_logs_indexer.index_path}")
