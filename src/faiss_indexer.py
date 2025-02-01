import faiss
import json
import numpy as np

class FAISSIndexer:
    def __init__(self, data_path="data/robotic_manuals.json", index_path="data/embeddings/robotic_manuals.faiss"):
        self.data_path = data_path
        self.index_path = index_path
        self.index = None
        self.documents = []
    
    def load_data(self):
        """Loads knowledge base from a JSON file and extracts embeddings."""
        with open(self.data_path, "r") as file:
            data = json.load(file)
            self.documents = data["documents"]
        return self.documents

    def create_faiss_index(self):
        """Creates a FAISS index from precomputed embeddings."""
        if not self.documents:
            self.load_data()
        
        dim = len(self.documents[0]["embedding"])
        self.index = faiss.IndexFlatL2(dim)

        for doc in self.documents:
            vector = np.array(doc["embedding"], dtype="float32").reshape(1, -1)
            self.index.add(vector)

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        print(f"✅ FAISS index saved at {self.index_path}")

    def get_relevant_docs(self, query_vector, top_k=3):
        """Retrieves top-k most relevant documents using FAISS."""
        if self.index is None:
            self.index = faiss.read_index(self.index_path)

        distances, indices = self.index.search(query_vector.reshape(1, -1), top_k)
        return [self.documents[i]["text"] for i in indices[0]]

if __name__ == "__main__":
    indexer = FAISSIndexer()
    indexer.create_faiss_index()

    # Example query vector (Replace with real embeddings from a language model)
    query_vector = np.array([0.12, 0.32, 0.52, 0.72, 0.92], dtype="float32")
    results = indexer.get_relevant_docs(query_vector)
    print("Relevant Documents:", results)
