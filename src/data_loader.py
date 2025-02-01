import json
import numpy as np
from src.faiss_indexer import FAISSIndexer

class KnowledgeRetriever:
    def __init__(self, data_path="data/robotic_manuals.json", index_path="data/embeddings/robotic_manuals.faiss"):
        self.indexer = FAISSIndexer(data_path, index_path)
        self.documents = self.indexer.load_data()

    def get_relevant_docs(self, query_vector, top_k=3):
        """Retrieves top-k most relevant documents using FAISS."""
        return self.indexer.get_relevant_docs(query_vector, top_k)

if __name__ == "__main__":
    retriever = KnowledgeRetriever()
    query_vector = np.array([0.12, 0.32, 0.52, 0.72, 0.92], dtype="float32")
    results = retriever.get_relevant_docs(query_vector)
    print("Top Relevant Documents:", results)
