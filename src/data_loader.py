import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.knowledge_graph import KnowledgeGraph
from src.faiss_indexer import FAISSIndexer

class KnowledgeRetriever:
    def __init__(self):
        """Initializes FAISS and Knowledge Graph for knowledge retrieval."""
        self.faiss_indexer = FAISSIndexer("data/robotic_manuals.json", "models/faiss_index/robotic_manuals.faiss")
        self.knowledge_graph = KnowledgeGraph()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Used for query embeddings

    def get_query_embedding(self, query):
        """Encodes the query using Sentence Transformers."""
        return self.embedding_model.encode([query])

    def get_relevant_docs(self, query, top_k=3):
        """Retrieves documents from FAISS based on query similarity."""
        query_embedding = self.get_query_embedding(query)
        distances, indices = self.faiss_indexer.index.search(np.array(query_embedding, dtype="float32"), top_k)

        retrieved_docs = []
        for i in indices[0]:
            if i < len(self.faiss_indexer.documents):
                retrieved_docs.append(self.faiss_indexer.documents[i]["text"])
        
        return retrieved_docs

    def get_graph_relations(self, node_id):
        """Retrieves related entities from the Knowledge Graph."""
        return self.knowledge_graph.get_related_nodes(node_id)

if __name__ == "__main__":
    retriever = KnowledgeRetriever()
    
    query = "How do I reset a robotic arm?"
    faiss_results = retriever.get_relevant_docs(query)
    print(f"📘 FAISS Retrieved Docs: {faiss_results}")

    node_id = "manual_001"
    kg_results = retriever.get_graph_relations(node_id)
    print(f"🔗 Related Knowledge Graph Nodes: {kg_results}")
