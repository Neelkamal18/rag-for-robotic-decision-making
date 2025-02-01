import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from src.knowledge_graph import KnowledgeGraph
from src.faiss_indexer import FAISSIndexer

FAISS_INDEX_PATH = "models/faiss_index/robotic_manuals.faiss"

class KnowledgeRetriever:
    def __init__(self):
        """Initializes FAISS and Knowledge Graph for knowledge retrieval."""
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"Error: FAISS index file {FAISS_INDEX_PATH} not found. Ensure FAISS indexing is complete.")

        try:
            self.faiss_indexer = FAISSIndexer("data/robotic_manuals.json", FAISS_INDEX_PATH)
            self.knowledge_graph = KnowledgeGraph()
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Used for query embeddings
            print("Knowledge Retriever Initialized Successfully")
        except Exception as e:
            raise RuntimeError(f"Error initializing KnowledgeRetriever: {str(e)}")

    def get_query_embedding(self, query):
        """Encodes the query using Sentence Transformers."""
        return self.embedding_model.encode([query])

    def get_relevant_docs(self, query, top_k=3):
        """Retrieves documents from FAISS based on query similarity."""
        try:
            query_embedding = self.get_query_embedding(query)
            distances, indices = self.faiss_indexer.index.search(np.array(query_embedding, dtype="float32"), top_k)

            retrieved_docs = []
            for i in indices[0]:
                if i < len(self.faiss_indexer.documents):
                    retrieved_docs.append(self.faiss_indexer.documents[i]["text"])

            if not retrieved_docs:
                print("No relevant documents found. Proceeding without FAISS results.")
            
            return retrieved_docs
        except Exception as e:
            print(f"Error retrieving documents from FAISS: {str(e)}")
            return []

    def get_graph_relations(self, node_id):
        """Retrieves related entities from the Knowledge Graph."""
        try:
            related_nodes = self.knowledge_graph.get_related_nodes(node_id)
            if not related_nodes:
                print(f"⚠️ No related nodes found for {node_id}.")
            return related_nodes
        except Exception as e:
            print(f"Error retrieving Knowledge Graph nodes: {str(e)}")
            return []

if __name__ == "__main__":
    retriever = KnowledgeRetriever()
    
    query = "How do I reset a robotic arm?"
    faiss_results = retriever.get_relevant_docs(query, top_k=5)  # Allow dynamic `top_k`
    print(f"📘 FAISS Retrieved Docs: {faiss_results}")

    node_id = "manual_001"
    kg_results = retriever.get_graph_relations(node_id)
    print(f"🔗 Related Knowledge Graph Nodes: {kg_results}")
