import faiss
import json
import networkx as nx

class KnowledgeRetriever:
    def __init__(self, knowledge_base_path, embedding_dim=768):
        self.knowledge_base_path = knowledge_base_path
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.knowledge_graph = nx.Graph()
        self.documents = []

    def load_data(self):
        """Loads knowledge base from a JSON file and indexes it."""
        with open(self.knowledge_base_path, "r") as file:
            data = json.load(file)
            for i, doc in enumerate(data["documents"]):
                self.documents.append(doc["text"])
                vector = doc["embedding"]
                self.index.add(vector.reshape(1, -1))
                self.knowledge_graph.add_node(doc["id"], text=doc["text"])

    def get_relevant_docs(self, query_vector, top_k=3):
        """Retrieves the top-k most relevant documents using FAISS."""
        distances, indices = self.index.search(query_vector.reshape(1, -1), top_k)
        return [self.documents[i] for i in indices[0]]

    def visualize_knowledge_graph(self):
        """Generates a visualization of the knowledge graph."""
        nx.draw(self.knowledge_graph, with_labels=True)
