import networkx as nx
import json

class KnowledgeGraph:
    def __init__(self, data_path="data/robotic_manuals.json"):
        self.graph = nx.Graph()
        self.data_path = data_path
        self.load_data()

    def load_data(self):
        """Loads documents and builds a knowledge graph."""
        with open(self.data_path, "r") as file:
            data = json.load(file)
            for doc in data["documents"]:
                self.graph.add_node(doc["id"], text=doc["text"])

    def get_related_nodes(self, node_id):
        """Returns related nodes from the knowledge graph."""
        return list(self.graph.neighbors(node_id))

if __name__ == "__main__":
    kg = KnowledgeGraph()
    print("Sample Graph Nodes:", list(kg.graph.nodes)[:5])
