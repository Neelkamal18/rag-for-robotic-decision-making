import networkx as nx
import json
import os
import pickle

GRAPH_DIR = "models/knowledge_graph/"
GRAPH_FILE = os.path.join(GRAPH_DIR, "knowledge_graph.pkl")

# Ensure the knowledge graph directory exists
os.makedirs(GRAPH_DIR, exist_ok=True)

class KnowledgeGraph:
    def __init__(self, data_path="data/robotic_manuals.json"):
        """Initializes or loads a Knowledge Graph."""
        self.graph = nx.Graph()
        self.data_path = data_path

        if os.path.exists(GRAPH_FILE):
            print(f"Loading existing Knowledge Graph from {GRAPH_FILE}...")
            if not self.load_graph():
                print("Failed to load Knowledge Graph. Rebuilding from source...")
                self.load_data()
                self.save_graph()
        else:
            print("Creating a new Knowledge Graph from data...")
            self.load_data()
            self.save_graph()

    def load_data(self):
        """Loads documents and builds a knowledge graph."""
        if not os.path.exists(self.data_path):
            print(f"Error: Data file {self.data_path} not found.")
            return False

        try:
            with open(self.data_path, "r") as file:
                data = json.load(file)
                if "documents" not in data:
                    print(f"Error: Invalid JSON format in {self.data_path}. Missing 'documents' key.")
                    return False

                for doc in data["documents"]:
                    self.graph.add_node(doc["id"], text=doc["text"])

                if len(self.graph.nodes) == 0:
                    print("Warning: No valid nodes found in the dataset.")
                    return False

            print(f"Knowledge Graph created with {len(self.graph.nodes)} nodes.")
            return True
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {self.data_path}.")
            return False

    def save_graph(self):
        """Saves the Knowledge Graph to a file."""
        if len(self.graph.nodes) == 0:
            print("Warning: Empty Knowledge Graph. Skipping save operation.")
            return

        with open(GRAPH_FILE, "wb") as file:
            pickle.dump(self.graph, file)
        print(f"Knowledge Graph successfully saved at {GRAPH_FILE}.")

    def load_graph(self):
        """Loads a saved Knowledge Graph."""
        try:
            with open(GRAPH_FILE, "rb") as file:
                self.graph = pickle.load(file)

            if len(self.graph.nodes) == 0:
                print("Warning: Loaded Knowledge Graph is empty. Rebuilding from source...")
                return False

            print(f"Successfully loaded Knowledge Graph with {len(self.graph.nodes)} nodes.")
            return True
        except (EOFError, pickle.UnpicklingError):
            print(f"Error: Corrupted Knowledge Graph file. Rebuilding from source...")
            return False

    def get_related_nodes(self, node_id):
        """Returns related nodes from the Knowledge Graph."""
        if node_id in self.graph:
            return list(self.graph.neighbors(node_id))
        else:
            print(f"Warning: Node '{node_id}' not found in the Knowledge Graph.")
            return []

if __name__ == "__main__":
    kg = KnowledgeGraph()
    print("🔍 Sample Graph Nodes:", list(kg.graph.nodes)[:5])
