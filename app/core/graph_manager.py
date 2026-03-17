import os
import networkx as nx
import pickle
from typing import List, Tuple, Dict
from app.core.logger_config import get_logger

logger = get_logger("graph_manager")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPH_PATH = os.path.join(BASE_DIR, "graph_store.pkl")

class GraphManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GraphManager, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.graph = nx.MultiDiGraph()
        self.last_load_time = 0
        self.load_graph()

    def load_graph(self):
        """Loads the graph from the D: drive pickle file if it has changed."""
        if os.path.exists(GRAPH_PATH):
            try:
                mtime = os.path.getmtime(GRAPH_PATH)
                if mtime > self.last_load_time:
                    with open(GRAPH_PATH, "rb") as f:
                        self.graph = pickle.load(f)
                    self.last_load_time = mtime
                    logger.info(f"Knowledge Graph reloaded. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
            except Exception as e:
                logger.error(f"Failed to load graph: {e}")
                if not hasattr(self, 'graph'):
                    self.graph = nx.MultiDiGraph()
        else:
            logger.info("No existing Knowledge Graph found.")
            self.graph = nx.MultiDiGraph()

    def save_graph(self):
        """Saves the graph to the D: drive pickle file (Zero-Footprint)."""
        try:
            with open(GRAPH_PATH, "wb") as f:
                pickle.dump(self.graph, f)
            logger.info(f"Knowledge Graph saved to {GRAPH_PATH}")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    def add_relationship(self, subject: str, predicate: str, object_: str, source_metadata: dict = None):
        """Adds a directed relationship (triplet) to the graph."""
        s = subject.strip().lower()
        o = object_.strip().lower()
        p = predicate.strip().lower()
        
        self.graph.add_edge(s, o, relation=p, **(source_metadata or {}))
        # We don't save on every edge for performance, but in this setup, we might want to for persistence
        # In a real app, save on batch completion.
        self.save_graph()

    def get_related_facts(self, entity: str, depth: int = 1) -> List[str]:
        """Traverses the graph to find neighbors and their relations."""
        search_key = entity.strip().lower()
        
        # Find all nodes that contains or are contained in the search_key
        # This handles 'Nebula' matching 'Project Nebula'
        target_nodes = [n for n in self.graph.nodes if search_key in n or n in search_key]
        
        if target_nodes:
            logger.info(f"Fuzzy Match: '{search_key}' -> {target_nodes}")
        
        facts = []
        for node in target_nodes:
            # Support depth 1 for now (simple neighborhood expansion)
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                logger.info(f"Found {len(neighbors)} neighbors for node '{node}': {neighbors}")
            
            for neighbor in neighbors:
                edge_data = self.graph.get_edge_data(node, neighbor)
                for _, data in edge_data.items():
                    rel = data.get("relation", "is related to")
                    facts.append(f"{node} {rel} {neighbor}")
            
            # Also look for reverse relationships
            preds = list(self.graph.predecessors(node))
            for pred in preds:
                edge_data = self.graph.get_edge_data(pred, node)
                for _, data in edge_data.items():
                    rel = data.get("relation", "is related to")
                    facts.append(f"{pred} {rel} {node}")
                
        return list(set(facts))

# Global instance
graph_manager = GraphManager()
