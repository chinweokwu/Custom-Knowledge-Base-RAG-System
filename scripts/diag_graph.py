import os
import sys
sys.path.append(os.getcwd())
from app.core.graph_manager import graph_manager

def diag():
    print(f"Nodes in Graph: {list(graph_manager.graph.nodes)}")
    print(f"Edges in Graph: {list(graph_manager.graph.edges(data=True))}")
    
    test_keys = ['nebula', 'project', 'protocol', 'cortex']
    for key in test_keys:
        facts = graph_manager.get_related_facts(key)
        print(f"Lookup '{key}': Found {len(facts)} facts -> {facts}")

if __name__ == "__main__":
    diag()
