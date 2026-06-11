import os
from typing import List, Tuple, Dict
from app.core.logger_config import get_logger
from app.core.neo4j_client import neo4j_client

logger = get_logger("graph_manager")

class GraphManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GraphManager, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        # No internal graph object needed, Neo4j handles it
        pass

    def add_relationship(self, subject: str, predicate: str, object_: str, source_metadata: dict = None):
        """Adds a directed relationship (triplet) to Neo4j."""
        s = subject.strip().lower()
        o = object_.strip().lower()
        # Clean predicate for Neo4j relationship type (must be alphanumeric/underscore)
        p = "".join(c if c.isalnum() else "_" for c in predicate.strip().upper())
        if not p: p = "RELATED_TO"
        
        cypher = (
            f"MERGE (s:Entity {{id: $s_id}}) "
            f"SET s.name = $s_name "
            f"MERGE (o:Entity {{id: $o_id}}) "
            f"SET o.name = $o_name "
            f"MERGE (s)-[r:{p}]->(o) "
            f"SET r += $meta"
        )
        params = {
            "s_id": s,
            "s_name": subject.strip(),
            "o_id": o,
            "o_name": object_.strip(),
            "meta": source_metadata or {}
        }
        try:
            neo4j_client.execute_write(cypher, params)
        except Exception as e:
            logger.error(f"Failed to add Neo4j relationship: {e}")

    def get_related_facts(self, entity: str, depth: int = 1) -> List[str]:
        """Traverses the Neo4j graph to find neighbors and their relations."""
        search_key = entity.strip().lower()
        
        # Cypher for multi-hop relationship retrieval
        cypher = (
            "MATCH (e:Entity) "
            "WHERE e.id CONTAINS $key OR $key CONTAINS e.id "
            "MATCH (e)-[r]-(neighbor) "
            "RETURN e.name as s, type(r) as p, neighbor.name as o"
        )
        
        try:
            records = neo4j_client.query(cypher, {"key": search_key})
            facts = []
            for record in records:
                facts.append(f"{record['s']} {record['p'].lower().replace('_', ' ')} {record['o']}")
            return list(set(facts))
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            return []

    def get_graph_data(self) -> Dict[str, List]:
        """Returns the graph in a format suitable for Vis.js or D3.js."""
        cypher_nodes = "MATCH (n:Entity) RETURN n.id as id, n.name as label LIMIT 500"
        cypher_edges = "MATCH (s:Entity)-[r]->(o:Entity) RETURN s.id as from, o.id as to, type(r) as label LIMIT 1000"
        
        try:
            nodes_res = neo4j_client.query(cypher_nodes)
            edges_res = neo4j_client.query(cypher_edges)
            
            nodes = [{"id": r["id"], "label": r["label"]} for r in nodes_res]
            edges = [{"from": r["from"], "to": r["to"], "label": r["label"].lower().replace('_', ' '), "arrows": "to"} for r in edges_res]
            
            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            logger.error(f"Failed to fetch graph data: {e}")
            return {"nodes": [], "edges": []}

    def purge_source_relations(self, source: str):
        """Deletes all relationships and orphan nodes associated with a source."""
        if not source:
            return
        cypher_rels = "MATCH ()-[r]->() WHERE r.source = $source DELETE r"
        cypher_nodes = "MATCH (n:Entity) WHERE not (n)-[]-() DELETE n"
        try:
            neo4j_client.execute_write(cypher_rels, {"source": source})
            neo4j_client.execute_write(cypher_nodes)
            logger.info(f"🗑️ Purged old Graph relationships and orphan nodes for source: '{source}'")
        except Exception as e:
            logger.error(f"Failed to purge Graph data for source '{source}': {e}")

# Global instance
graph_manager = GraphManager()
