import os
from neo4j import GraphDatabase
from app.core.logger_config import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger("neo4j_client")

class Neo4jClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Neo4jClient, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        # Use localhost for scripts running on host, 'neo4j' for container-to-container
        # We try both or rely on ENV
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def query(self, cypher, parameters=None):
        if not self.driver:
            logger.error("Neo4j driver not initialized.")
            return []
        
        with self.driver.session() as session:
            result = session.run(cypher, parameters)
            return [record for record in result]

    def execute_write(self, cypher, parameters=None):
        if not self.driver:
            logger.error("Neo4j driver not initialized.")
            return
        
        with self.driver.session() as session:
            session.run(cypher, parameters)

# Global instance
neo4j_client = Neo4jClient()
