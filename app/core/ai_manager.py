import os
import asyncio
import numpy as np
from typing import List, Optional, Any, Tuple
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from app.core.logger_config import get_logger
from dotenv import load_dotenv
import httpx

load_dotenv()
logger = get_logger("ai_manager")

class AIManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIManager, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        # --- Groq Configuration (Privacy-First Cloud) ---
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.model_name_llm = "llama-3.3-70b-versatile"
        self.model_name_ingest = "llama-3.1-8b-instant"  # Faster, higher-limit model for bulk data
        
        # --- LangChain Compatibility Layer ---
        if self.groq_api_key:
            # 1. Reasoning LLM (Heavy, High-IQ)
            self.llm = ChatGroq(
                model=self.model_name_llm, 
                groq_api_key=self.groq_api_key,
                temperature=0.1,
                max_retries=6,   
                timeout=30       
            )
            
            # 2. Ingestion LLM (Light, High-Throughput)
            # This handles bulk tasks like question generation to save your TPD quota
            self.llm_ingest = ChatGroq(
                model=self.model_name_ingest,
                groq_api_key=self.groq_api_key,
                temperature=0.1,
                max_retries=3,
                timeout=20
            )
            
            logger.info(f"Groq Cloud APIs Initialized (Reasoning: {self.model_name_llm} | Ingestion: {self.model_name_ingest})")
        else:
            self.llm = None
            self.llm_ingest = None
            logger.warning("GROQ_API_KEY not found in .env. LLM features will be disabled.")
        
        # --- Neural Fusion Ensemble (5x MiniLM) - 100% Local ---
        self.model_names = [
            "sentence-transformers/all-MiniLM-L6-v2",         # General Purpose
            "sentence-transformers/paraphrase-MiniLM-L6-v2",  # Semantic Clarity
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",# Technical QA/Codes
            "sentence-transformers/all-MiniLM-L12-v2",        # Higher-IQ Depth
            "sentence-transformers/msmarco-MiniLM-L6-cos-v5"  # High-Precision Search
        ]
        
        logger.info(f"Initializing Local Neural Fusion Ensemble (5x Models)...")
        # Set cache to D Drive
        # Set cache to relative path for Linux/Docker compatibility
        os.environ["HF_HOME"] = os.getenv("HF_HOME", "./models_cache")
        
        self.ensemble = {}
        for name in self.model_names:
            short_name = name.split('/')[-1]
            logger.info(f"Loading Ensemble Layer: {short_name}...")
            self.ensemble[short_name] = SentenceTransformer(name)
        
        return True

    async def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for a single text using local ensemble."""
        res = await self.get_embeddings_batch([text])
        return res[0]

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """High-Throughput Local Neural Fusion across a batch."""
        batch_size = len(texts)
        logger.info(f"Executing Local Neural Fusion (Batch Size: {batch_size})...")
        
        vectors = []
        for i, (short_name, model) in enumerate(self.ensemble.items()):
            logger.info(f"[{i+1}/5] Processing Neural Layer: {short_name}...")
            # to_thread keeps the event loop alive during intensive CPU work
            vec = await asyncio.to_thread(model.encode, texts, batch_size=64, show_progress_bar=False)
            vectors.append(vec)
        
        # Concatenate into 1920-dimension vectors
        fusion_batch = np.concatenate(vectors, axis=1).tolist()
        logger.info(f"Local Neural Fusion Complete.")
        return fusion_batch

    def get_model_name(self) -> str:
        return f"local_ensemble_5x + Groq:{self.model_name_llm}"

    def get_embedding_dimension(self) -> int:
        return 1920 # 5x MiniLM (384 * 5)

    async def call_llm(self, chain: Any, inputs: dict) -> str:
        """Calls Groq Cloud API for high-speed synthesis."""
        if not self.llm:
            return "[GROQ ERROR] API Key missing. Please check .env"
            
        try:
            # Construct prompt for Groq from the inputs
            context = inputs.get("context", "")
            question = inputs.get("question", "")
            
            # Using the LangChain integration directly for synthesis
            prompt = (
                f"You are a Lead Engineer. Use the following context to answer the question: {question}\n\n"
                f"CONTEXT:\n{context}\n\n"
                "Provide a detailed technical synthesis. Focus on precision and privacy."
            )
            
            # Use invoke for modern LangChain compatibility
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Groq API Error: {e}")
            return f"[Groq Synthesis Error] Details: {str(e)}"

    async def generate_synthetic_questions(self, text: str) -> List[str]:
        """Generates 3 synthetic questions using cloud-based Groq."""
        if not self.llm or not text or len(text.strip()) < 10:
            return []

        # Jittered delay to prevent simultaneous hits from multiple worker processes
        import random
        await asyncio.sleep(random.uniform(0.1, 0.8))

        prompt = (
            "You are an expert Lead Engineer. Given the following data fragment, "
            "generate 3 brief questions that a human would ask to find this data. "
            "Output exactly 3 questions, one per line, no numbering. Output ONLY the questions.\n\n"
            f"DATA:\n{text}\n\n"
            "QUESTIONS:"
        )

        try:
            # Shift to ingest-optimized model
            response = await asyncio.to_thread(self.llm_ingest.invoke, prompt)
            content = response.content
            questions = [q.strip() for q in content.split('\n') if q.strip()]
            import re
            questions = [re.sub(r'^\d+\.\s*', '', q).strip() for q in questions if q.strip()]
            return questions[:3]
        except Exception as e:
            if "429" in str(e):
                logger.warning("⚠️ Groq Ingest Quota Exceeded. Skipping synthetic questions for this chunk.")
            else:
                logger.error(f"Groq Question Generation Error: {e}")
            return []
            
    async def generate_hyde_document(self, query: str) -> str:
        """
        Phase 15: HyDE (Hypothetical Document Embeddings)
        Generates a hallucinated, perfect technical answer to the user's query.
        This fake answer is then vectorized and used to search the DB, matching
        "Answer-to-Answer" semantics strictly.
        """
        if not self.llm or not query:
            return ""

        prompt = (
            "You are an expert technical documentation writer. "
            f"A user has asked the following question: '{query}'. "
            "Write a very brief, highly factual paragraph containing the exact technical answer "
            "as it would appear in an official manual or documentation page. "
            "Do not include intro/outro filler. Just write the factual answer directly."
        )

        try:
            logger.info("Generating HyDE (Hypothetical Document) for query...")
            # Use to_thread to keep async loop unlocked
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            hyde_doc = response.content.strip()
            logger.info(f"HyDE Generation Complete. Preview: {hyde_doc[:100]}...")
            return hyde_doc
        except Exception as e:
            logger.error(f"HyDE Generation Error: {e}")
            return ""

    async def evaluate_context_sufficiency(self, query: str, context: str) -> dict:
        """
        Phase 17: Agentic Reasoning (Self-Evaluation)
        Asks the LLM if the provided context is enough to fully answer the query.
        Returns a JSON-compatible dict with evaluation results.
        """
        if not self.llm:
            return {"sufficient": True, "reason": "LLM Offline"}

        prompt = (
            "You are a critical Technical Auditor. Analyze the following retrieved context against the user question.\n\n"
            f"USER QUESTION: {query}\n\n"
            f"RETRIEVED CONTEXT:\n{context}\n\n"
            "Evaluate if this context contains 100% of the information needed to answer accurately.\n"
            "Return a JSON object with exactly these fields:\n"
            "- 'sufficient': (boolean) True if no more search is needed.\n"
            "- 'missing_info': (string) What specific detail is still missing (if any).\n"
            "- 'suggested_query': (string) A targeted search query to find that missing info.\n"
            "- 'thought': (string) Your internal reasoning (brief).\n"
            "Return ONLY the JSON. No conversation."
        )

        try:
            logger.info("Evaluating context sufficiency (Agentic Phase)...")
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            content = response.content.strip()
            
            # Basic JSON extraction logic
            import json
            import re
            
            # Clean up potential markdown formatting
            clean_json = re.search(r'\{.*\}', content, re.DOTALL)
            if clean_json:
                data = json.loads(clean_json.group(0))
                logger.info(f"Sufficiency Evaluation: {data.get('sufficient')} | Thought: {data.get('thought')}")
                return data
            return {"sufficient": True, "reason": "Parsing failed, falling back to one-shot."}
            
        except Exception as e:
            logger.error(f"Sufficiency Evaluation Error: {e}")
            return {"sufficient": True, "reason": str(e)}

    async def extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Phase 18: GraphRAG (Triplet Extraction)
        Parses text into (Subject, Relation, Object) triplets for the Knowledge Graph.
        """
        if not self.llm or not text or len(text.strip()) < 20:
            return []

        # Jittered delay to prevent simultaneous hits from multiple worker processes
        import random
        await asyncio.sleep(random.uniform(0.1, 0.5))

        prompt = (
            "You are a Knowledge Graph Engineer. Extract exactly 3-5 high-value technical relationships from the text below.\n"
            "Focus on entities like Systems, Protocols, Identifiers, and their Relationships.\n\n"
            "Format: Subject | Relation | Object\n"
            "Example: Project Nebula | uses | Cortex-X Protocol\n\n"
            f"TEXT:\n{text}\n\n"
            "TRIPLETS:"
        )

        try:
            logger.info("Extracting Knowledge Graph triplets...")
            # Shift to ingest-optimized model
            response = await asyncio.to_thread(self.llm_ingest.invoke, prompt)
            lines = response.content.strip().split('\n')
            
            triplets = []
            for line in lines:
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) == 3:
                        triplets.append((parts[0], parts[1], parts[2]))
            
            logger.info(f"Extracted {len(triplets)} triplets.")
            return triplets
        except Exception as e:
            if "429" in str(e):
                logger.warning("⚠️ Groq Ingest Quota Exceeded. Skipping GraphRAG triplets for this chunk.")
            else:
                logger.error(f"Triplet Extraction Error: {e}")
            return []

# Global instance
ai_manager = AIManager()

# Compatibility Exports
llm = ai_manager.llm
