import os
import asyncio
import numpy as np
from typing import List, Optional, Any, Tuple
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from app.core.logger_config import get_logger
from app.core.config import settings
from dotenv import load_dotenv
import httpx
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

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
        # --- API Identity ---
        self.groq_api_key = settings.GROQ_API_KEY
        self.anthropic_api_key = settings.ANTHROPIC_API_KEY
        
        # --- LLM Models ---
        self.model_name_llm = settings.LLM_MODEL
        self.model_name_ingest = settings.LLM_INGEST_MODEL
        
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

            # 3. Vision LLM (Multimodal)
            self.llm_vision = ChatGroq(
                model="llama-3.2-11b-vision-preview",
                groq_api_key=self.groq_api_key,
                temperature=0.1,
                max_retries=3,
                timeout=20
            )
            
            logger.info(f"Groq Cloud APIs Initialized (Reasoning: {self.model_name_llm} | Ingestion: {self.model_name_ingest} | Vision: llama-3.2-11b)")
        else:
            self.llm = None
            self.llm_ingest = None
            self.llm_vision = None
            logger.warning("GROQ_API_KEY not found in .env. LLM features will be disabled.")

        # --- Anthropic Configuration (Master Analyst) ---
        if self.anthropic_api_key:
            self.llm_analyst = ChatAnthropic(
                model="claude-3-5-sonnet-20240620",
                anthropic_api_key=self.anthropic_api_key,
                temperature=0.1,
                max_tokens=4096,
                # Enable Prompt Caching via beta headers
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
            )
            logger.info("Claude 3.5 Analyst Initialized (with Prompt Caching support).")
        else:
            self.llm_analyst = None
            logger.warning("ANTHROPIC_API_KEY not found. Claude Analyst will be offline.")
        
        # --- NEW: Ultra-HD Heterogeneous Ensemble (3,712-D Dense) ---
        self.model_names = [
            "BAAI/bge-m3",                                    # Powerhouse (Dense + Sparse + ColBERT)
            "google/canine-s",                               # Character-level (The Technical ID Shield)
            "microsoft/codebert-base",                        # Logic & Syntax specialist
            "sentence-transformers/all-mpnet-base-v2",        # High-Resolution Reasoning
            "sentence-transformers/all-MiniLM-L6-v2"          # Speed/Structural Anchor
        ]
        
        # --- Dynamic Offloading Check ---
        if settings.OFFLOAD_EMBEDDINGS:
            logger.info("📡 Model Offloading Enabled. Skipping local embedding model loading.")
            self.ensemble = {}
            self.sparse_ef = None
            self.clip_model = None
            return True

        # --- NEW: Sparse Matrix Encoder (32,768-D) ---
        # BGE-M3 is the industry standard for sparse lexical expansion
        self.sparse_ef = BGEM3EmbeddingFunction(
            model_name='BAAI/bge-m3', 
            device='cpu' # Switch to 'cuda' if GPU available
        )
        
        # --- Multi-Modal Engine (CLIP) ---
        self.clip_model_name = "sentence-transformers/clip-ViT-B-32"
        
        logger.info(f"Initializing Local Neural Fusion Ensemble (5x Models + CLIP)...")
        # Set cache to relative path for Linux/Docker compatibility
        os.environ["HF_HOME"] = settings.MODELS_CACHE_DIR
        
        # --- Parallel Loading for Fast Boot ---
        from concurrent.futures import ThreadPoolExecutor
        
        def load_model(name):
            short_name = name.split('/')[-1]
            logger.info(f"Loading Layer: {short_name}...")
            return short_name, SentenceTransformer(name)

        self.ensemble = {}
        all_models_to_load = self.model_names + [self.clip_model_name]
        
        logger.info(f"Spinning up Parallel Loading for {len(all_models_to_load)} layers...")
        with ThreadPoolExecutor(max_workers=len(all_models_to_load)) as executor:
            results = list(executor.map(load_model, all_models_to_load))
            
        for short_name, model in results:
            if "clip" in short_name.lower():
                self.clip_model = model
            else:
                self.ensemble[short_name] = model
        
        logger.info(f"✅ Neural Fusion Ensemble and CLIP fully loaded.")
        return True

    # --- OFF-LOADED INFERENCE CLIENT STUBS ---
    async def _post_remote(self, endpoint: str, json_data: dict) -> dict:
        url = f"{settings.EMBEDDING_SERVER_URL.rstrip('/')}{endpoint}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=json_data)
            resp.raise_for_status()
            return resp.json()

    async def _get_remote_hybrid_embeddings(self, text: str) -> Tuple[np.ndarray, dict]:
        """Calls offloaded model server to fetch both dense (3712-D) and sparse representations."""
        try:
            data = await self._post_remote("/embeddings/hybrid", {"text": text})
            dense_vector = np.array(data["dense"])
            sparse_vector = data["sparse"]
            return dense_vector, sparse_vector
        except Exception as e:
            logger.error(f"Failed to fetch remote hybrid embeddings: {e}. Falling back to zero-vectors.")
            return np.zeros(settings.DENSE_DIMENSION), {}

    async def _get_remote_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Calls offloaded model server to fetch batched dense (3712-D) embeddings."""
        try:
            data = await self._post_remote("/embeddings/batch", {"texts": texts})
            return data["vectors"]
        except Exception as e:
            logger.error(f"Failed to fetch remote batch embeddings: {e}. Falling back to zero-vectors.")
            return [[0.0] * settings.DENSE_DIMENSION for _ in texts]

    async def _get_remote_clip_embedding(self, input_data: Any) -> List[float]:
        """Calls offloaded model server to fetch multi-modal CLIP embedding."""
        try:
            if isinstance(input_data, str) and not os.path.exists(input_data):
                payload = {"text": input_data}
            else:
                path = str(input_data)
                if os.path.exists(path):
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        with open(path, "rb") as f:
                            url = f"{settings.EMBEDDING_SERVER_URL.rstrip('/')}/embeddings/clip"
                            resp = await client.post(url, files={"file": f})
                            resp.raise_for_status()
                            return resp.json()["vector"]
                payload = {"text": str(input_data)}
                
            data = await self._post_remote("/embeddings/clip", payload)
            return data["vector"]
        except Exception as e:
            logger.error(f"Failed to fetch remote CLIP embedding: {e}. Falling back to zero-vector.")
            return [0.0] * 512

    async def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for a single text using local ensemble or remote server."""
        res = await self.get_embeddings_batch([text])
        return res[0]

    async def get_hybrid_embeddings(self, text: str) -> Tuple[np.ndarray, dict]:
        """
        Generates the 3,712-D Dense Hybrid vector AND the 32k-D Sparse vector.
        """
        if settings.OFFLOAD_EMBEDDINGS:
            return await self._get_remote_hybrid_embeddings(text)
            
        # 1. Generate Dense Ensemble (Multi-Model Fusion)
        logger.info(f"Generating Hybrid Dense-Sparse Fingerprint...")
        tasks = [
            asyncio.to_thread(model.encode, [text], batch_size=1, show_progress_bar=False)
            for model in self.ensemble.values()
        ]
        
        # Parallel Execution for Dense
        vectors = await asyncio.gather(*tasks)
        dense_vector = np.concatenate(vectors, axis=1)[0]
        
        # 2. Generate Sparse Matrix (32,768-D)
        # BGE-M3 generates a dictionary of {token_id: weight}
        sparse_output = await asyncio.to_thread(self.sparse_ef, [text])
        sparse_vector = sparse_output[0]
        return dense_vector, sparse_vector

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """High-Throughput Neural Fusion across a batch (Local or Remote)."""
        if settings.OFFLOAD_EMBEDDINGS:
            return await self._get_remote_embeddings_batch(texts)
            
        batch_size = len(texts)
        logger.info(f"Executing Local Neural Fusion (Batch Size: {batch_size})...")
        tasks = [
            asyncio.to_thread(model.encode, texts, batch_size=64, show_progress_bar=False)
            for model in self.ensemble.values()
        ]
        
        # Parallel Execution
        vectors = await asyncio.gather(*tasks)
        ensemble_vectors = np.concatenate(vectors, axis=1).tolist()
        logger.info(f"Parallel Neural Fusion Complete.")
        return ensemble_vectors

    def calculate_token_interaction(self, query: str, document: str) -> float:
        """
        Simulates ColBERT Late Interaction (MaxSim).
        Calculates a token-level alignment score with high weights for technical terms.
        """
        import re
        q_tokens = set(re.findall(r'\b\w{3,}\b', query.lower()))
        d_tokens = set(re.findall(r'\b\w{3,}\b', document.lower()))
        
        if not q_tokens:
            return 0.0
            
        score = 0.0
        for q_t in q_tokens:
            if q_t in d_tokens:
                is_technical = q_t.upper() in query or q_t.upper() in document
                score += 5.0 if is_technical else 1.0
                
        return score / len(q_tokens)

    async def get_clip_embedding(self, input_data: Any) -> List[float]:
        """
        Generates a CLIP embedding for either text (str) or an image (PIL.Image or path).
        Dimension: 512
        """
        if settings.OFFLOAD_EMBEDDINGS:
            return await self._get_remote_clip_embedding(input_data)
            
        logger.info("Generating CLIP embedding (Multi-Modal)...")
        vector = await asyncio.to_thread(self.clip_model.encode, input_data)
        return vector.tolist()

    def get_model_name(self) -> str:
        return f"local_ensemble_5x + Groq:{self.model_name_llm}"

    def get_embedding_dimension(self) -> int:
        return settings.DENSE_DIMENSION

    async def call_llm(self, chain: Any, inputs: dict) -> str:
        """Calls Groq Cloud API for high-speed synthesis."""
        if not self.llm:
            return "[GROQ ERROR] API Key missing. Please check .env"
            
        try:
            # Construct prompt for Groq from the inputs
            context = inputs.get("context", "")
            question = inputs.get("question", "")
            confidence = inputs.get("confidence", "HIGH")
            confidence_score = inputs.get("confidence_score", 4)
            
            # --- Intelligence Rubric Prompting ---
            if confidence_score >= 5:
                confidence_instruction = "The sources provide EXCEPTIONAL evidence. Answer with absolute technical authority and high detail. Do not hold back.\n"
            elif confidence_score == 4:
                confidence_instruction = "The sources provide HIGH-quality evidence. Answer with authority and precision.\n"
            elif confidence_score == 3:
                confidence_instruction = "The sources provide MEDIUM-quality evidence. Answer what is verified and clearly identify any missing technical details or gaps.\n"
            elif confidence_score == 2:
                confidence_instruction = "WARNING: The information found is LOW-confidence or loose. State what you know but emphasize that evidence is weak. Use cautious language.\n"
            else:
                confidence_instruction = "CRITICAL WARNING: No direct evidence found. If you cannot answer based on sources, state that clearly. Do not speculate.\n"

            # Build conversation history block if available
            history = inputs.get("conversation_history", [])
            history_block = ""
            if history:
                history_lines = []
                for msg in history[-6:]:  # Last 3 turns
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    history_lines.append(f"{role}: {msg.get('content', '')}")
                history_block = "PREVIOUS CONVERSATION (for context on follow-up questions):\n" + "\n".join(history_lines) + "\n\n"

            # A+C Combined: Expert Reasoning + Chain-of-Thought synthesis
            prompt = (
                f"You are a Senior Technical Analyst who has just reviewed the following documentation regarding: '{question}'.\n\n"
                f"{history_block}"
                f"CONFIDENCE LEVEL: {confidence} ({confidence_score}/5)\n"
                f"INSTRUCTION: {confidence_instruction}\n"
                f"SOURCES:\n{context}\n\n"
                "THINKING (internal, do not show this):\n"
                "- Is this a follow-up to the conversation history?\n"
                "- Which sources are truly authoritative? Which are noise?\n"
                "- VISUAL EVIDENCE: If any source starts with '[VISUAL_ANALYSIS]', it is an image/diagram. Reference it clearly in your answer (e.g., 'As shown in the diagram...').\n"
                "- How do I bridge gaps between fragments?\n\n"
                "OUTPUT RULES:\n"
                "1. Answer directly and concisely. No introductory filler.\n"
                "2. If the sources contain technical IDs or codes, preserve them exactly.\n"
                "3. If multiple sources confirm a fact, state it once as a verified truth.\n"
                "4. If the documentation is insufficient, explicitly state what is missing.\n"
                "5. If referencing a visual source, mention it by its source metadata if available.\n\n"
                "ANSWER:"
            )

            # Choose the strongest model available
            primary_llm = self.llm_analyst if self.llm_analyst else self.llm
            if not primary_llm:
                return "[AI ERROR] No Reasoning Model available (Groq/Anthropic missing)."

            response = await asyncio.to_thread(primary_llm.invoke, prompt)
            raw_answer = response.content.strip()

            # --- PHASE 23: GRAPH-GROUNDED TRUTH GUARD ---
            if settings.ENABLE_GRAPH_TRUTH_GUARD and self.llm_analyst:
                raw_answer = await self.verify_against_graph(raw_answer, question)

            # Remove any "THINKING" block the LLM accidentally included
            if "THINKING" in raw_answer.upper() and "ANSWER" in raw_answer.upper():
                raw_answer = raw_answer.split("ANSWER:")[-1].strip()

            # Guardrails
            prohibited_phrases = ["as an ai model", "i don't have personal opinions", "my training data"]
            for phrase in prohibited_phrases:
                if phrase in raw_answer.lower():
                    raw_answer = raw_answer.replace(phrase, "[Filtered]")

            return raw_answer
            
        except Exception as e:
            logger.error(f"Groq API Error: {e}")
            return f"[Groq Synthesis Error] Details: {str(e)}"

    async def generate_synthetic_questions(self, text: str) -> List[str]:
        """Generates 3 synthetic questions using cloud-based Groq."""
        if not self.llm or not text or len(text.strip()) < 10:
            return []

        # Jittered delay to prevent simultaneous hits from multiple worker processes
        import random
        await asyncio.sleep(random.uniform(2.0, 5.0))  # Throttle to avoid Groq 429

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
            "IMPORTANT: Check for SOURCE BIAS. If all information comes from just one document/source, "
            "mark 'sufficient': false and suggest a query to find evidence from OTHER sources to ensure balance.\n"
            "Return a JSON object with exactly these fields:\n"
            "- 'sufficient': (boolean) True if no more search is needed.\n"
            "- 'missing_info': (string) What specific detail or diverse source is still missing.\n"
            "- 'suggested_query': (string) A targeted search query to find that missing or diverse info.\n"
            "- 'thought': (string) Your internal reasoning regarding the content AND the diversity of sources.\n"
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
        await asyncio.sleep(random.uniform(2.0, 4.0))  # Throttle to avoid Groq 429

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

    async def describe_image(self, image_bytes: bytes) -> str:
        """
        Phase 21: Vision Layer (Semantic Image Description)
        Uses Groq's multimodal vision model to describe technical diagrams/images.
        """
        if not self.llm_vision:
            return "Visual element (AI Vision Offline)"

        import base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Jittered delay to prevent simultaneous hits from multiple worker processes
        import random
        await asyncio.sleep(random.uniform(1.0, 3.0)) 

        from langchain_core.messages import HumanMessage
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this technical image/diagram in detail for a Knowledge Base. Focus on labels, connections, and the technical purpose. Be concise but highly accurate."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )

        try:
            logger.info("Generating technical description for image (Vision Phase)...")
            response = await asyncio.to_thread(self.llm_vision.invoke, [message])
            description = response.content.strip()
            logger.info(f"Vision Description Complete. Length: {len(description)}")
            return description
        except Exception as e:
            logger.error(f"Vision Description Error: {e}")
            return "Technical diagram/image from documentation."

    async def verify_against_graph(self, answer: str, original_query: str) -> str:
        """
        Audits the generated answer against the Knowledge Graph (Neo4j).
        If contradictions are found, the AI is forced to reconcile them.
        """
        from app.core.graph_manager import graph_manager
        
        # 1. Extract entities from the answer to check against the graph
        import re
        words = re.findall(r'\b[A-Z][A-Z0-9\-]{2,}\b', answer) # Look for technical acronyms/IDs
        entities = list(set(words))[:10]
        
        if not entities:
            return answer

        # 2. Fetch ground truth facts from Neo4j
        graph_facts = []
        for ent in entities:
            facts = graph_manager.get_related_facts(ent)
            graph_facts.extend(facts)
        
        if not graph_facts:
            return answer

        logger.info(f"Truth Guard: Auditing answer against {len(graph_facts)} graph facts...")
        
        # 3. Ask the Analyst to verify and reconcile
        verification_prompt = (
            "You are a Quality Assurance Auditor. You must verify a technical answer against the 'Neural Knowledge Graph' (Ground Truth).\n\n"
            f"ORIGINAL ANSWER:\n{answer}\n\n"
            f"NEURAL KNOWLEDGE GRAPH FACTS (Ground Truth):\n" + "\n".join([f"- {f}" for f in list(set(graph_facts))[:20]]) + "\n\n"
            "INSTRUCTIONS:\n"
            "1. If any fact in the graph directly contradicts the answer (e.g., different port numbers, different owners), REWRITE the answer to match the Graph facts.\n"
            "2. If the Graph facts add critical missing detail, incorporate it.\n"
            "3. If the answer is already consistent with the graph, return the answer UNCHANGED.\n"
            "4. Maintain the professional tone and concise structure.\n\n"
            "VERIFIED TECHNICAL ANSWER:"
        )

        try:
            response = await asyncio.to_thread(self.llm_analyst.invoke, verification_prompt)
            verified_answer = response.content.strip()
            if len(verified_answer) > 10:
                logger.info("Truth Guard: Answer verified and reconciled with Graph.")
                return verified_answer
            return answer
        except Exception as e:
            logger.error(f"Truth Guard Error: {e}")
            return answer

# Global instance
ai_manager = AIManager()

# Compatibility Exports
llm = ai_manager.llm
