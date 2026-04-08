import os
import sys
import asyncio

# Ensure project root is in sys.path for portable environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from celery import Celery
from dotenv import load_dotenv
from app.core.logger_config import get_logger
from app.core.ai_manager import ai_manager
# from app.core.chroma_client import get_chroma_collection
from app.core.milvus_client import milvus_client, init_milvus_collection, COLLECTION_NAME
from app.core.graph_manager import graph_manager
import polars as pl
from datetime import datetime, timezone
import uuid

# Initialize Logger
logger = get_logger("celery_tasks")

load_dotenv()

# Initialize Celery
app = Celery(
    'ai_memory_tasks',
    broker=os.getenv("REDIS_URL"),
    backend=os.getenv("REDIS_URL")
)

@app.task(
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True
)
def process_and_store_batch(self, chunks, metadata):
    """
    Background task to generate embeddings for a BATCH of chunks.
    Implements Hybrid AI Fallback and SOURCE AUTHORITY injection.
    """
    logger.info(f"Task Started: process_and_store_batch with {len(chunks)} chunks.")
    if not chunks:
        logger.warning("Received empty chunks list. Skipping ingestion commit.")
        return {"status": "skipped", "count": 0, "reason": "empty_chunks"}
    
    # Ensure Milvus is ready
    init_milvus_collection()
        
    try:
        total = len(chunks)
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': total, 'status': 'Initializing...'})
        
        # Determine Source Authority (X-Algo Weighting)
        source_type = metadata.get("type", "unknown")
        authority = 1.0 # Default
        if source_type == "official_doc":
            authority = 1.2
        elif source_type in ["chat_log", "action_trace"]:
            authority = 0.8
        
        metadata["authority"] = authority
        
        # AI Model Selection & Hydration (Hybrid RAG)
        model_name = ai_manager.get_model_name()
        metadata["embedding_model"] = model_name
        metadata["dimensions"] = ai_manager.get_embedding_dimension()
        
        logger.info(f"Using {model_name} for batch ingestion. Dimensions: {metadata['dimensions']}")

        # Generate Embeddings & Synthetic Questions
        embeddings_list = []
        enriched_metadata_list = []
        final_child_chunks = []
        
        sub_batch_size = 100
        for i in range(0, total, sub_batch_size):
            batch_end = min(i + sub_batch_size, total)
            sub_batch = chunks[i:batch_end]
            
            # Handle list of tuples (hierarchical) vs list of strings (legacy)
            # JSON serialization in Celery/Redis often converts tuples to lists
            is_hierarchical = (isinstance(sub_batch[0], (list, tuple)) and len(sub_batch[0]) == 2)
            sub_chunks = [c[0] if is_hierarchical else c for c in sub_batch]
            
            self.update_state(state='PROGRESS', meta={
                'current': i, 
                'total': total, 
                'status': f'Generating Embeddings & AI Questions ({i}-{batch_end}/{total})...'
            })
            
            # 1. Get Embeddings
            batch_vecs = asyncio.run(ai_manager.get_embeddings_batch(sub_chunks))
            embeddings_list.extend(batch_vecs)

            # 2. Get Synthetic Questions (Semantic Enrichment)
            async def get_questions_for_batch(texts):
                results = []
                for t in texts:
                    q_list = await ai_manager.generate_synthetic_questions(t)
                    results.append(q_list)
                    await asyncio.sleep(0.5)
                return results

            questions_batch = asyncio.run(get_questions_for_batch(sub_chunks))
            
            # 3. Extract Knowledge Graph Triplets (Phase 18)
            async def get_graph_data(texts):
                all_triplets = []
                # Only extract from a subset if batch is huge, or process all if quality matters
                for t in texts:
                    triplets = await ai_manager.extract_triplets(t)
                    all_triplets.append(triplets)
                    # Small delay to prevent LLM rate limits in sequential extraction
                    await asyncio.sleep(0.3)
                return all_triplets

            triplets_batch = asyncio.run(get_graph_data(sub_chunks))

            # 4. Create per-chunk metadata & Store Triplets
            for j, q_list in enumerate(questions_batch):
                chunk_meta = metadata.copy()
                chunk_meta["synthetic_questions"] = q_list
                
                # Add extracted triplets to the Global Graph
                for s, p, o in triplets_batch[j]:
                    graph_manager.add_relationship(s, p, o, {"source": metadata.get("source"), "chunk_idx": i+j})
                
                # Injected Parent Logic
                if is_hierarchical:
                    child_text = sub_batch[j][0]
                    parent_text = sub_batch[j][1]
                    chunk_meta["parent_content"] = parent_text
                    final_child_chunks.append(child_text)
                else:
                    final_child_chunks.append(sub_batch[j])
                    
                enriched_metadata_list.append(chunk_meta)

        # Prepare data for high-speed insertion into Milvus Lite
        # Phase 19: Vectorized Metadata Enrichment (Polars)
        df_meta = pl.DataFrame(enriched_metadata_list)
        
        # 1. Join synthetic questions into a searchable string
        if "synthetic_questions" in df_meta.columns:
            df_meta = df_meta.with_columns(
                pl.col("synthetic_questions").list.join(" | ")
            )
            
        # 2. Ensure all fields are Milvus-compatible (Stringify complex objects)
        for col, dtype in zip(df_meta.columns, df_meta.dtypes):
            if dtype in [pl.List, pl.Object, pl.Struct] and col != "embedding":
                 df_meta = df_meta.with_columns(pl.col(col).cast(pl.String))
                 
        # 3. Add global metadata (Timestamp)
        df_meta = df_meta.with_columns(
            pl.lit(datetime.now(timezone.utc).isoformat()).alias("created_at")
        )
        
        # 4. Final assembly for Milvus insertion
        clean_metadatas = df_meta.to_dicts()
        data_to_insert = []
        for j in range(len(clean_metadatas)):
            record = {
                "embedding": embeddings_list[j],
                "content": final_child_chunks[j],
                **clean_metadatas[j]
            }
            data_to_insert.append(record)

        # collection = get_chroma_collection()
        # collection.add(
        #     ids=ids,
        #     embeddings=embeddings_list,
        #     metadatas=clean_metadatas,
        #     documents=final_child_chunks
        # )
        
        milvus_client.insert(
            collection_name=COLLECTION_NAME,
            data=data_to_insert
        )
        
        logger.info(f"Task SUCCESS: Processed {len(chunks)} chunks using {model_name}.")
        return {"status": "success", "count": len(chunks), "model": model_name}
    except Exception as exc:
        logger.exception(f"Task FAILED: {exc}")
        raise self.retry(exc=exc)

@app.task(
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True
)
def process_and_store_memory(self, text, metadata):
    """
    Individual task with deduplication and Hybrid AI Fallback.
    """
    logger.info("Task Started: process_and_store_memory.")
    try:
        self.update_state(state='PROGRESS', meta={'status': 'Generating Embedding...'})
        # Determine model
        model_name = ai_manager.get_model_name()
        metadata["embedding_model"] = model_name
        metadata["dimensions"] = ai_manager.get_embedding_dimension()

        # Generate Embedding
        vector = asyncio.run(ai_manager.get_embeddings(text))

        # Generate Synthetic Questions (Phase 4)
        questions = asyncio.run(ai_manager.generate_synthetic_questions(text))
        metadata["synthetic_questions"] = questions

        self.update_state(state='PROGRESS', meta={'status': 'Storing in Vector DB...'})
        
        # Ensure Milvus is ready
        init_milvus_collection()

        # Clean metadata for Milvus Lite (Polars)
        df_temp = pl.DataFrame([metadata])
        if "synthetic_questions" in df_temp.columns:
            df_temp = df_temp.with_columns(pl.col("synthetic_questions").list.join(" | "))
            
        for col, dtype in zip(df_temp.columns, df_temp.dtypes):
            if dtype in [pl.List, pl.Object, pl.Struct]:
                 df_temp = df_temp.with_columns(pl.col(col).cast(pl.String))
                 
        df_temp = df_temp.with_columns(
            pl.lit(datetime.now(timezone.utc).isoformat()).alias("created_at")
        )
        clean_meta = df_temp.to_dicts()[0]
        
        # collection = get_chroma_collection()
        # collection.add(
        #     ids=[str(uuid.uuid4())],
        #     embeddings=[vector],
        #     metadatas=[clean_meta],
        #     documents=[text]
        # )

        milvus_client.insert(
            collection_name=COLLECTION_NAME,
            data=[{
                "embedding": vector,
                "content": text,
                **clean_meta
            }]
        )
        
        logger.info(f"Task COMPLETED: process_and_store_memory success using {model_name}.")
        return {"status": "success", "model": model_name}
    except Exception as exc:
        logger.exception(f"Task FAILED: process_and_store_memory error: {exc}")
        raise self.retry(exc=exc)
