import os
import sys
import asyncio
from celery import Celery
from app.core.logger_config import get_logger
from app.core.ai_manager import ai_manager
from app.core.milvus_client import milvus_client, init_milvus_collection, COLLECTION_NAME, VISUAL_COLLECTION_NAME
from app.core.graph_manager import graph_manager
from app.core.config import settings
import polars as pl
from datetime import datetime, timezone
import uuid

from dotenv import load_dotenv

# Initialize Logger
logger = get_logger("celery_tasks")

load_dotenv()

# Initialize Celery
celery_app = Celery(
    'ai_memory_tasks',
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

# Route tasks to proper queues for performance isolation
celery_app.conf.update(
    task_routes={
        'app.services.tasks.process_file_ingestion': {'queue': 'heavy_ingest'},
        'app.services.tasks.process_and_store_batch': {'queue': 'heavy_ingest'},
        'app.services.tasks.process_and_store_memory': {'queue': 'default'},
    }
)

def _process_chunks(self, chunks, metadata):
    """
    Inner helper to generate embeddings, synthetic questions, and triplets for chunks, 
    and store them in Milvus.
    """
    logger.info(f"Processing {len(chunks)} chunks for database insertion.")
    if not chunks:
        logger.warning("Received empty chunks list. Skipping ingestion commit.")
        return {"status": "skipped", "count": 0, "reason": "empty_chunks"}
    
    # Ensure Milvus is ready
    init_milvus_collection()
    
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
    enriched_metadata_list = []
    final_child_chunks = []
    batch_dense = []
    batch_sparse = []
    
    sub_batch_size = 100
    for i in range(0, total, sub_batch_size):
        batch_end = min(i + sub_batch_size, total)
        sub_batch = chunks[i:batch_end]
        
        # Handle list of tuples (hierarchical) vs list of strings (legacy)
        is_hierarchical = (isinstance(sub_batch[0], (list, tuple)) and len(sub_batch[0]) == 2)
        sub_chunks = [c[0] if is_hierarchical else c for c in sub_batch]
        
        self.update_state(state='PROGRESS', meta={
            'current': i, 
            'total': total, 
            'status': f'Generating Embeddings & AI Questions ({i}-{batch_end}/{total})...'
        })
        
        # 1. Get Hybrid Embeddings (Dense 4k + Sparse 32k)
        async def get_hybrid_batch(texts):
            tasks = [ai_manager.get_hybrid_embeddings(t) for t in texts]
            return await asyncio.gather(*tasks)

        logger.info(f"Generating Hybrid (Dense+Sparse) Vectors for batch of {len(sub_chunks)}...")
        hybrid_results = asyncio.run(get_hybrid_batch(sub_chunks))
        batch_dense.extend([r[0].tolist() for r in hybrid_results])
        batch_sparse.extend([r[1] for r in hybrid_results])

        skip_enrichment = metadata.get("skip_enrichment", False)
        if skip_enrichment:
            questions_batch = [[] for _ in sub_chunks]
            triplets_batch = [[] for _ in sub_chunks]
        else:
            # 2 & 3. Get Synthetic Questions & Triplets in parallel with throttle
            async def process_chunks_parallel(texts):
                semaphore = asyncio.Semaphore(8) 
                
                async def process_single(t):
                    async with semaphore:
                        q_task = ai_manager.generate_synthetic_questions(t)
                        g_task = ai_manager.extract_triplets(t)
                        return await asyncio.gather(q_task, g_task)

                tasks = [process_single(t) for t in texts]
                results = await asyncio.gather(*tasks)
                return results

            logger.info(f"Launching throttled parallel AI enrichment for {len(sub_chunks)} chunks...")
            parallel_results = asyncio.run(process_chunks_parallel(sub_chunks))
            questions_batch = [r[0] for r in parallel_results]
            triplets_batch = [r[1] for r in parallel_results]

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
    df_meta = pl.DataFrame(enriched_metadata_list)
    
    # 1. Join synthetic questions into a searchable string
    if "synthetic_questions" in df_meta.columns:
        df_meta = df_meta.with_columns(
            pl.col("synthetic_questions").list.join(" | ")
        )
        
    # 2. Ensure all fields are Milvus-compatible
    for col, dtype in zip(df_meta.columns, df_meta.dtypes):
        if dtype in [pl.List, pl.Object, pl.Struct] and col != "embedding":
             df_meta = df_meta.with_columns(pl.col(col).cast(pl.String))
             
    # 3. Add global metadata (Timestamp)
    df_meta = df_meta.with_columns(
        pl.lit(datetime.now(timezone.utc).isoformat()).alias("created_at")
    )
    
    # 4. Final assembly and routing (Multi-Modal)
    clean_metadatas = df_meta.to_dicts()
    text_data_to_insert = []
    visual_data_to_insert = []
    
    for j in range(len(clean_metadatas)):
        meta = clean_metadatas[j]
        is_visual = meta.get("is_visual", False)
        
        if is_visual:
            v_emb = meta.get("visual_embedding")
            if isinstance(v_emb, str):
                import json
                v_emb = json.loads(v_emb)
            
            record = {
                "embedding": v_emb,
                "content": final_child_chunks[j],
                **meta
            }
            record.pop("visual_embedding", None)
            visual_data_to_insert.append(record)
        else:
            record = {
                "embedding": batch_dense[j],
                "sparse_vector": batch_sparse[j],
                "content": final_child_chunks[j],
                **meta
            }
            text_data_to_insert.append(record)

    # Just-in-Time (JIT) Purge: delete old chunks and graph facts right before inserting the new ones.
    # This prevents a data blackout window if the extraction/embedding fails midway.
    filename = metadata.get("filename")
    source = metadata.get("source")
    if filename:
        try:
            init_milvus_collection()
            # Delete from text collection
            milvus_client.delete(collection_name=COLLECTION_NAME, filter=f"filename == '{filename}'")
            # Delete from visual collection
            milvus_client.delete(collection_name=VISUAL_COLLECTION_NAME, filter=f"filename == '{filename}'")
            logger.info(f"🗑️ JIT Purged old chunks for filename '{filename}' from Milvus.")
        except Exception as e:
            logger.warning(f"Failed to delete old chunks for filename '{filename}': {e}")
            
    if source:
        try:
            from app.core.graph_manager import graph_manager
            graph_manager.purge_source_relations(source)
        except Exception as e:
            logger.warning(f"Failed to delete old Neo4j facts for source '{source}': {e}")

    if text_data_to_insert:
        milvus_client.insert(
            collection_name=COLLECTION_NAME,
            data=text_data_to_insert
        )
        
    if visual_data_to_insert:
        logger.info(f"Storing {len(visual_data_to_insert)} visual fragments in dedicated collection.")
        milvus_client.insert(
            collection_name=VISUAL_COLLECTION_NAME,
            data=visual_data_to_insert
        )
    
    logger.info(f"Task SUCCESS: Processed {len(chunks)} chunks using {model_name}.")
    return {"status": "success", "count": len(chunks), "model": model_name}

@celery_app.task(
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True
)
def process_and_store_batch(self, chunks, metadata):
    """
    Background task to generate embeddings for a pre-extracted BATCH of chunks.
    """
    logger.info(f"Task Started: process_and_store_batch with {len(chunks)} chunks.")
    try:
        return _process_chunks(self, chunks, metadata)
    except Exception as exc:
        logger.exception(f"Task FAILED: {exc}")
        raise self.retry(exc=exc)

@celery_app.task(
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True
)
def process_file_ingestion(self, file_path, heavy_parsing, metadata):
    """
    Background task to fully extract and chunk a document, then generate embeddings
    and index them in Milvus and Neo4j. Runs entirely in the Celery worker.
    """
    logger.info(f"Task Started: process_file_ingestion for {file_path} (Heavy: {heavy_parsing})")
    self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'status': 'Extracting layout & structural blocks (Docling)...'})
    
    from app.services.loaders import extract_chunks_from_source
    
    try:
        # 1. Perform document loading and structural chunking (fully async/background)
        chunks = extract_chunks_from_source(file_path, heavy_parsing, hierarchical=True)
        logger.info(f"Extracted {len(chunks)} chunks from {file_path}.")
        
        # 2. Ingest the generated chunks
        return _process_chunks(self, chunks, metadata)
    except Exception as exc:
        logger.exception(f"File Ingestion Task FAILED: {exc}")
        raise self.retry(exc=exc)

@celery_app.task(
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

        # Generate Hybrid Embeddings (Dense + Sparse)
        dense_vector, sparse_vector = asyncio.run(ai_manager.get_hybrid_embeddings(text))

        if not metadata.get("skip_enrichment", False):
            # Generate Synthetic Questions (Phase 4)
            questions = asyncio.run(ai_manager.generate_synthetic_questions(text))
            metadata["synthetic_questions"] = questions
        else:
            metadata["synthetic_questions"] = []

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
        
        milvus_client.insert(
            collection_name=COLLECTION_NAME,
            data=[{
                "embedding": dense_vector.tolist(),
                "sparse_vector": sparse_vector,
                "content": text,
                **clean_meta
            }]
        )
        
        logger.info(f"Task COMPLETED: process_and_store_memory success using {model_name}.")
        return {"status": "success", "model": model_name}
    except Exception as exc:
        logger.exception(f"Task FAILED: process_and_store_memory error: {exc}")
        raise self.retry(exc=exc)
