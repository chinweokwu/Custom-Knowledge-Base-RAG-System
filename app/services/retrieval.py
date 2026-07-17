import os
import json
import asyncio
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone
from sentence_transformers import CrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from app.core.logger_config import get_logger
from app.core.ai_manager import ai_manager
from app.core.milvus_client import milvus_client, init_milvus_collection, COLLECTION_NAME, VISUAL_COLLECTION_NAME
from pymilvus import AnnSearchRequest, RRFRanker
from app.core.graph_manager import graph_manager
from app.core.config import settings
THRESHOLD = settings.THRESHOLD
MAX_CONFIDENCE_SCORE = 5.0

# Initialize Logger
logger = get_logger("retrieval_service")

_reranker_instance = None

def get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        logger.info("Lazy-loading Cross-Encoder model...")
        _reranker_instance = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _reranker_instance

# Advanced Intelligence Transformation Prompt
# This handles vague queries and identifies technical patterns
rewrite_prompt = ChatPromptTemplate.from_template(
    "You are an AI 'Prompt Engineer' for a high-precision Vector Search engine. "
    "Your goal is to expand the user's vague query into a rich, descriptive technical search. "
    "Rules: \n"
    "1. If the user uses vague words like 'it', 'them', or 'fix', infer the context based on common IT troubleshooting. \n"
    "2. Identify potential user IDs or Alphanumeric codes and maintain them. \n"
    "3. Provide exactly 1-3 diverse variations of the query, separated by the pipe character '|'. \n"
    "Example Query: 'jwx fix' -> 'Technical resolution and fix details for user jwx1369347 | Maintenance logs and alarm clearance for jwx | Discussion about server fixes involving jwx identifiers' \n"
    "Target Query: {query} \n"
    "Intelligence Expansion:"
)

async def get_hybrid_context(query_text: str, limit: int) -> List[Dict[str, Any]]:
    """
    Core retrieval function using RRF (Reciprocal Rank Fusion) + X-Algo Enhancements.
    Includes: Recency Bias, Source Weighting, and Document Diversity.
    """
    logger.info(f"Starting hybrid retrieval for query: '{query_text}'")
    
    # 1. Candidate Retrieval (Parallel Resonance)
    queries_to_search = [query_text]
    
    # --- CONCURRENT PHASE 1: Intelligence Gathering (Expansion + HyDE + CLIP) ---
    logger.info("Executing Concurrent Intelligence Gathering (Expansion + HyDE + CLIP)...")
    
    tasks = [
        ai_manager.get_clip_embedding(query_text)
    ]
    
    do_expand = len(query_text.split()) < 8
    if do_expand:
        tasks.append(ai_manager.call_llm(rewrite_prompt, {"query": query_text}))
    else:
        tasks.append(asyncio.sleep(0, result=None))
        
    tasks.append(ai_manager.generate_hyde_document(query_text))

    # Execute Concurrently
    clip_vector, expanded_raw, hyde_doc = await asyncio.gather(*tasks)

    # Process Expanded Variations
    if expanded_raw and isinstance(expanded_raw, str):
        variations = [v.strip() for v in expanded_raw.split('|') if v.strip()]
        if variations:
            queries_to_search.extend(variations[:2])
            logger.info(f"Queries expanded into {len(queries_to_search)} branches.")

    # Process HyDE
    if hyde_doc:
        queries_to_search.append(hyde_doc)
        logger.info("HyDE document added to search pool.")

    # 1b. Smart ID Extraction & Priority Guard
    now = datetime.now(timezone.utc)
    ids = re.findall(r'[a-zA-Z]{2,3}\d{6,8}', query_text)
    
    # 1b. Multi-Query Neural Fusion (Generate Hybrid Vectors)
    dense_vector, sparse_vector = await ai_manager.get_hybrid_embeddings(query_text)
    
    # --- CONCURRENT PHASE 2: Parallel Search Branches (Native 32k Hybrid) ---
    logger.info(f"Executing Native Hybrid Search (Dense 4k + Sparse 32k)...")
    init_milvus_collection()
    
    # Construct search requests
    # Request 1: Dense (Semantic Meaning)
    req_dense = AnnSearchRequest(
        data=[dense_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=50
    )
    
    # Request 2: Sparse (Exact Keyword/ID Matching)
    req_sparse = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
        limit=50
    )

    async def execute_native_hybrid():
        try:
            res = await asyncio.to_thread(
                milvus_client.hybrid_search,
                collection_name=COLLECTION_NAME,
                reqs=[req_dense, req_sparse],
                ranker=RRFRanker(k=settings.RRF_K),
                limit=50,
                output_fields=["content", "created_at", "synthetic_questions", "parent_content", "authority"]
            )
            branch_results = []
            if res and len(res) > 0:
                for hit in res[0]:
                    meta = hit.entity.to_dict()
                    created_at_time = now
                    if "created_at" in meta:
                        try:
                            created_at_time = datetime.fromisoformat(meta["created_at"].replace('Z', '+00:00'))
                        except: pass
                    branch_results.append((str(hit.id), meta.get("content", ""), meta, created_at_time))
            return branch_results
        except Exception as e:
            logger.error(f"Native Hybrid search failed: {e}")
            return []

    async def visual_search_branch(v):
        try:
            v_res = await asyncio.to_thread(
                milvus_client.search,
                collection_name=VISUAL_COLLECTION_NAME,
                data=[v],
                limit=5,
                output_fields=["content", "created_at", "media_url", "meaning_type"]
            )
            v_results = []
            if v_res and len(v_res[0]) > 0:
                for hit in v_res[0]:
                    meta = hit["entity"]
                    meta["is_visual"] = True
                    created_at_time = now
                    if "created_at" in meta:
                        try:
                            created_at_time = datetime.fromisoformat(meta["created_at"].replace('Z', '+00:00'))
                        except: pass
                    v_results.append((str(hit["id"]), meta.get("content", ""), meta, created_at_time))
            return v_results
        except Exception as e:
            logger.error(f"Visual search failed: {e}")
            return []

    # Execute Hybrid Text and Visual Search in parallel
    hybrid_task = execute_native_hybrid()
    visual_task = visual_search_branch(clip_vector)
    
    vector_results, visual_results = await asyncio.gather(hybrid_task, visual_task)
    
    logger.info(f"Retrieved {len(vector_results)} hybrid (Dense+Sparse) candidates and {len(visual_results)} visual candidates.")

    # 3. Final Scoring & X-Algo Weighting (Authority & Recency)
    # Since Milvus already did the RRF for query variations, we just apply our custom boosters
    logger.info("Applying X-Algo Boosters to native hybrid results...")
    scores = {} # id -> (content, metadata, created_at, score)
    
    def calculate_boosted_score(rank, content, metadata, created_at, query_text):
        # Base score from RRF rank (approximated as Milvus doesn't return the raw RRF score easily in all versions)
        base_rrf = 1.0 / (settings.RRF_K + rank + 1)
        
        # A. Source Authority Boost
        filename_lower = metadata.get("filename", "").lower()
        authority = metadata.get("authority", 1.0)
        if "handbook" in filename_lower or "manual" in filename_lower or "guide" in filename_lower:
            authority *= settings.AUTHORITY_MANUAL
        elif "chat" in filename_lower or "interaction" in filename_lower:
            authority *= settings.AUTHORITY_CHAT
        
        # B. ID-First Match Boost (X-Algo Shield)
        id_boost = 1.0
        for uid in ids:
            if uid.lower() in content.lower():
                id_boost = settings.BOOST_ID_MATCH
                break

        # C. Semantic Enrichment Boost (Phase 4)
        synth_boost = 1.0
        synth_qs_raw = metadata.get("synthetic_questions", "")
        synth_qs = [q.strip() for q in synth_qs_raw.split('|')] if isinstance(synth_qs_raw, str) else []
        for q in synth_qs:
            if query_text.lower() in q.lower() or q.lower() in query_text.lower():
                synth_boost = settings.BOOST_SYNTHETIC 
                break

        # D. Temporal Decay Boost (Newer is Better)
        temporal_boost = 1.0
        if created_at:
            days_old = (now - created_at).days
            years_old = days_old / 365.25
            temporal_boost = 1.0 / (1.0 + (years_old * settings.TEMPORAL_DECAY_RATE))

        return base_rrf * authority * id_boost * synth_boost * temporal_boost

    # Process Final Ranks
    for rank, (doc_id, content, meta, created_at) in enumerate(vector_results):
        scores[doc_id] = [content, meta, created_at, calculate_boosted_score(rank, content, meta, created_at, query_text)]
    
    # Merge Visual Results
    for rank, (doc_id, content, meta, created_at) in enumerate(visual_results):
        clip_score = (1.0 / (settings.RRF_K + rank + 1)) * 4.0 
        if doc_id in scores:
            scores[doc_id][3] += clip_score
        else:
            scores[doc_id] = [content, meta, created_at, clip_score]

    # 4. Final Re-ranking with Cross-Encoder (Increased to 100 for diversity)
    top_candidates = sorted(scores.values(), key=lambda x: x[3], reverse=True)[:100] 
    
    if not top_candidates:
        return []

    # 5. Round-Robin Diversity Selector (Layer 2)
    # Group results by source to force equal representation
    source_groups = {}
    for doc in top_candidates:
        source_id = doc[1].get("source_url") or doc[1].get("filename") or "unknown"
        if source_id not in source_groups:
            source_groups[source_id] = []
        source_groups[source_id].append(doc)

    final_candidates = []
    # Interleave results from different sources (Round Robin)
    max_per_source = 10
    for i in range(max_per_source):
        found_any = False
        for source_id in list(source_groups.keys()):
            if len(source_groups[source_id]) > i:
                final_candidates.append(source_groups[source_id][i])
                found_any = True
        if not found_any:
            break

    # Phase 21: Small-to-Big (Neighbor Expansion) - BATCHED PRE-FETCH
    # We fetch adjacent chunks to give the LLM full situational awareness
    # To avoid N+1 query problem, we collect all targets first
    expansion_targets = []
    for doc in final_candidates:
        src = doc[1].get("source") or doc[1].get("filename")
        idx = doc[1].get("chunk_idx")
        if src and idx is not None:
            expansion_targets.append((src, idx))

    neighbor_map = {} # (source, idx) -> content
    if expansion_targets:
        try:
            # Construct a filter for all sources and their relevant index ranges
            sources = list(set(t[0] for t in expansion_targets))
            all_indices = [t[1] for t in expansion_targets]
            min_idx = min(all_indices) - 1
            max_idx = max(all_indices) + 1
            
            source_filter = " || ".join([f"source == '{s}'" for s in sources])
            combined_filter = f"({source_filter}) and chunk_idx >= {min_idx} and chunk_idx <= {max_idx}"
            
            all_neighbors = milvus_client.query(
                collection_name=COLLECTION_NAME,
                filter=combined_filter,
                output_fields=["content", "chunk_idx", "source"]
            )
            for n in all_neighbors:
                neighbor_map[(n["source"], n["chunk_idx"])] = n["content"]
            logger.info(f"Batched Neighbor Expansion: Fetched {len(all_neighbors)} neighbors for {len(expansion_targets)} targets.")
        except Exception as e:
            logger.warning(f"Batched expansion query failed: {e}")

    # Final Re-score the diversified candidates with Cross-Encoder + Late Interaction
    logger.info(f"Applying Cross-Encoder re-ranking to {len(final_candidates)} diversified candidates...")
    pairs = [[query_text, doc[0]] for doc in final_candidates]
    rerank_scores = get_reranker().predict(pairs)
    
    final_docs = []
    seen_contents = set() # STRICT DEDUPLICATION
    
    # Extract clean query keywords for rescue matching
    rescue_terms = [w.lower() for w in re.findall(r'\b\w{3,}\b', query_text)
                    if w.lower() not in {'what', 'about', 'this', 'that', 'tell', 'give', 'explain', 'show', 'the', 'for', 'and', 'are', 'how', 'why'}]

    for i, doc in enumerate(final_candidates):
        content = doc[0]
        # Skip if we've already seen this content (or a very close variant)
        content_hash = "".join(content.lower().split())[:100]
        if content_hash in seen_contents:
            continue
        seen_contents.add(content_hash)

        # Base Cross-Encoder Score
        score = float(rerank_scores[i])
        
        # --- PHASE 22: LATE INTERACTION SIMULATION (MaxSim) ---
        # Adds token-level interaction scoring to catch exact technical matches
        token_score = ai_manager.calculate_token_interaction(query_text, content)
        score += (token_score * 3.0) # Weight the late interaction
        
        content_lower = content.lower()

        # X-Algo Protection Shields
        ids_found = [uid for uid in ids if uid.lower() in content_lower]
        has_id_match = len(ids_found) > 0
        if has_id_match: score += 5.0

        # Keyword Rescue (ULTRA BOOST)
        keyword_hits = sum(1 for term in rescue_terms if term in content_lower)
        rescue_boost = 0.0
        if keyword_hits >= 1:
            rescue_boost = settings.BOOST_KEYWORD + (keyword_hits * 1.0) 
            score += rescue_boost

        # --- Retrieval Reason Tag ---
        if has_id_match:
            retrieval_reason = f"ID match: {ids_found[0]}"
        elif rescue_boost > 4.0:
            retrieval_reason = f"Strong keyword match ({keyword_hits} terms)"
        elif rescue_boost > 0:
            retrieval_reason = f"Keyword match: {', '.join(t for t in rescue_terms if t in content_lower)[:40]}"
        else:
            retrieval_reason = "Semantic similarity"

        # Apply Batched Neighbor Expansion
        source = doc[1].get("source") or doc[1].get("filename")
        chunk_idx = doc[1].get("chunk_idx")
        expanded_content = content
        if source and chunk_idx is not None:
            prev_c = neighbor_map.get((source, chunk_idx - 1), "")
            next_c = neighbor_map.get((source, chunk_idx + 1), "")
            parts = []
            if prev_c: parts.append(prev_c)
            parts.append(content)
            if next_c: parts.append(next_c)
            expanded_content = "\n\n".join(parts)

        final_docs.append({
            "content": expanded_content, 
            "score": score,
            "metadata": doc[1],
            "created_at": doc[2],
            "retrieval_reason": retrieval_reason,
            "context_content": doc[1].get("parent_content", expanded_content)
        })

    # Sort and apply threshold — drop anything scoring below the noise floor
    final_docs = [d for d in sorted(final_docs, key=lambda x: x["score"], reverse=True)
                  if d["score"] >= THRESHOLD]

    # --- GRAPH EXPANSION (Phase 18 / Layer 3) ---
    graph_context = []
    # 1. Extract entities from query — allow short codes like OWS, 5G, RCA
    stop_words = {'what', 'about', 'this', 'that', 'from', 'with', 'there',
                  'their', 'where', 'when', 'have', 'been', 'does', 'will',
                  'more', 'some', 'into', 'than', 'then', 'them', 'they',
                  'also', 'just', 'like', 'tell', 'give', 'show', 'explain'}
    words = re.findall(r'\b[A-Za-z0-9][\w\-]*\b', query_text)
    # Keep original casing for technical codes (OWS, RCA) but search lowercase
    potential_entities = [w.strip("'s") for w in words
                         if len(w) >= 2 and w.lower() not in stop_words]

    # 2. Only expand from retrieved docs if they are actually relevant (score > 0)
    for doc in final_docs[:3]:
        if doc['score'] > 0:
            doc_words = re.findall(r'\b[A-Z][A-Z0-9]{1,}\b', doc['content'])  # Uppercase acronyms only
            potential_entities.extend(doc_words[:5])

    potential_entities = list(set(potential_entities))
    
    for entity in potential_entities:
        related = graph_manager.get_related_facts(entity)
        if related:
            logger.info(f"🕸️ [NEURAL LINK] Found {len(related)} graph facts for entity: '{entity}'")
        for relation in related:
            # CRITICAL: Only include graph facts that are actually relevant to the query
            # Check if the relation text contains at least one query entity word
            relation_lower = relation.lower()
            is_relevant = any(e.lower() in relation_lower for e in potential_entities if len(e) >= 2)
            if is_relevant:
                graph_context.append({
                    "content": f"[Graph Fact] {relation}",
                    "metadata": {"type": "graph_relation", "source": "Neural Knowledge Graph"}
                })

    if graph_context:
        unique_graph = []
        seen = set()
        for g in graph_context:
            if g['content'] not in seen:
                # Lowered to 0.01 to ensure technical content always wins
                g['score'] = 0.01
                unique_graph.append(g)
                seen.add(g['content'])
        logger.info(f"🕸️ Injecting {len(unique_graph)} relevant graph facts into results.")
        final_docs = final_docs + unique_graph[:5]

    if not final_docs:
        logger.warning(f"KNOWLEDGE GAP: Query '{query_text}' returned no results.")

    sorted_docs = sorted(final_docs, key=lambda x: x["score"], reverse=True)[:limit]

    # --- RESTORED & CALIBRATED: 5-4-3-2-1 Intelligence Rubric ---
    def determine_confidence(docs, query, sufficiency_eval=None):
        vector_scores = [d["score"] for d in docs if d.get("metadata", {}).get("type") != "graph_relation"]
        top_score = max(vector_scores) if vector_scores else -100.0
        
        # Factor 1: Agentic Sufficiency (Phase 17)
        sufficiency_penalty = 0
        is_sufficient = True
        if sufficiency_eval:
            is_sufficient = sufficiency_eval.get("sufficient", True)
            if not is_sufficient:
                sufficiency_penalty = 1
        
        # Factor 2: Score-based confidence
        # CALIBRATION: With +15 and +20 boosts, technical matches will now be > 10.0
        if top_score >= 8.0:
            level = 5
            label = "EXCEPTIONAL"
            reason = "Direct technical match found in authoritative Handbook/Manual."
        elif top_score >= 4.0:
            level = 4
            label = "HIGH"
            reason = "Strong semantic evidence retrieved from technical documentation."
        elif top_score >= 1.0:
            level = 3
            label = "MEDIUM"
            reason = "Relevant fragments found, but requires some interpretation."
        elif top_score >= -5.0:
            level = 2
            label = "LOW"
            reason = "Limited or loose matches found; evidence is weak."
        else:
            level = 1
            label = "NONE"
            reason = "No authoritative fragments matched this query."

        # Apply Agentic Adjustment
        if not is_sufficient and level > 2:
            level -= 1
            label = "MEDIUM" if level == 3 else "LOW"
            reason += " (Researcher flagged context as potentially incomplete)"

        # Factor 3: Source Agreement
        unique_sources = set(d["metadata"].get("filename", "unknown") for d in docs[:3])
        if len(unique_sources) > 1 and level >= 3:
            reason += f" Verified across {len(unique_sources)} different sources."

        return level, label, reason

    confidence_level, label, confidence_reason = determine_confidence(sorted_docs, query_text)

    logger.info(f"📊 Intelligence Rubric: {confidence_level}/5 ({label}) | Reason: {confidence_reason}")
    
    for doc in sorted_docs:
        doc["confidence"] = label
        doc["confidence_score"] = confidence_level
        doc["confidence_reason"] = confidence_reason

    return sorted_docs

async def memory_streamer(query_text: str, limit: int):
    """
    Generator that streams hybrid-ranked search results.
    """
    try:
        results = await get_hybrid_context(query_text, limit)
        for result in results:
            yield json.dumps(result) + "\n"
            await asyncio.sleep(0.01)
    except Exception as e:
        logger.exception(f"Error in memory_streamer: {e}")
        yield json.dumps({"error": str(e)}) + "\n"

async def perform_agentic_search(message: str, limit: int, conversation_history: list = None) -> Dict[str, Any]:
    """
    Phase 17: Agentic Reasoning (The Researcher)
    Implements a self-correcting retrieval loop that performs follow-up research
    if the initial context is insufficient.
    """
    logger.info(f"Agentic Search request: '{message}'")
    thought_process = []
    
    # --- ITERATION 1: Initial Search ---
    iteration = 1
    current_query = message
    thought_process.append(f"Iteration {iteration}: Performing initial hybrid search for '{current_query}'...")
    
    context_docs = await get_hybrid_context(current_query, limit)
    
    if not context_docs:
        logger.info("No context found initially. Attempting one expansion...")
        thought_process.append("No results found. Attempting query expansion...")
        context_docs = await get_hybrid_context(current_query, limit) 

    # --- AGENTIC LOOP (Max 3 Iterations for deep research) ---
    all_context_docs = list(context_docs)
    
    for _ in range(2): # Max 2 follow-up research loops
        if not all_context_docs:
            break
            
        # Prepare context for evaluation
        eval_context = "\n\n".join([f"Fragment {i} [Source: {d['metadata'].get('filename', 'unknown')}]: {d['content'][:500]}" for i, d in enumerate(all_context_docs[:10])])
        
        # Call the AI Evaluator (Phase 17)
        evaluation = await ai_manager.evaluate_context_sufficiency(message, eval_context)
        
        is_sufficient = evaluation.get("sufficient", True)
        if is_sufficient:
            thought_process.append("Current context deemed sufficient. Proceeding to final synthesis.")
            break
            
        # Insufficient - Expand Research
        iteration += 1
        missing = evaluation.get("missing_info", "contextual detail")
        suggested = evaluation.get("suggested_query", message)
        
        logger.info(f"Agentic Research Loop {iteration}. Missing: {missing}. Researching: {suggested}")
        thought_process.append(f"Iteration {iteration}: Missing info found ({missing}). Researching: '{suggested}'...")
        
        # Perform follow-up search
        extra_docs = await get_hybrid_context(suggested, limit=10)
        
        # Deduplicate and merge
        existing_contents = {d['content'] for d in all_context_docs}
        merged_count = 0
        for d in extra_docs:
            if d['content'] not in existing_contents:
                all_context_docs.append(d)
                existing_contents.add(d['content'])
                merged_count += 1
        
        thought_process.append(f"Research Loop Complete. Merged {merged_count} new technical fragments.")

    if not all_context_docs:
        return {
            "answer": "I'm sorry, I don't have enough verified information in my memory to answer that accurately.",
            "sources": [],
            "thought_process": thought_process
        }

    # --- FINAL SYNTHESIS ---
    context_parts = []
    for i, d in enumerate(all_context_docs):
        meta = d.get("metadata", {})
        source_type = meta.get("type", "source")
        llm_context = d.get("context_content", d["content"])
        
        block = f"--- SOURCE [{i+1}] (Type: {source_type}) ---\n{llm_context}\n"
        context_parts.append(block)

    context_text = "\n\n".join(context_parts)
    
    # Build the final Analyst Prompt
    confidence = all_context_docs[0].get("confidence", "MEDIUM") if all_context_docs else "LOW"
    try:
        answer = await ai_manager.call_llm(None, {
            "context": context_text,
            "question": message,
            "confidence": confidence,
            "conversation_history": conversation_history or []
        })
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        answer = "[Synthesis Offline] - Data retrieved but brain failed to process."

    return {
        "answer": answer,
        "sources": all_context_docs[:limit],
        "model": ai_manager.get_model_name(),
        "thought_process": thought_process,
        "confidence": confidence
    }

async def synthesize_dashboard_report(query: str, limit: int, conversation_history: list = None) -> Dict[str, Any]:
    """
    Unified search logic for the Admin Dashboard.
    Returns both the retrieved context and a generated answer.
    """
    try:
        context_docs = await get_hybrid_context(query, limit)
        
        # 2. Generate Answer (Master Analyst Synthesis)
        answer = "No relevant neural fragments passed the sensitivity threshold for this query."
        if context_docs:
            context_parts = [f"Source [{i+1}]: {d['content']}" for i, d in enumerate(context_docs)]
            context_text = "\n\n".join(context_parts)
            confidence = context_docs[0].get("confidence", "MEDIUM")
            try:
                answer = await ai_manager.call_llm(None, {
                    "context": context_text,
                    "question": query,
                    "confidence": confidence,
                    "conversation_history": conversation_history or []
                })
            except Exception as e:
                logger.error(f"Synthesis failed: {e}")
                answer = "[Neural Synthesis Offline - Using raw Retrieval-Only mode]"

        confidence = context_docs[0].get("confidence", "MEDIUM") if context_docs else "LOW"
        return {
            "query": query,
            "context": context_docs,
            "answer": answer,
            "confidence": confidence
        }
    except Exception as e:
        logger.exception(f"Dashboard synthesis failed: {e}")
        raise
