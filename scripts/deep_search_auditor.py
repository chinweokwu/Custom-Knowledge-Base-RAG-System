import asyncio
import os
import json
import statistics
from datetime import datetime
from app.core.database import pool
from app.api.main import get_hybrid_context
from app.core.ai_manager import ai_manager
from dotenv import load_dotenv

load_dotenv()

class RAGAuditor:
    def __init__(self):
        self.test_cases = [
            {
                "id": "AUDIT_001",
                "content": "CRITICAL PROTOCOL: In case of a hub power failure, the backup generator must be engaged within 45 seconds to prevent data loss at Site Alpha.",
                "query": "What is the protocol for hub power failure at Site Alpha?",
                "metadata": {"type": "official_doc", "category": "energy_protocol"}
            },
            {
                "id": "AUDIT_002",
                "content": "USER TROUBLESHOOTING: User jwx1369347 reported that their VPN connection drops every 2 hours specifically when using the fiber optic uplink.",
                "query": "Why does jwx1369347 have VPN drops?",
                "metadata": {"type": "chat_log", "category": "network_issue"}
            },
            {
                "id": "AUDIT_003",
                "content": "X-ALGO ENGINE SPECS: The 5x Neural Fusion ensemble utilizes a 1920-dimension vector space to achieve 99.8% semantic precision on technical logs.",
                "query": "How many dimensions does the 5x Neural Fusion use?",
                "metadata": {"type": "internal_spec", "category": "ai_engine"}
            },
            {
                "id": "AUDIT_004",
                "content": "MAINTENANCE LOG: The cooling fans on Rack 7 were replaced by technician mwx1328172 on 2026-03-01 to resolve recurring thermal alarms.",
                "query": "Who fixed the cooling fans on Rack 7?",
                "metadata": {"type": "maintenance_record", "category": "hardware"}
            },
            {
                "id": "AUDIT_005",
                "content": "SITE GEOGRAPHY: Ibadan Hub serves as the primary data relay for the Southwest region, connecting over 140 remote stations.",
                "query": "Which hub serves the Southwest region data relay?",
                "metadata": {"type": "site_info", "category": "geography"}
            }
        ]
        self.results = []

    async def setup(self):
        print("\n--- Setting up Audit Ground Truth ---")
        for case in self.test_cases:
            # Generate Synthetic Questions via Llama (Real Enrichment)
            print(f"Hydrating Intelligence for {case['id']}...")
            questions = await ai_manager.generate_synthetic_questions(case['content'])
            
            # Prepare Metadata
            meta = case['metadata'].copy()
            meta['audit_id'] = case['id']
            meta['synthetic_questions'] = questions
            meta['embedding_model'] = ai_manager.get_model_name()
            
            # Vectorize
            vector = await ai_manager.get_embeddings(case['content'])
            
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO ai_memory (content, embedding, metadata) VALUES (%s, %s, %s) ON CONFLICT (content) DO NOTHING",
                        (case['content'], str(vector).replace(' ', ''), json.dumps(meta))
                    )
                    conn.commit()
        print("DONE: Ground Truth hydrated.")

    async def run_audit(self):
        print("\n--- Running Neural Audit (MRR / Precision) ---")
        reciprocal_ranks = []
        precision_at_1 = []
        precision_at_5 = []

        for case in self.test_cases:
            print(f"\nQUERY: '{case['query']}'")
            start_time = datetime.now()
            results = await get_hybrid_context(case['query'], limit=10)
            end_time = datetime.now()
            
            latency = (end_time - start_time).total_seconds()
            
            # Find the position of the Ground Truth
            found_rank = -1
            for i, r in enumerate(results):
                if r['metadata'].get('audit_id') == case['id']:
                    found_rank = i + 1
                    break
            
            if found_rank > 0:
                rr = 1.0 / found_rank
                reciprocal_ranks.append(rr)
                precision_at_1.append(1 if found_rank == 1 else 0)
                precision_at_5.append(1 if found_rank <= 5 else 0)
                print(f"MATCH FOUND: Rank {found_rank} | Score: {results[found_rank-1]['score']:.4f} | Latency: {latency:.2f}s")
            else:
                reciprocal_ranks.append(0)
                precision_at_1.append(0)
                precision_at_5.append(0)
                print(f"MISSED: Ground Truth not in top 10.")

        # Aggregates
        avg_mrr = statistics.mean(reciprocal_ranks)
        avg_p1 = statistics.mean(precision_at_1)
        avg_p5 = statistics.mean(precision_at_5)

        print("\n" + "="*40)
        print("FINAL AUDIT REPORT (Local RAG Metrics)")
        print("="*40)
        print(f"Mean Reciprocal Rank (MRR):  {avg_mrr:.4f} (Target: >0.80)")
        print(f"Precision @ 1:              {avg_p1*100:.1f}%   (Target: >70%)")
        print(f"Precision @ 5:              {avg_p5*100:.1f}%   (Target: >95%)")
        print("="*40)
        
        if avg_mrr > 0.9:
            print("STATUS: ELITE PERFORMANCE. Neural Fusion & Llama-3 synthesis are locked.")
        elif avg_mrr > 0.7:
            print("STATUS: STABLE. Retrieval is accurate for most queries.")
        else:
            print("STATUS: DEGRADED. Check hybrid weights or embedding model health.")

    async def cleanup(self):
        print("\n--- Cleaning up Audit Data ---")
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM ai_memory WHERE metadata->>'audit_id' IS NOT NULL")
                conn.commit()
        print("DONE: DB Restored.")

async def main():
    auditor = RAGAuditor()
    try:
        await auditor.setup()
        await auditor.run_audit()
    finally:
        await auditor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
