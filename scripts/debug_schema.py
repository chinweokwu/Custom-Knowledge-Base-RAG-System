import asyncio
from app.core.database import pool
from app.core.logger_config import get_logger

logger = get_logger("debug_schema")

async def check_schema():
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                print("\n--- 📝 Table Definition: ai_memory ---")
                
                # Check column types
                cur.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'ai_memory';
                """)
                cols = cur.fetchall()
                for col in cols:
                    print(f"Column: {col[0]:<15} | Type: {col[1]:<15} | Null: {col[2]}")
                
                print("\n--- 🔑 Constraints & Indexes ---")
                # Check constraints
                cur.execute("""
                    SELECT conname, pg_get_constraintdef(c.oid)
                    FROM pg_constraint c
                    JOIN pg_namespace n ON n.oid = c.connamespace
                    WHERE conrelid = 'ai_memory'::regclass;
                """)
                cons = cur.fetchall()
                for con in cons:
                    print(f"Constraint: {con[0]:<20} | Def: {con[1]}")

                # Check indexes
                cur.execute("""
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = 'ai_memory';
                """)
                idxs = cur.fetchall()
                for idx in idxs:
                    print(f"Index: {idx[0]:<20} | Def: {idx[1]}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_schema())
