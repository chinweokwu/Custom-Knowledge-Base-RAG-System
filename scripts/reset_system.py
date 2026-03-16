import redis
import psycopg
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Clear PostgreSQL
db_url = os.getenv("DATABASE_URL")
try:
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE ai_memory RESTART IDENTITY;")
            print("PostgreSQL 'ai_memory' table truncated.")
except Exception as e:
    print(f"PostgreSQL Error: {e}")

# 2. Clear Redis
redis_url = os.getenv("REDIS_URL")
try:
    r = redis.from_url(redis_url)
    r.flushall()
    print("Redis cache flushed (FLUSHALL).")
except Exception as e:
    print(f"Redis Error: {e}")
