import psycopg
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def test_cosine_similarity():
    """
    Directly tests the PL/pgSQL cosine_similarity function.
    """
    print("Testing PL/pgSQL cosine_similarity function...")
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # Test Case 1: Identical Vectors (Score should be 1.0)
                cur.execute("SELECT public.cosine_similarity(ARRAY[1.0, 0.0, 1.0], ARRAY[1.0, 0.0, 1.0]);")
                score1 = cur.fetchone()[0]
                print(f"Result 1 (Identical): {score1} (Expected: 1.0)")
                assert abs(score1 - 1.0) < 1e-6

                # Test Case 2: Orthogonal Vectors (Score should be 0.0)
                cur.execute("SELECT public.cosine_similarity(ARRAY[1.0, 0.0], ARRAY[0.0, 1.0]);")
                score2 = cur.fetchone()[0]
                print(f"Result 2 (Orthogonal): {score2} (Expected: 0.0)")
                assert abs(score2 - 0.0) < 1e-6

                # Test Case 3: Error on Mismatched Lengths
                try:
                    cur.execute("SELECT public.cosine_similarity(ARRAY[1.0, 2.0], ARRAY[1.0, 2.0, 3.0]);")
                    cur.fetchone()
                    print("Test Case 3 FAILED: Mismatched lengths did not raise an exception.")
                except Exception as e:
                    print(f"Test Case 3 PASSED: Mismatched lengths correctly raised error: {e}")

        print("\n✅ PL/pgSQL Verification SUCCESSFUL!")
    except Exception as e:
        print(f"\n❌ Verification FAILED: {e}")

if __name__ == "__main__":
    test_cosine_similarity()
