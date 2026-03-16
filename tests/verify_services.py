import redis
import os
from dotenv import load_dotenv
from app.services.tasks import app as celery_app
import socket

load_dotenv()

def check_redis():
    print("🔍 Checking Redis...")
    try:
        r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        r.ping()
        print("✅ Redis is UP and responding.")
        return True
    except Exception as e:
        print(f"❌ Redis is DOWN: {e}")
        return False

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def check_celery():
    print("🔍 Checking Celery Worker status...")
    try:
        # inspect() requires an active worker
        inspector = celery_app.control.inspect()
        active = inspector.active()
        if active:
            print(f"✅ Celery Worker is ACTIVE: {list(active.keys())}")
            return True
        else:
            print("⚠️  Celery Worker is NOT detected. (Is it running? Run: celery -A tasks worker --loglevel=info)")
            return False
    except Exception as e:
        print(f"❌ Error checking Celery: {e}")
        return False

if __name__ == "__main__":
    print("--- 🛠️ Service Health Check ---")
    redis_ok = check_redis()
    
    # Check FastAPI default port
    api_port = 8000
    if check_port(api_port):
        print(f"✅ FastAPI Server is LISTENING on port {api_port}.")
    else:
        print(f"⚠️  FastAPI Server is NOT detected on port {api_port}.")

    celery_ok = check_celery()
    
    print("\n--- 📝 Summary ---")
    if redis_ok and celery_ok:
        print("🚀 ALL SYSTEMS GO! Your background services are ready.")
    else:
        print("🛑 ACTION REQUIRED: Please run 'start_app.bat' to launch missing services.")
