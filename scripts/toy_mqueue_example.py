import redis
import time
import threading
import json

# =====================================================================
# A Toy Message Queue (MQueue) Implementation using Redis
# This shows the fundamental concepts behind RabbitMQ, Celery, and Kafka
# =====================================================================

# 1. Connect to our existing local Redis server (from start_app.bat)
r = redis.Redis(host='localhost', port=6380, decode_responses=True)
QUEUE_NAME = "task_queue"

def producer():
    """
    The PRODUCER: Generates tasks and puts them into the queue.
    In your app, this is the FastAPI server accepting a file upload.
    """
    print("[Producer] Started...")
    for i in range(1, 6):
        task = {
            "task_id": i,
            "filename": f"document_{i}.pdf",
            "action": "extract_text"
        }
        # LPUSH pushes the task to the Left side of the Redis list
        r.lpush(QUEUE_NAME, json.dumps(task))
        print(f"[Producer] 📥 Sent to Queue: {task['filename']}")
        time.sleep(0.5) # Simulate time taken between uploads
    print("[Producer] Finished uploading 5 files.")

def consumer():
    """
    The CONSUMER (Worker): Reads tasks from the queue and processes them.
    In your app, this is Celery.
    """
    print("[Consumer] Worker started. Waiting for jobs...")
    while True:
        # BRPOP continuously checks the Right side of the Redis list. 
        # If the queue is empty, it 'Blocks' (waits) until a task arrives.
        result = r.brpop(QUEUE_NAME, timeout=3)
        
        if result:
            queue_name, task_data = result
            task = json.loads(task_data)
            
            print(f"[Consumer] ⚙️ Processing: {task['filename']}...")
            time.sleep(1.5) # Simulate heavy AI processing
            print(f"[Consumer] ✅ Finished: {task['filename']}")
        else:
            print("[Consumer] Queue is empty. Going to sleep...")
            break

if __name__ == "__main__":
    # Clear any old items in the queue
    r.delete(QUEUE_NAME)
    
    # We run the Consumer in a separate background thread (like Celery does in another window)
    worker_thread = threading.Thread(target=consumer)
    worker_thread.start()
    
    # The main thread acts as the Producer
    time.sleep(1) # Give worker a second to start
    producer()
    
    # Wait for the worker to finish
    worker_thread.join()
