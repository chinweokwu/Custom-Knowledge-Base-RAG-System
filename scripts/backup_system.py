#!/usr/bin/env python
import os
import sys
# Resolve project root relative to this script's path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import sqlite3
import shutil
from datetime import datetime
import argparse
from app.core.config import settings

BACKUP_DIR_DEFAULT = os.path.join(settings.BASE_DIR, "backups")

def run_cmd(cmd: list) -> subprocess.CompletedProcess:
    """Helper to run shell commands safely."""
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR executing command {' '.join(cmd)}: {e.stderr}")
        raise

def is_container_running(name: str) -> bool:
    """Check if a docker container is active."""
    try:
        res = run_cmd(["docker", "ps", "--filter", f"name={name}", "--format", "{{.Names}}"])
        return name in res.stdout.strip().split('\n')
    except Exception:
        return False

def backup_milvus_lite(dest_path: str):
    """Safely performs a transaction-aware online backup of local Milvus Lite (SQLite)."""
    db_file = settings.MILVUS_URI
    if not os.path.isabs(db_file):
        db_file = os.path.join(settings.BASE_DIR, db_file)
        
    if not os.path.exists(db_file):
        print(f"⚠️ Milvus Lite DB file not found at {db_file}. Skipping local SQLite backup.")
        return

    print(f"📦 Backing up Milvus Lite database: {db_file}...")
    backup_db_path = os.path.join(dest_path, "milvus_lite.db")
    
    try:
        # SQLite Online Backup API prevents locks/corruption
        src_conn = sqlite3.connect(db_file)
        dest_conn = sqlite3.connect(backup_db_path)
        with dest_conn:
            src_conn.backup(dest_conn)
        src_conn.close()
        dest_conn.close()
        print(f"✅ Milvus Lite SQLite backup complete: {backup_db_path}")
    except Exception as e:
        print(f"❌ Failed to backup Milvus Lite: {e}")
        # Fallback to copy if backup api fails
        shutil.copy2(db_file, backup_db_path)
        print(f"⚠️ Fallback copy complete.")

def backup_docker_volume(volume_name: str, dest_path: str, filename: str):
    """Uses a temporary container to archive a named Docker volume."""
    print(f"📦 Creating archive for Docker Volume '{volume_name}'...")
    abs_dest = os.path.abspath(dest_path)
    
    # Run Alpine container to tar the volume contents
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{volume_name}:/volume",
        "-v", f"{abs_dest}:/backup",
        "alpine",
        "tar", "-czf", f"/backup/{filename}", "-C", "/volume", "."
    ]
    try:
        run_cmd(cmd)
        print(f"✅ Volume '{volume_name}' backed up successfully to: {filename}")
    except Exception as e:
        print(f"❌ Failed to backup volume '{volume_name}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Automated Enterprise RAG Backup Utility")
    parser.add_argument("--dir", default=BACKUP_DIR_DEFAULT, help="Directory to store backup files")
    parser.add_argument("--keep", type=int, default=7, help="Number of recent backups to keep")
    parser.add_argument("--stop-containers", action="store_true", help="Stop docker containers temporarily to ensure absolute consistency")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_backup_dir = os.path.join(args.dir, f"backup_{timestamp}")
    os.makedirs(current_backup_dir, exist_ok=True)
    
    print(f"=== Starting RAG Automated Backup [Timestamp: {timestamp}] ===")
    
    # 1. Detect configuration type
    is_milvus_local = settings.MILVUS_URI.endswith(".db") or "localhost" not in settings.MILVUS_URI
    
    # 2. Check for running Docker containers
    neo4j_active = is_container_running("ai_kb_neo4j")
    milvus_active = is_container_running("ai_kb_milvus")
    minio_active = is_container_running("ai_kb_minio")
    
    active_containers = []
    if neo4j_active: active_containers.append("ai_kb_neo4j")
    if milvus_active: active_containers.append("ai_kb_milvus")
    if minio_active: active_containers.append("ai_kb_minio")
    
    # 3. Optional: Stop containers for 100% safe cold filesystem copy
    stopped_containers = []
    if args.stop_containers and active_containers:
        print(f"🛑 Stopping active database containers for cold backup: {', '.join(active_containers)}...")
        for container in active_containers:
            try:
                run_cmd(["docker", "stop", container])
                stopped_containers.append(container)
            except Exception:
                print(f"⚠️ Failed to stop container {container}, proceeding online.")
                
    try:
        # 4. Perform Backups
        # A. Milvus Lite Backup
        if is_milvus_local:
            backup_milvus_lite(current_backup_dir)
            
        # B. Standalone Milvus (MinIO Volume) Backup
        if milvus_active or not is_milvus_local:
            # We backup the milvus_data volume (contains files/metadata)
            backup_docker_volume("ai_knowledge_based_version_milvus_data", current_backup_dir, "milvus_data.tar.gz")
            
        # C. Neo4j Graph DB Volume Backup
        if neo4j_active or os.path.exists("/var/lib/docker/volumes/ai_knowledge_based_version_neo4j_data"):
            backup_docker_volume("ai_knowledge_based_version_neo4j_data", current_backup_dir, "neo4j_data.tar.gz")
            
    finally:
        # 5. Restart any stopped containers
        if stopped_containers:
            print(f"🚀 Restarting stopped containers: {', '.join(stopped_containers)}...")
            for container in reversed(stopped_containers):
                try:
                    run_cmd(["docker", "start", container])
                except Exception as e:
                    print(f"❌ Failed to restart container {container}: {e}")
                    
    # 6. Rotate Old Backups
    print("🧹 Cleaning up old backups...")
    all_backups = sorted(
        [os.path.join(args.dir, d) for d in os.listdir(args.dir) if d.startswith("backup_")],
        key=os.path.getmtime
    )
    if len(all_backups) > args.keep:
        to_delete = all_backups[:-args.keep]
        for path in to_delete:
            print(f"🗑️ Deleting old backup: {path}")
            shutil.rmtree(path)
            
    print(f"🎉 Backup process completed. Files saved in: {current_backup_dir}")

if __name__ == "__main__":
    main()
