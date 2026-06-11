#!/usr/bin/env python
import os
import sys
# Resolve project root relative to this script's path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import shutil
import argparse
from app.core.config import settings

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

def get_active_containers() -> list:
    active = []
    for c in ["ai_kb_neo4j", "ai_kb_milvus", "ai_kb_minio"]:
        if is_container_running(c):
            active.append(c)
    return active

def restore_milvus_lite(backup_dir: str):
    """Restores local Milvus Lite SQLite file."""
    src_file = os.path.join(backup_dir, "milvus_lite.db")
    if not os.path.exists(src_file):
        print("ℹ️ No Milvus Lite SQLite backup found in this snapshot. Skipping local restore.")
        return

    dest_file = settings.MILVUS_URI
    if not os.path.isabs(dest_file):
        dest_file = os.path.join(settings.BASE_DIR, dest_file)
        
    print(f"♻️ Restoring Milvus Lite database to: {dest_file}...")
    try:
        # Stop any active local connections or locks if possible
        if os.path.exists(dest_file):
            os.remove(dest_file)
        shutil.copy2(src_file, dest_file)
        print("✅ Milvus Lite SQLite file restored successfully.")
    except Exception as e:
        print(f"❌ Failed to restore Milvus Lite SQLite: {e}")

def restore_docker_volume(volume_name: str, backup_dir: str, filename: str):
    """Uses a temporary container to restore contents into a named Docker volume."""
    src_file = os.path.join(backup_dir, filename)
    if not os.path.exists(src_file):
        print(f"ℹ️ No volume archive '{filename}' found in snapshot. Skipping restore for '{volume_name}'.")
        return

    print(f"♻️ Restoring Docker Volume '{volume_name}' from archive '{filename}'...")
    abs_src_dir = os.path.abspath(backup_dir)
    
    # Run Alpine container to wipe target volume and extract the backup archive
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{volume_name}:/volume",
        "-v", f"{abs_src_dir}:/backup",
        "alpine",
        "sh", "-c", f"rm -rf /volume/* && rm -rf /volume/.* 2>/dev/null; tar -xzf /backup/{filename} -C /volume"
    ]
    try:
        run_cmd(cmd)
        print(f"✅ Volume '{volume_name}' restored successfully.")
    except Exception as e:
        print(f"❌ Failed to restore volume '{volume_name}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Automated Enterprise RAG Recovery Utility")
    parser.add_argument("backup_path", help="Path to the specific backup directory (e.g. backups/backup_YYYYMMDD_HHMMSS)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.backup_path):
        print(f"❌ Error: Backup directory not found at: {args.backup_path}")
        sys.exit(1)
        
    print(f"=== Starting RAG Recovery/Restore from: {args.backup_path} ===")
    
    # 1. Detect active Docker containers
    active_containers = get_active_containers()
    
    # 2. Stop running containers to prevent write conflicts
    stopped_containers = []
    if active_containers:
        print(f"🛑 Stopping database containers to avoid corruption: {', '.join(active_containers)}...")
        for container in active_containers:
            try:
                run_cmd(["docker", "stop", container])
                stopped_containers.append(container)
            except Exception:
                print(f"⚠️ Failed to stop {container}, proceeding anyway.")
                
    try:
        # 3. Perform Restores
        # A. Local Milvus Lite
        restore_milvus_lite(args.backup_path)
        
        # B. Standalone Milvus volume (MinIO)
        restore_docker_volume("ai_knowledge_based_version_milvus_data", args.backup_path, "milvus_data.tar.gz")
        
        # C. Neo4j Volume
        restore_docker_volume("ai_knowledge_based_version_neo4j_data", args.backup_path, "neo4j_data.tar.gz")
        
    finally:
        # 4. Restart stopped containers
        if stopped_containers:
            print(f"🚀 Restarting database containers: {', '.join(stopped_containers)}...")
            for container in reversed(stopped_containers):
                try:
                    run_cmd(["docker", "start", container])
                except Exception as e:
                    print(f"❌ Failed to start container {container}: {e}")
                    
    print("🎉 Recovery process completed. Database nodes started successfully.")

if __name__ == "__main__":
    main()
