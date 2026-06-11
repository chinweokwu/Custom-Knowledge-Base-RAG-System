import os
import sys
import subprocess
import socket
import importlib.util
from pathlib import Path

# Professional ANSI Colors
class Colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"

def print_step(msg):
    print(f"{Colors.BLUE}{Colors.BOLD}🔍 {msg}{Colors.END}")

def print_ok(msg):
    print(f"{Colors.GREEN}  ✅ {msg}{Colors.END}")

def print_warn(msg):
    print(f"{Colors.YELLOW}  ⚠️  {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}  ❌ {msg}{Colors.END}")

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

class NeuralDoctor:
    def __init__(self, fix=False):
        self.fix = fix
        self.root_dir = Path(__file__).parent.parent.absolute()
        self.errors = 0

    def check_python_deps(self):
        print_step("Checking Python Dependencies...")
        requirements_file = self.root_dir / "requirements.txt"
        if not requirements_file.exists():
            print_error("requirements.txt not found!")
            return

        with open(requirements_file, "r") as f:
            deps = [line.split(">=")[0].split("==")[0].strip() for line in f if line.strip() and not line.startswith("#")]

        # Mapping for packages whose import name differs from install name
        import_map = {
            "opencv-python": "cv2",
            "beautifulsoup4": "bs4",
            "python-dotenv": "dotenv",
            "pymupdf": "fitz",
            "pillow": "PIL",
            "pypdf": "pypdf",
            "psycopg[binary]": "psycopg",
        }

        missing = []
        for dep in deps:
            # Get the expected import name
            pkg_name = dep.replace("-", "_").split("[")[0].lower()
            import_name = import_map.get(dep.split("[")[0].lower(), pkg_name)
            
            if importlib.util.find_spec(import_name) is None:
                missing.append(dep)

        if missing:
            print_warn(f"Missing {len(missing)} Python packages: {', '.join(missing)}")
            if self.fix:
                print_step("Attempting to fix Python dependencies...")
                pip_cmd = [sys.executable, "-m", "pip", "install"] + missing
                subprocess.run(pip_cmd)
            else:
                self.errors += 1
        else:
            print_ok("All Python dependencies are installed.")

    def check_node_deps(self):
        print_step("Checking Extension (Node.js) Dependencies...")
        plugin_dir = self.root_dir / "extensions" / "tech-brain-plugin"
        if not plugin_dir.exists():
            print_warn("Plugin directory not found. Skipping.")
            return

        node_modules = plugin_dir / "node_modules"
        if not node_modules.exists():
            print_warn("node_modules missing in plugin directory.")
            if self.fix:
                print_step("Installing Node.js dependencies...")
                subprocess.run(["npm", "install"], cwd=str(plugin_dir))
            else:
                self.errors += 1
        else:
            print_ok("Node.js dependencies are installed.")

    def check_infra(self):
        print_step("Checking Infrastructure (Redis, Milvus, Neo4j)...")
        
        # Redis
        if check_port(6379):
            print_ok("Redis is running on port 6379.")
        else:
            print_error("Redis is NOT detected on port 6379.")
            self.errors += 1

        # FastAPI
        if check_port(8000):
            print_warn("Port 8000 (FastAPI) is already in use. You might have a stale process.")
        
        # Check Milvus (Simple check for milvus_lite or running container)
        milvus_db = self.root_dir / "milvus_lite.db"
        if milvus_db.exists():
            print_ok(f"Milvus Lite DB found: {milvus_db.name}")
        else:
            print_warn("Milvus Lite DB not found. It will be initialized on first run.")

    def check_env(self):
        print_step("Checking Environment Variables...")
        env_file = self.root_dir / ".env"
        if not env_file.exists():
            print_error(".env file is missing!")
            self.errors += 1
            return

        with open(env_file, "r") as f:
            content = f.read()
        
        keys = ["GROQ_API_KEY", "ANTHROPIC_API_KEY", "MILVUS_URI"]
        for key in keys:
            if key not in content:
                print_warn(f"Missing {key} in .env file.")
            elif f"{key}=" in content and len(content.split(f"{key}=")[1].split("\n")[0].strip()) < 5:
                 print_warn(f"{key} is defined but appears empty.")

    def run(self):
        print(f"\n{Colors.BOLD}--- Neural Brain Doctor ---{Colors.END}\n")
        self.check_python_deps()
        self.check_node_deps()
        self.check_infra()
        self.check_env()
        
        print(f"\n{Colors.BOLD}--- Health Summary ---{Colors.END}")
        if self.errors == 0:
            print(f"{Colors.GREEN}All systems optimal. You are ready for launch!{Colors.END}")
            return True
        else:
            print(f"{Colors.RED}Found {self.errors} issues that require attention.{Colors.END}")
            if not self.fix:
                print(f"Run {Colors.BOLD}'python scripts/doctor.py --fix'{Colors.END} to attempt auto-repair.")
            return False

if __name__ == "__main__":
    fix_mode = "--fix" in sys.argv
    doctor = NeuralDoctor(fix=fix_mode)
    success = doctor.run()
    sys.exit(0 if success else 1)
