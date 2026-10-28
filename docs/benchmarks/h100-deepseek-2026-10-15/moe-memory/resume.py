from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(ROOT/'finalize.py'), '--resume-prepare'], check=True)
subprocess.run([sys.executable, str(ROOT/'seal.py')], check=True)
