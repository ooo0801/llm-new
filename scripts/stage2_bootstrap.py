import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for script in ["stage2_assets.py", "stage2_preflight.py"]:
    print("BOOTSTRAP",script,flush=True)
    subprocess.run([sys.executable,"-u",str(ROOT/"scripts"/script)],cwd=ROOT,check=True)
