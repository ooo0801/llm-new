"""Wait for verified download, then run exactly one calibration worker."""
from pathlib import Path
import fcntl,json,subprocess,time
base=Path('/root/autodl-tmp/token-integrity')
root=base/'runs/calibration-7b-80-v1'
with (root/'LAUNCH.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time()
    while not (root/'SMALL_MODEL_REMOVAL.json').exists():
        if time.time()-start>10800:raise RuntimeError('Download did not finish within three hours; no calibration started')
        time.sleep(10)
    assert (base/'models/qwen7b/DOWNLOAD_MANIFEST.json').exists()
    subprocess.run([str(base/'env/bin/python'),'-u',str(base/'repo/scripts/run_calibration_7b_80.py')],check=True,cwd=base/'repo')
