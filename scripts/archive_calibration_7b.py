"""Archive complete 7B calibration, including adapter and gradient evidence."""
from pathlib import Path
import fcntl,hashlib,json,shutil,tarfile
base=Path('/root/autodl-tmp/token-integrity');root=base/'runs/calibration-7b-80-v1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()
assert json.loads((root/'STATUS.json').read_text())['phase']=='COMPLETE'
with (root/'RUN.lock').open('r') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ['calibration-7b-80-v1','download-calibration-7b']:
        shutil.copy2(base/'setup'/f'{name}.log',root/f'{name}.log')
    files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    archive=base/'runs/calibration-7b-80-v1-results.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:tar.add(root,arcname=root.name)
    manifest=dict(archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,files=files)
    (base/'runs/calibration-7b-80-v1-ARCHIVE.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'}),flush=True)
