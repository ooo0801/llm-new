"""Pinned Qwen 7B download, hashed inventory, then authorized 1.5B cleanup."""
import os
os.environ['HF_HUB_DISABLE_XET']='1'
from pathlib import Path
import hashlib,json,shutil,time
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests
from huggingface_hub import HfApi,snapshot_download

base=Path('/root/autodl-tmp/token-integrity')
root=base/'runs/calibration-7b-80-v1';root.mkdir(parents=True,exist_ok=True)
target=base/'models/qwen7b'
repo='Qwen/Qwen2.5-7B-Instruct'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()
pin=root/'MODEL_PIN.json'
if pin.exists():meta=json.loads(pin.read_text())
else:
    info=HfApi(endpoint='https://hf-mirror.com').model_info(repo,files_metadata=True)
    meta=dict(repo_id=repo,revision=info.sha,metadata_endpoint='https://hf-mirror.com',files=[dict(path=s.rfilename,sha256=s.lfs.sha256 if s.lfs else None) for s in info.siblings])
    pin.write_text(json.dumps(meta,indent=2))
print('PINNED',meta['revision'],flush=True)
snapshot_download(repo_id=repo,revision=meta['revision'],local_dir=str(target),endpoint='https://hf-mirror.com',
    allow_patterns=['*.json','merges.txt','vocab.json','LICENSE','README.md'],max_workers=4)
for item in meta['files']:
    if not item['path'].endswith('.safetensors'):continue
    path=target/item['path']
    if path.exists() and sha(path)==item['sha256']:continue
    url='https://modelscope.cn/models/'+repo+'/resolve/master/'+item['path']
    response=requests.get(url,headers={'Range':'bytes=0-0'},timeout=60)
    assert response.status_code==206
    size=int(response.headers['Content-Range'].rsplit('/',1)[1]);chunk=32*1024*1024
    part=path.with_suffix('.part');checkpoint=path.with_suffix('.parts.json')
    done=set(json.loads(checkpoint.read_text())) if checkpoint.exists() else set()
    if not part.exists():done=set()
    fd=os.open(part,os.O_CREAT|os.O_RDWR,0o600);os.ftruncate(fd,size)
    def fetch(index):
        start=index*chunk;end=min(size,start+chunk)-1
        for attempt in range(5):
            try:
                r=requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(30,120))
                assert r.status_code==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{size}'
                data=r.content;assert len(data)==end-start+1
                written=0
                while written<len(data):written+=os.pwrite(fd,data[written:],start+written)
                return index
            except Exception:
                if attempt==4:raise
                time.sleep(2**attempt)
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures=[pool.submit(fetch,i) for i in range((size+chunk-1)//chunk) if i not in done]
            for future in as_completed(futures):
                done.add(future.result());checkpoint.write_text(json.dumps(sorted(done)))
                if len(done)%16==0:print('RANGE_DOWNLOAD',item['path'],len(done),'/',(size+chunk-1)//chunk,flush=True)
    finally:os.close(fd)
    assert sha(part)==item['sha256'],('download hash mismatch',str(part))
    part.replace(path);checkpoint.unlink(missing_ok=True)
    print('SHARD_VERIFIED',item['path'],flush=True)
files=[]
for item in meta['files']:
    path=target/item['path']
    if not path.is_file():continue
    digest=sha(path)
    if item['sha256']:assert digest==item['sha256'],str(path)
    files.append(dict(path=item['path'],sha256=digest,size=path.stat().st_size,remote_sha256=item['sha256']))
assert any(r['path'].endswith('.safetensors') for r in files)
config=json.loads((target/'config.json').read_text())
manifest=dict(repo_id=repo,revision=meta['revision'],files=files,config=config)
(target/'DOWNLOAD_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
(root/'MODEL_DOWNLOAD_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
# User explicitly requested removing only the previously downloaded small model.
old=(base/'models/qwen15b').resolve()
assert old==Path('/root/autodl-tmp/token-integrity/models/qwen15b') and old.parent==(base/'models').resolve()
if old.exists():
    old_manifest=json.loads((old/'DOWNLOAD_MANIFEST.json').read_text())
    assert old_manifest['repo_id']=='Qwen/Qwen2.5-1.5B-Instruct'
    assert sha(old/'model.safetensors')=='dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'
    dest=root/'retired_1p5b_metadata';dest.mkdir(exist_ok=True)
    for p in old.iterdir():
        if p.is_file() and p.suffix in ['.json','.txt']:shutil.copy2(p,dest/p.name)
    shutil.rmtree(old)
    (root/'SMALL_MODEL_REMOVAL.json').write_text(json.dumps(dict(path=str(old),removed=True,time=time.time(),retained='all runs and adapters; model metadata copied'),indent=2))
print('DOWNLOAD_VERIFIED_OLD_MODEL_REMOVED',flush=True)
