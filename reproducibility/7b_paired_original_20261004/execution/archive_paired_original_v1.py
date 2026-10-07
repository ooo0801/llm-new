import pathlib,json,hashlib,tarfile,time,fcntl
B=pathlib.Path('/root/autodl-tmp/token-integrity');R=B/'runs/fingerprint-7b-paired-original-v1';O=B/'runs/fingerprint-7b-mcc-vs-top-v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
with (R/'RUN.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((R/'STATUS.json').read_text())['phase']=='COMPLETE'
 files=sorted(x for x in R.rglob('*') if x.is_file())+[B/'setup/run_paired_original_v1.py',B/'setup/paired-original-v1.log',pathlib.Path(__file__).resolve()]
 dependencies={str(x.relative_to(B)):dict(sha256=sha(x),size=x.stat().st_size) for x in O.rglob('*') if x.is_file() and x.name!='RUN.lock'}
 manifest={str(x.relative_to(B)):dict(sha256=sha(x),size=x.stat().st_size) for x in files}
 out=B/'fingerprint-paired-original-v1.tar.gz'
 with tarfile.open(out,'w:gz') as t:
  for x in files:t.add(x,arcname=str(x.relative_to(B)),recursive=False)
 a=dict(time=time.time(),files=manifest,parent_files=dependencies,archive_sha256=sha(out))
 (B/'fingerprint-paired-original-v1-manifest.json').write_text(json.dumps(a,indent=2))
 print(json.dumps(dict(archive=str(out),size=out.stat().st_size,sha256=a['archive_sha256'])))
