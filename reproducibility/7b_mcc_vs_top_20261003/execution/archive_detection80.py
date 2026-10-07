import pathlib,json,hashlib,tarfile,time
B=pathlib.Path('/root/autodl-tmp/token-integrity'); R=B/'runs/fingerprint-7b-mcc-vs-top-v1'; C=B/'runs/calibration-7b-100-geo-v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert json.loads((R/'STATUS.json').read_text())['phase']=='COMPLETE'
assert (R/'RESULTS.json').exists() and (R/'DECISIONS.json').exists()
p=json.loads((R/'PLAN.json').read_text()); m=json.loads((C/'MODEL_MANIFEST.json').read_text())
for x in m['files']:assert sha(pathlib.Path(p['model']['name'])/x['path'])==x['sha256']
for name,key in [('CALIBRATION_PLAN.json','calibration_plan_sha256'),('CALIBRATION.json','calibration_sha256')]:assert sha(C/name)==p[key]
files=sorted(x for root in [R,B/'setup'] for x in root.rglob('*') if x.is_file())
manifest={str(x.relative_to(B)):dict(sha256=sha(x),size=x.stat().st_size) for x in files}
cal={str(x.relative_to(C)):dict(sha256=sha(x),size=x.stat().st_size) for x in C.rglob('*') if x.is_file()}
out=B/'fingerprint-detection80-final.tar.gz'
with tarfile.open(out,'w:gz') as t:
 for x in files:t.add(x,arcname=str(x.relative_to(B)),recursive=False)
a=dict(status='PASS',time=time.time(),model_verified=True,files=manifest,calibration_files=cal,archive_sha256=sha(out))
(B/'fingerprint-detection80-manifest.json').write_text(json.dumps(a,indent=2))
print(json.dumps(dict(archive=str(out),size=out.stat().st_size,sha256=a['archive_sha256'],files=len(files),calibration_files=len(cal))))
