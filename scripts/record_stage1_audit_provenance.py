from pathlib import Path
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
out = root / 'results/stage1_audit_20260905'
def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

expected = json.loads((root / 'environment/STAGE1_RUNTIME_MANIFEST.json').read_text())
models = {}
for key in ['target_model', 'semantic_encoder']:
    info = expected[key]
    path = Path(info['snapshot_path'])
    weights = path / 'model.safetensors'
    if not weights.exists():
        weights = path / 'pytorch_model.bin'
    models[key] = {'revision': info['revision'], 'weight_file': str(weights),
                   'weight_sha256': sha(weights), 'expected_weight_sha256': info['weight_sha256'],
                   'weight_match': sha(weights) == info['weight_sha256']}
lock = root / 'environment/requirements-stage1.lock'
lock_sha = sha(lock)
lock_lf = hashlib.sha256(lock.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
tracked = [root / 'scripts/run_h9_stage1_attack_stat_calibration.py',
           root / 'scripts/audit_h9_stage1_20260905.py', Path(__file__),
           root / 'configs/h9_stage1_qwen05b_attack_stat_calibration.yaml',
           root / 'configs/h9_stage1_qwen05b_gaussian_escalation.yaml']
payload = {
    'python': sys.version, 'platform': platform.platform(),
    'gpu': subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu',
                          '--format=csv,noheader'], capture_output=True, text=True).stdout,
    'packages': {d.metadata['Name']: d.version for d in md.distributions() if d.metadata.get('Name')},
    'models': models,
    'lock_sha256': lock_sha, 'lock_lf_sha256': lock_lf,
    'historical_expected_lock_sha256': expected['environment']['lock_file_sha256'],
    'execution_files': {str(p.relative_to(root)): sha(p) for p in tracked},
    'source_snapshot_sha256': sha(Path('/root/autodl-tmp/h9-stage1-a3e2f59.zip')),
    'source_commit': 'a3e2f592f5bedbdace77256627f0fa767971307c',
    'note': 'Execution and audit additions are uncommitted files on a source archive; hashes identify actual executed files.',
}
(out / 'EXECUTION_PROVENANCE.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'model_weight_matches': {k: v['weight_match'] for k, v in models.items()},
                  'lock_byte_match': lock_sha == expected['environment']['lock_file_sha256'],
                  'lock_lf_match': lock_lf == expected['environment']['lock_file_sha256']}))
