import hashlib,json,tarfile
from pathlib import Path
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--data-dir', type=Path, required=True, help='Local experiment backup directory')
_data_dir = _parser.parse_args().data_dir.resolve()
h=_data_dir
a=h/'generation-ablation-v1-results.tar.gz'
m=json.loads((h/'ARCHIVE.json').read_text(encoding='utf-8'))
assert a.stat().st_size==m['bytes']
assert hashlib.sha256(a.read_bytes()).hexdigest()==m['sha256']
with tarfile.open(a) as t:
    for member in t.getmembers():
        path=(h/member.name).resolve()
        assert path.is_relative_to(h.resolve())
        assert not member.issym() and not member.islnk()
    t.extractall(h,filter='data')
print('Archive size and SHA256 verified; extracted safely.')
