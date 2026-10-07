import _bootstrap
import json
from pathlib import Path
import torch
from calibration_80_pool import build,CATS
from llm_integrity.macro_proxy import differentiable_macro_proxy
base=Path('/root/autodl-tmp/token-integrity/runs')
old=json.loads((base/'calibration-scale-v1/CALIBRATION_PLAN.json').read_text())
s=build(old['selected'])
t=[json.loads(x) for x in (base/'fresh-v3/train.jsonl').read_text().splitlines()]
assert not {r['prompt'] for r in s}&{r['prompt'] for r in t}
assert s[:20]==old['selected']
for n in [20,40,80]:assert all(sum(r['category']==c for r in s[:n])==n//4 for c in CATS)
torch.manual_seed(42)
e=torch.randn(1,3,4,requires_grad=True)
w=torch.randn(12,7);vs=[torch.randn(12,7) for _ in range(5)]
a=e.flatten()@w
scores=[differentiable_macro_proxy(a,e.flatten()@v,proxy='js',top_k=7) for v in vs]
gs=[torch.autograd.grad(x,e,retain_graph=True)[0] for x in scores]
direct=torch.autograd.grad(torch.stack(scores).mean(),e)[0]
torch.testing.assert_close(direct,torch.stack(gs).mean(0))
assert not torch.isclose(direct.norm(),torch.stack([g.norm() for g in gs]).mean())
print('PASS: nested 20/40/80 balance, training isolation, mean-JS gradient equals mean of gradient vectors; differs from mean norms')
