"""Read-only audit of saved Stage2 search artifacts; never run inference or change selection."""
import argparse
import json
from collections import Counter
from pathlib import Path
from build_discrete_candidate_portfolio import candidate_is_eligible


def summarize(path):
    row=json.loads(path.read_text(encoding='utf-8'))
    s=row['effective_parameters']; history=[h for h in row['history'] if 'round' in h]
    candidates=[c for h in history for c in h.get('reranked_candidates',[])]
    eligible=[c for c in candidates if candidate_is_eligible(c,s['minimum_nondegraded_families'])]
    return {'source_id':row['source']['id'],'method':row['method'],'accepted':row['accepted'],
            'failure':row['failure'],'requested_rounds':row['rounds_requested'],'saved_rounds':len(history),
            'reported_rounds_completed':row['rounds_completed'],'committed_rounds':row['committed_rounds'],
            'parameters':s,'seconds':row['seconds'],'reranked_candidates':len(candidates),
            'eligible_unique_prompts':len({c['decoded_prompt'] for c in eligible}),
            'task_checked':sum('task_validation' in c for c in candidates),
            'task_passed':sum(c.get('task_validation',{}).get('task_passed') is True for c in candidates),
            'candidate_family_counts':dict(Counter(c.get('training_nondegraded_families') for c in candidates)),
            'initial_prompt':row['initial_prompt'],'optimized_prompt':row['optimized_prompt'],
            'interpretation':'Internal proxy results only; no behavior sensitivity or semantic equivalence claim'}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('folder',type=Path); args=p.parse_args()
    rows=[summarize(f) for f in sorted(args.folder.glob('*/*.json'))]
    print(json.dumps({'completed_files':len(rows),'rows':rows},ensure_ascii=False,indent=2))
