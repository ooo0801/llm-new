"""Independent checks and utility reporting for the frozen scaling experiment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from dataclasses import asdict
import numpy as np
import _bootstrap
from run_scaling_matrix import read,save,ident,clean,log,load_responses,check_responses,endpoint_bundle


def bindings(root,p):
    import torch
    from safetensors.torch import load_file
    from stage2_prepare import training_data
    from stage2_attack_calibration import utility_pool
    refs=read(root/'MATRIX_REFERENCE.json')
    assert len(p['prompts'])==len(refs)==20 and len(p['variants'])==40
    assert len(set(r['prompt'] for r in p['prompts']))==20
    assert not {r['prompt'] for r in p['prompts']} & {r['prompt'] for r in training_data()+utility_pool()}
    assert [r['prompt_id'] for r in refs]==[r['id'] for r in p['prompts']]
    for i,r in enumerate(refs):
        cache=root/'matrix_null_mc'/f'{i}.npz'
        assert hashlib.sha256(cache.read_bytes()).hexdigest()==r['null_mc_sha256']
    records=[]
    for v in p['variants']:
        if v['family']!='finetuning':continue
        path=root/'matrix_adapters'/v['variant_id'];report=read(root/'matrix_training'/f"{v['variant_id']}.json")
        assert report['variant']==v and report['completed_steps']==v['configuration']['steps']
        tensors=load_file(str(path/'adapter_model.safetensors'));c=read(path/'adapter_config.json');scale=c['lora_alpha']/c['r'];squared=0.;count=0
        for key,a in tensors.items():
            if '.lora_A.' not in key:continue
            a=a.double();b=tensors[key.replace('.lora_A.','.lora_B.')].double()
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            part=float(torch.trace((b.T@b)@(a@a.T)))*scale**2
            assert part>0;squared+=part;count+=1
        assert count==196
        records.append(dict(variant_id=v['variant_id'],modules=count,effective_delta_frobenius=math.sqrt(squared),adapter_sha256=hashlib.sha256((path/'adapter_model.safetensors').read_bytes()).hexdigest()))
    save(root/'MATRIX_BINDING_AUDIT.json',dict(status='PASS',plan_sha256=ident(p),adapters=records,prompt_scope='Fixed unoptimized pool rows; exact train/utility disjointness, no template independence claim'))


def decoding(root,p):
    import torch
    from transformers import set_seed
    from llm_integrity.modeling import load_model
    bundle=load_model(p['model']);refs=read(root/'MATRIX_REFERENCE.json');records=[]
    try:
        for i,(row,ref) in enumerate(zip(p['prompts'],refs)):
            ids=bundle.tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(bundle.device)
            assert ident(ids[0].tolist())==ref['input_ids_sha256']
            for ti,temp in enumerate(p['temperatures']):
                set_seed(289000000+i*10+ti)
                with torch.inference_mode():result=bundle.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=1,do_sample=True,temperature=temp,top_k=p['top_k'],top_p=p['top_p'],pad_token_id=bundle.tokenizer.pad_token_id,return_dict_in_generate=True,output_scores=True)
                q=torch.softmax(result.scores[0].float(),-1)[0].double().cpu().numpy();expected=np.asarray(ref['probabilities'][ti]);support=ref['token_ids'];outside=float(q.sum()-q[support].sum());error=float(np.max(np.abs(q[support]-expected)))
                assert outside<=1e-6 and np.allclose(q[support],expected,atol=1e-6,rtol=1e-5),(row['id'],temp,error,outside)
                records.append(dict(prompt_id=row['id'],temperature=temp,max_abs_error=error,outside_mass=outside))
        save(root/'MATRIX_DECODING_AUDIT.json',dict(status='PASS',records=records,additional_audit_requests=60))
    finally:bundle.close();clean()


def utility(root,p):
    from run_resf_small import generate,exact_task
    from stage2_attack_calibration import utility_pool
    questions=[r for c in ['logic','math','instruction','structured'] for r in [r for r in utility_pool() if r['category']==c][:8]]
    endpoints=[None]+p['variants'];reports={}
    for ei,row in enumerate(endpoints):
        label='intact' if row is None else row['variant_id'];path=root/'matrix_utility'/f'{label}.json'
        if path.exists():reports[label]=read(path);continue
        log(phase='utility_audit',endpoint=label)
        loaded,b=endpoint_bundle(p,row,root);answers=[]
        try:
            for j,q in enumerate(questions):
                a=generate(b,q,290000000+ei*100+j)
                answers.append(dict(id=q['id'],passed=exact_task(q,a['response']),**a))
            reports[label]=dict(records=answers,execution=asdict(loaded.report) if loaded else None)
            save(path,reports[label])
        finally:
            if loaded:loaded.close()
            else:b.close()
            clean()
    baseline={r['id'] for r in reports['intact']['records'] if r['passed']};summary={}
    for label,value in reports.items():
        lost=[r['id'] for r in value['records'] if r['id'] in baseline and not r['passed']]
        summary[label]=dict(correct=sum(r['passed'] for r in value['records']),total=32,baseline_correct=len(baseline),lost=lost,lost_fraction=len(lost)/len(baseline) if baseline else None)
    save(root/'MATRIX_UTILITY.json',dict(status='REPORT_ONLY',additional_greedy_requests=1312,endpoints=summary))


def results(root,p):
    from llm_integrity.resf_token import detect
    from scipy.stats import beta
    final=read(root/'MATRIX_RESULTS.json');refs=read(root/'MATRIX_REFERENCE.json')
    labels=[f'normal_panel_{i:02d}' for i in range(20)]+[v['variant_id'] for v in p['variants']]
    assert set(final['reports'])==set(labels) and final['plan']==p
    total=0
    for ei,label in enumerate(labels):
        rows=load_responses(root/'matrix_responses'/f'{label}.jsonl');check_responses(rows,p,label,ei)
        assert len(rows)==2000;total+=len(rows)
        decisions=[]
        for pi,ref in enumerate(refs):
            selected=sorted([r for r in rows if r['prompt_id']==ref['prompt_id']],key=lambda r:r['response_index'])
            assert [r['response_index'] for r in selected]==list(range(100))
            with np.load(root/'matrix_null_mc'/f'{pi}.npz') as data:null={int(k):data[k] for k in data.files}
            decisions.append(dict(prompt_id=ref['prompt_id'],**detect([r['token_id'] for r in selected],ref['token_ids'],ref['probabilities'],null,panel_size=20,panel_alpha=.05,e_fraction=.2,looks=tuple(p['looks']))))
        expected=final['reports'][label]
        # BLAS reductions can differ at roundoff level between Linux and Windows.
        # Only the raw statistic gets tolerance; alarms, rules, looks and p-values
        # must still match exactly.
        assert len(decisions)==len(expected['prompts'])
        for actual,stored in zip(decisions,expected['prompts']):
            assert len(actual['traces'])==len(stored['traces'])
            for a,s in zip(actual['traces'],stored['traces']):
                if a['statistic'] is not None and s['statistic'] is not None:
                    assert math.isclose(a['statistic'],s['statistic'],rel_tol=1e-10,abs_tol=1e-10)
                    a['statistic']=s['statistic']
        assert dict(panel_alarm=any(r['alarm'] for r in decisions),prompts=decisions)==expected
    def rate(labels):
        n=len(labels);k=sum(final['reports'][label]['panel_alarm'] for label in labels)
        return dict(n=n,k=k,rate=k/n,ci95=[float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.])
    groups=[]
    for family in ['gaussian_noise','finetuning']:
        for strength in sorted({v['strength'] for v in p['variants'] if v['family']==family}):
            group=[v['variant_id'] for v in p['variants'] if v['family']==family and v['strength']==strength]
            prompt_counts={r['id']:sum(next(z for z in final['reports'][label]['prompts'] if z['prompt_id']==r['id'])['alarm'] for label in group) for r in p['prompts']}
            groups.append(dict(family=family,strength=strength,**rate(group),per_prompt_hits=prompt_counts))
    save(root/'MATRIX_VERIFIED_SUMMARY.json',dict(status='PASS',responses=total,normal=rate(labels[:20]),groups=groups,ci_scope='Pointwise exact binomial intervals over seed/model panels; not independent prompt-level trials'))
    log(phase='result_audit',responses=total,normal=rate(labels[:20]),groups=[{k:v for k,v in r.items() if k!='per_prompt_hits'} for r in groups])


def main():
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['bindings','decoding','utility','results']);a.add_argument('--root',type=Path,required=True);args=a.parse_args()
    import torch
    torch.set_num_threads(4)
    root=args.root;p=read(root/'MATRIX_PLAN.json')
    try:
        import fcntl
    except ImportError:
        if args.phase!='results':
            raise RuntimeError('GPU/server audits require Linux; only result verification runs on Windows')
        # The local one-shot supervisor owns the result-copy directory.
        results(root,p)
        return
    with (root/('AUDIT_'+args.phase+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        globals()[args.phase](root,p)
if __name__=='__main__':main()
