"""Paired, frozen-config reporting; no test-dependent selection."""
import math
import numpy as np

def wilson(hits,n):
    z=1.959963984540054;p=hits/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-radius),min(1.,center+radius)]

def summarize_decisions(plan,rows):
    by={(r['endpoint'],r['panel'],r['n']):r for r in rows}
    assert len(rows)==len(by)==60*12
    attacks=plan['test_variants'];normals=[f'normal_{i:02d}' for i in range(20)];results=[];comparisons=[]
    for m in [4,8]:
        for n in [25,50,80]:
            for method in ['mcc','top_sensitivity']:
                panel=f'{method}_{m}'
                hits=sum(by[v['variant_id'],panel,n]['alarm'] for v in attacks);fp=sum(by[v,panel,n]['alarm'] for v in normals)
                strata={}
                for family in ['gaussian_noise','finetuning']:
                    for strength in sorted({v['strength'] for v in attacks if v['family']==family}):
                        group=[v for v in attacks if v['family']==family and v['strength']==strength]
                        k=sum(by[v['variant_id'],panel,n]['alarm'] for v in group)
                        strata[f'{family}:{strength}']=dict(hits=k,total=len(group),rate=k/len(group),wilson=wilson(k,len(group)))
                results.append(dict(method=method,m=m,n=n,query_cap=m*n,hits=hits,attack_total=40,tpr=hits/40,tpr_wilson=wilson(hits,40),
                    false_alarms=fp,normal_total=20,fpr=fp/20,fpr_wilson=wilson(fp,20),meets_empirical_target=hits>=38 and fp<=1,strata=strata))
            a=np.array([by[v['variant_id'],f'mcc_{m}',n]['alarm'] for v in attacks],dtype=int)
            b=np.array([by[v['variant_id'],f'top_sensitivity_{m}',n]['alarm'] for v in attacks],dtype=int)
            wins=int(((a==1)&(b==0)).sum());losses=int(((a==0)&(b==1)).sum());d=wins+losses
            exact=min(1.,2*sum(math.comb(d,i) for i in range(min(wins,losses)+1))/(2**d)) if d else 1.
            rng=np.random.default_rng(100900000+m*100+n);diff=[]
            groups=[[i for i,v in enumerate(attacks) if v['family']==family and v['strength']==strength] for family in ['gaussian_noise','finetuning'] for strength in sorted({v['strength'] for v in attacks if v['family']==family})]
            for _ in range(2000):
                ix=np.concatenate([rng.choice(group,5,replace=True) for group in groups]);diff.append(float(np.mean(a[ix]-b[ix])))
            comparisons.append(dict(m=m,n=n,tpr_difference=float(np.mean(a-b)),mcc_only_hits=wins,top_only_hits=losses,
                paired_bootstrap_percentiles=np.percentile(diff,[2.5,50,97.5]).tolist(),mcnemar_exact_unadjusted=exact))
    best={}
    for method in ['mcc','top_sensitivity']:
        valid=sorted([r for r in results if r['method']==method and r['meets_empirical_target']],key=lambda r:(r['query_cap'],r['m'],r['n']))
        best[method]=valid[0] if valid else None
    return dict(configurations=results,paired_comparisons=comparisons,empirically_lowest_budget=best,
        caveat='Test instances are new within known attack families; 20 normal panels are repeated sampling, not independent model implementations. Unadjusted tests are descriptive; sample target is not a population guarantee.')
