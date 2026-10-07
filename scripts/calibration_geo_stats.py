"""CPU-only, paired category-bootstrap statistics for the frozen 100-text run."""
import numpy as np
CATS=['logic','math','extraction','classification']

def summarize_arrays(u,g,d,h,categories,geometric=True):
    sm=np.median(u,axis=0)
    cm=np.array([np.median(d[np.array(categories)==c],axis=0) for c in CATS])
    valid=np.all(np.isfinite(sm)) and np.all(sm>1e-20) and np.all(np.isfinite(cm)) and np.all(cm>1e-20)
    if not valid:return dict(valid=False,micro=sm.tolist(),category_medians=cm.tolist(),reason='nonfinite or <=1e-20 category median/micro scale')
    sd=np.exp(np.mean(np.log(cm),axis=0)) if geometric else np.median(d,axis=0)
    if np.any(sd<=1e-20) or not np.all(np.isfinite(sd)):return dict(valid=False,reason='invalid macro scale')
    mg=float(np.median(g/sm));ag=float(np.median(h/sd))
    if not np.isfinite(mg) or mg<=1e-20 or not np.isfinite(ag):return dict(valid=False,reason='invalid gradient ratio denominator')
    w=ag/mg
    return dict(valid=True,micro=sm.tolist(),macro=sd.tolist(),category_medians=cm.tolist(),
        gradient_micro_median=mg,gradient_macro_median=ag,micro_weight_unclipped=w,
        micro_weight=float(np.clip(w,.1,10)),macro_weight=1.,weight_clipped=w<.1 or w>10)

def calculate(plan,mi,ma):
    blocks=plan['blocks'];families=list(plan['family_weights']);n=len(plan['selected'])
    micro={(r['source_index'],r['block']['block_type']):r for r in mi}
    macro={(r['source_index'],r['family']):r for r in ma}
    assert len(mi)==len(micro)==n*7 and len(ma)==len(macro)==n*5
    u=np.array([[micro[i,b]['raw_micro_score'] for b in blocks] for i in range(n)])
    g=np.array([[micro[i,b]['raw_embedding_gradient_norm'] for b in blocks] for i in range(n)])
    d=np.array([[macro[i,f]['score'] for f in families] for i in range(n)])
    h=np.array([[macro[i,f]['gradient'] for f in families] for i in range(n)])
    assert all(np.isfinite(a).all() and (a>=0).all() for a in [u,g,d,h])
    categories=np.array([r['category'] for r in plan['selected']])
    def named(s,method):
        s=dict(s,macro_aggregation=method)
        if 'micro' in s:s['micro_scales']=dict(zip(blocks,s.pop('micro')))
        if 'macro' in s:s['macro_scales']=dict(zip(families,s.pop('macro')))
        if 'category_medians' in s:s['category_medians']={c:dict(zip(families,row)) for c,row in zip(CATS,s['category_medians'])}
        return s
    labels=['category_median_equal_weight_geometric_mean','pooled_median_diagnostic']
    summaries={label:named(summarize_arrays(u,g,d,h,categories,geo),label) for label,geo in zip(labels,[True,False])}
    rng=np.random.default_rng(930030000);boots={label:[] for label in labels};invalid={label:0 for label in labels}
    for _ in range(1000):
        ix=np.concatenate([rng.choice(np.flatnonzero(categories==c),25,replace=True) for c in CATS])
        for label,geo in zip(labels,[True,False]):
            s=summarize_arrays(u[ix],g[ix],d[ix],h[ix],categories[ix],geo)
            if not s['valid']:invalid[label]+=1;continue
            boots[label].append(s['micro']+s['macro']+[s['micro_weight'],s['micro_weight_unclipped']])
    intervals={}
    for label in labels:
        a=np.array(boots[label]);keys=blocks+families+['micro_weight','micro_weight_unclipped']
        intervals[label]=dict(valid=len(a),invalid=invalid[label],percentiles={k:np.percentile(a[:,j],[2.5,50,97.5]).tolist() for j,k in enumerate(keys)} if len(a) else {})
    contributions={}
    for label in labels:
        s=summaries[label]
        if not s['valid']:continue
        nd=d/np.array(list(s['macro_scales'].values()))
        contributions[label]={c:dict(zip(families,np.mean(nd[categories==c],axis=0).tolist())) for c in CATS}
    rawstats={}
    for kind,values,grads,keys in [('micro',u,g,blocks),('macro',d,h,families)]:
        for j,key in enumerate(keys):
            rawstats[key]={c:dict(count=int(len(values[categories==c])),percentiles=np.percentile(values[categories==c,j],[0,25,50,75,100]).tolist(),zero_scores=int((values[categories==c,j]==0).sum()),zero_gradients=int((grads[categories==c,j]==0).sum())) for c in CATS}
    return dict(summaries=summaries,bootstrap=dict(seed=930030000,repetitions=1000,methods=intervals,replicates=boots),
        normalized_category_mean_contributions=contributions,raw_statistics=rawstats)
