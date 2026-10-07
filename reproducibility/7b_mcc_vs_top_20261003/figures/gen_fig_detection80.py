import pathlib,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=pathlib.Path(__file__).resolve().parents[1];R=B;F=B/'figures';F.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))

res=read(R/'RESULTS.json');fp=read(R/'FINGERPRINTS.json');rows=res['configurations']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
colors={'mcc':'#0072B2','top_sensitivity':'#D55E00'};labels={'mcc':'MCC','top_sensitivity':'Top sensitivity'}
fig,axs=plt.subplots(2,2,figsize=(9,6),sharex='col')
for j,m in enumerate([4,8]):
 for method in colors:
  rr=sorted([x for x in rows if x['m']==m and x['method']==method],key=lambda x:x['n']);xs=[x['query_cap'] for x in rr]
  for i,(metric,ci) in enumerate([('tpr','tpr_wilson'),('fpr','fpr_wilson')]):
   ax=axs[i,j];y=np.array([x[metric]*100 for x in rr]);lo=np.array([x[ci][0]*100 for x in rr]);hi=np.array([x[ci][1]*100 for x in rr]);style='o-' if method=='mcc' else 's--'
   ax.plot(xs,y,style,color=colors[method],label=labels[method]);ax.fill_between(xs,lo,hi,color=colors[method],alpha=.10);ax.set_xticks(xs)
 axs[0,j].set_title(f'{m} prompts');axs[0,j].set_ylim(75,102);axs[0,j].axhline(95,color='gray',ls=':',lw=1)
 axs[1,j].set_ylim(-1,27);axs[1,j].axhline(5,color='gray',ls=':',lw=1);axs[1,j].set_xlabel('Query cap (prompts x samples)')
axs[0,0].set_ylabel('Attack detection (%)');axs[1,0].set_ylabel('Normal false alarms (%)');axs[0,0].legend(loc='lower right')
fig.suptitle('Frozen fingerprints: 40 attacks and 20 normal sampling panels');fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(F/f'detection_budget.{ext}',dpi=300)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(9,3.5))
for method in colors:
 v=[fp['panels'][f'{method}_{m}'] for m in [4,8]]
 axs[0].plot([4,8],[x['weighted_candidate_coverage']*100 for x in v],'o-',color=colors[method],label=labels[method])
 axs[1].plot([4,8],[x['mean_sensitivity'] for x in v],'o-',color=colors[method],label=labels[method])
for ax in axs:ax.set_xticks([4,8]);ax.set_xlabel('Fingerprint size');ax.legend()
axs[0].set_ylabel('Weighted candidate-union coverage (%)');axs[1].set_ylabel('Mean normalized joint sensitivity');fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(F/f'coverage_sensitivity.{ext}',dpi=300)
plt.close(fig)
