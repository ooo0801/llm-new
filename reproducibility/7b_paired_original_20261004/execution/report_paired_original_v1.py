"""Same-selected-source sensitive versus original paired diagnostic."""
from pathlib import Path
import json,math,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent
R=B/'paired_original_backup/runs/fingerprint-7b-paired-original-v1'
O=B/'detection80_backup/runs/fingerprint-7b-mcc-vs-top-v1'
def read(p):return json.loads(p.read_text(encoding='utf8'))
assert read(B/'PAIRED_ORIGINAL_LOCAL_AUDIT.json')['status']=='PASS'
p=read(R/'PLAN.json');prompts=read(R/'PROMPTS.json')
old=read(O/'RESULTS.json');new=read(R/'RESULTS.json')
sd={(x['endpoint'],x['panel'],x['n']):x for x in read(O/'DECISIONS.json')}
od={(x['endpoint'],x['panel'],x['n']):x for x in read(R/'DECISIONS.json')}
variants=p['test_variants'];endpoints=[x['variant_id'] for x in variants]
groups=[[i for i,v in enumerate(variants) if v['family']==fam and v['strength']==strength] for fam in ['gaussian_noise','finetuning'] for strength in sorted({v['strength'] for v in variants if v['family']==fam})]
records=[];strata=[]
for ix,s in enumerate(old['configurations']):
 o=next(x for x in new['configurations'] if (x['method'],x['m'],x['n'])==(s['method'],s['m'],s['n']))
 name=f"{s['method']}_{s['m']}";n=s['n']
 a=np.array([sd[e,name,n]['alarm'] for e in endpoints],int);b=np.array([od[e,name,n]['alarm'] for e in endpoints],int)
 wins=int(((a==1)&(b==0)).sum());losses=int(((a==0)&(b==1)).sum());d=wins+losses
 exact=min(1.,2*sum(math.comb(d,i) for i in range(min(wins,losses)+1))/2**d) if d else 1.
 rng=np.random.default_rng(p['bootstrap_seed']+ix)
 boot=[float(np.mean((a-b)[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])])) for _ in range(2000)]
 r=dict(method=s['method'],m=s['m'],n=n,budget=s['m']*n,sensitive_hits=s['hits'],original_hits=o['hits'],sensitive_false_alarms=s['false_alarms'],original_false_alarms=o['false_alarms'],difference=float(np.mean(a-b)),sensitive_only=wins,original_only=losses,paired_ci=np.percentile(boot,[2.5,97.5]).tolist(),mcnemar_unadjusted=exact,sensitive_wilson=s['tpr_wilson'],original_wilson=o['tpr_wilson'])
 records.append(r)
 for key,v in s['strata'].items():strata.append(dict(method=s['method'],m=s['m'],n=n,stratum=key,sensitive_hits=v['hits'],original_hits=o['strata'][key]['hits'],total=5))
single=[]
# Same m=8 protocol for within-prompt hit fractions; duplicates across panels omitted.
for x in prompts:
 name=next(name for name,v in read(R/'FINGERPRINTS.json')['panels'].items() if v['size']==8 and x['id'] in v['prompt_ids'])
 for n in p['queries']:
  single.append(dict(source_id=x['source_id'],category=x['category'],n=n,panel_size=8,sensitive_hits=sum(sd[e,name,n]['prompts'][x['sensitive_id']]['alarm'] for e in endpoints),original_hits=sum(od[e,name,n]['prompts'][x['id']]['alarm'] for e in endpoints)))
payload=dict(configurations=records,strata=strata,single_prompt_hits=single,interpretation='Post-hoc same-selected-source paired diagnostic; no selection repeated; no independent confirmation.')
(B/'PAIRED_ORIGINAL_COMPARISON.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
for name,rows in [('paired_original_results',records),('paired_original_strata',strata),('paired_original_single',single)]:
 with (B/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
figdir=B/'paired_original_figures';figdir.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
fig,axes=plt.subplots(2,2,figsize=(10,7),sharey=True)
for ax,(method,m) in zip(axes.flat,[(a,b) for a in ['mcc','top_sensitivity'] for b in [4,8]]):
 rr=[x for x in records if x['method']==method and x['m']==m]
 for prefix,label,color,marker in [('sensitive','Sensitive','#D55E00','o'),('original','Original','#0072B2','s')]:
  xx=[x['budget'] for x in rr];yy=[100*x[prefix+'_hits']/40 for x in rr];ci=np.array([x[prefix+'_wilson'] for x in rr])*100
  ax.plot(xx,yy,color=color,marker=marker,label=label);ax.fill_between(xx,ci[:,0],ci[:,1],color=color,alpha=.12)
 ax.set_title(f"{'MCC' if method=='mcc' else 'Top sensitivity'} | {m} prompts");ax.set_xlabel('Query budget');ax.set_ylabel('Attack detection (%)');ax.set_ylim(-2,103);ax.grid(alpha=.15);ax.legend()
fig.suptitle('Same selected sources; 40 paired attack instances\nShading: pointwise Wilson 95% intervals');fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(figdir/('paired_detection.'+ext),bbox_inches='tight')
plt.close(fig)
lines=['# 7B敏感指纹与同来源普通原文的检测对照','', '## 实验范围','',
'固定上一轮MCC与Top敏感度排序各4/8条指纹，将15条入选敏感提示词逐条替换为优化前的同来源原文。没有重新生成、排名或运行MCC；相同40个攻击端点、20个正常采样面板、每条25/50/80次，所有检测协议一致。原文有自己的参考分布及MC校准。', '',
'模型、软件和20个LoRA适配器身份核验通过；15条敏感参考重新计算通过严格复用门槛。敏感侧复用旧72000响应；普通侧独立新种子72000响应。正常20面板来自一个正常模型的重复采样。', '',
'这是看到此前结果后补充的同来源诊断，回答最终入选来源的文本改写收益，不能替代普通池独立选择基线或新的独立确认实验。', '',
'## 检测与误报','', '|方法|条数|每条次数|预算|敏感检出/40|原文检出/40|敏感误报/20|原文误报/20|检测率差（百分点）|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in records:lines.append(f"|{r['method']}|{r['m']}|{r['n']}|{r['budget']}|{r['sensitive_hits']}|{r['original_hits']}|{r['sensitive_false_alarms']}|{r['original_false_alarms']}|{r['difference']*100:.1f}|")
lines+=['','## 配对不确定性','','|方法|m|n|仅敏感命中|仅原文命中|差值95%区间（百分点）|McNemar未校正p|','|---|---:|---:|---:|---:|---|---:|']
for r in records:lines.append(f"|{r['method']}|{r['m']}|{r['n']}|{r['sensitive_only']}|{r['original_only']}|[{r['paired_ci'][0]*100:.1f}, {r['paired_ci'][1]*100:.1f}]|{r['mcnemar_unadjusted']:.5g}|")
lines+=['','配对bootstrap按攻击类型×强度在每层5个种子中重采样2000次，不包含生成或选择的不确定性；区间为逐配置、未进行多重比较校正。零宽度区间不代表总体差异确定为零。', '',
'## 结果解读','']
for method in ['mcc','top_sensitivity']:
 r=next(x for x in records if x['method']==method and x['m']==4 and x['n']==25)
 lines.append(f"在最低预算100次查询时，{method}敏感侧检出{r['sensitive_hits']}/40，普通原文检出{r['original_hits']}/40，观察差值为{r['difference']*100:.1f}个百分点。是否存在可推广的优势还应结合上述区间和误报率，不能仅凭观察比例断言。")
lines+=['','本对照只评估已被选中来源的改写效果；不证明整个敏感生成流程优于从全部普通池精心选出的指纹。攻击家族已知且当前端点此前已经测过，不能称为独立确认。40/40也不等于总体检测率被证明100%。', '',
'## 分层及单条提示词','','完整8个类型/强度分层见paired_original_strata.csv；15条来源的单条命中统计见paired_original_single.csv。单条命中采用m=8的同一多重检验阈值，仅作为贡献诊断。','',
'![同来源检测对照](paired_original_figures/paired_detection.png)','',
'完整性：PAIRED_ORIGINAL_LOCAL_AUDIT.json为PASS；归档及父实验依赖全部哈希通过，15对来源、参考复用、MC、72000响应、720判定及12配置独立重算核验通过。']
(B/'7B敏感指纹与同源原文检测对照报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(status='REPORT_READY',configurations=len(records))))
