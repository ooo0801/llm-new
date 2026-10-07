"""Post-hoc accepted-only panels, exhaustively paired with their own originals."""
import csv,hashlib,itertools,json,tarfile
from pathlib import Path
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--data-dir', type=Path, required=True, help='Local experiment backup directory')
_data_dir = _parser.parse_args().data_dir.resolve()
import numpy as np
H=_data_dir;R=H/'paired-prompt-budget-v2';O=H/'accepted_only'
O.mkdir(exist_ok=True)
def read(f):return json.loads(f.read_text(encoding='utf-8'))
def save(f,x):f.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def csvout(name,rows):
 with (O/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
p=read(R/'PLAN.json');d=read(R/'RESULTS.json');rank=read(R/'RANKING.json')
archive=H/'paired-prompt-budget-v2-results.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==read(H/'ARCHIVE.json')['sha256']
assert read(H/'LOCAL_AUDIT.json')['status']=='PASS'
files=['PLAN.json','RESULTS.json','RANKING.json','PROMPT_DECISIONS.json']
hashes={}
with tarfile.open(archive) as t:
 for name in files:
  data=(R/name).read_bytes();hashes[name]=hashlib.sha256(data).hexdigest()
  assert data==t.extractfile('paired-prompt-budget-v2/'+name).read()
labels=d['endpoint_order'];assert len(labels)==60
dec={(v['prompt_id'],v['m'],v['n'],v['endpoint']):v for v in read(R/'PROMPT_DECISIONS.json')}
assert len(dec)==50400
accepted={g:[s for s in p['source_order'] if next(x for x in p['mapping'] if x['proxy']==g and x['source_id']==s)['accepted']] for g in p['proxies']}
names={'js':'JS','topk_continuous':'Top-K','raw_logit_l2':'logit','ordinary':'原文'}
details=[];summary=[];ranked=[];single=[]
def evaluate(g,sources,m,n,optimized):
 ids=[p['groups'][g][s] if optimized else s for s in sources]
 alarms=np.array([[dec[i,m,n,e]['alarm'] for e in labels] for i in ids],dtype=bool)
 costs=np.array([[dec[i,m,n,e]['query'] for e in labels] for i in ids])
 alarm=alarms.any(axis=0)
 first=np.where(alarm,alarms.argmax(axis=0),m-1)
 spent=(costs*(np.arange(m)[:,None]<=first)).sum(axis=0)
 assert np.all(spent<=m*n)
 hits=int(alarm[20:].sum());fp=int(alarm[:20].sum())
 return dict(hits=hits,fp=fp,tpr=hits/40,fpr=fp/20,target=hits>=38 and fp<=1,strict=hits==40 and fp==0,
             mean_actual_queries=float(spent.mean()))
def paired(g,sources,m,n,kind):
 a=evaluate(g,sources,m,n,False);b=evaluate(g,sources,m,n,True)
 return dict(group=g,kind=kind,m=m,n=n,budget=m*n,sources=';'.join(sources),
             **{'original_'+k:v for k,v in a.items()},**{'optimized_'+k:v for k,v in b.items()},delta_tpr=b['tpr']-a['tpr'])
for g,sources in accepted.items():
 for m in p['sizes']:
  if m>len(sources):continue
  combos=list(itertools.combinations(sources,m))
  for n in p['queries']:
   rows=[paired(g,s,m,n,'exhaustive') for s in combos];details.extend(rows)
   summary.append(dict(group=g,k=len(sources),m=m,n=n,budget=m*n,panels=len(rows),
       **{prefix+metric:float(np.mean([r[prefix+metric] for r in rows])) for prefix in ['original_','optimized_'] for metric in ['tpr','fpr','target','strict','mean_actual_queries']},
       delta_tpr=float(np.mean([r['delta_tpr'] for r in rows])),
       improved=sum(r['delta_tpr']>1e-12 for r in rows),tied=sum(abs(r['delta_tpr'])<=1e-12 for r in rows),worse=sum(r['delta_tpr']< -1e-12 for r in rows)))
   if m==1:single.extend(rows)
   ordered=[s for s in rank['orders'][g] if s in sources][:m]
   ranked.append(paired(g,ordered,m,n,'frozen_development_order'))
common=[s for s in p['source_order'] if all(s in accepted[g] for g in accepted)]
common_rows=[]
for m in p['sizes']:
 if m>len(common):continue
 for n in p['queries']:
  combos=list(itertools.combinations(common,m))
  for g in ['ordinary']+p['proxies']:
   rows=[evaluate(g,s,m,n,g!='ordinary') for s in combos]
   common_rows.append(dict(group=g,k=len(common),m=m,n=n,panels=len(rows),mean_tpr=float(np.mean([r['tpr'] for r in rows])),mean_fpr=float(np.mean([r['fpr'] for r in rows]))))
csvout('panel_details.csv',details);csvout('panel_summary.csv',summary);csvout('ranked_paired_panels.csv',ranked);csvout('single_prompt_pairs.csv',single)
if common_rows:csvout('common_accepted_sources.csv',common_rows)
# Independent Boolean-OR recomputation of every saved panel's hit/fp count.
for row in details+ranked:
 sources=row['sources'].split(';');g=row['group'];assert set(sources)<=set(accepted[g]) and len(set(sources))==row['m']
 for prefix,opt in [('original_',False),('optimized_',True)]:
  ids=[p['groups'][g][s] if opt else s for s in sources]
  alarm=[any(dec[i,row['m'],row['n'],e]['alarm'] for i in ids) for e in labels]
  assert sum(alarm[20:])==row[prefix+'hits'] and sum(alarm[:20])==row[prefix+'fp']
save(O/'AUDIT.json',dict(status='PASS',input_sha256=hashes,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
 accepted_sources=accepted,common_sources=common,exhaustive_paired_configs=len(details),ranked_paired_configs=len(ranked),
 excludes_fallback=True,matched_source_baselines=True,threshold_uses_actual_panel_size=True,new_model_queries=0))
lines=['# 仅接受改写面板：同源三种优化策略的补充分析','',
'## 主要结果','',
'排除全部回退原文后，JS接受改写相对自身原文仍有较大的检测收益，Top-K也有改善但幅度较小；logit在100次采样下平均退化。这个结论来自同来源配对，不是将不同接受池直接作绝对检测率排名。','',
'单提示词、每条100次时：JS的14条接受来源从38.393%提高到83.929%（+45.536个百分点），Top-K的15条从40.333%提高到66.667%（+26.333个百分点），logit的7条从63.214%降到55.357%（−7.857个百分点）。相应改写后的平均误报率分别为0.714%、0.667%、0%。','',
'JS两条×100次共有91个合法来源组合：平均检测率从61.401%提高到95.632%，双方平均误报率均为0%；满足至少38/40检出且最多1/20误报的组合由7/91（7.692%）增至69/91（75.824%），严格全检出零误报由1/91增至27/91。这表明JS接受产物更容易组成有效小面板，但并不是任意两条都能达标。','',
'Top-K两条×100次的平均检测率从64.238%提高到85.333%，改写平均误报率1.333%；logit则从87.143%下降到73.929%，双方平均误报率0%。通过代理接受条件不等于获得真实检测收益。','',
'三种方法共同接受的5条来源中，100次采样的单提示词平均检测率为原文59.00%、JS87.50%、Top-K76.50%、logit51.50%；这个直接同源对照也支持本批数据中JS表现较好，但样本较小，不宜外推。','',
'本分析明确补充了“成功产出的敏感提示词有多强”的证据，但仍不能据此证明双层结构不可替代，也不能替代后续微观/宏观消融结论。','',
'## 分析对象与统计口径','',
'本报告使用已完成的 paired-prompt-budget-v2（JS、Top-K、logit宏观代理，均与微观目标联合），不是后续微观/宏观消融实验。所有计算在本地完成，没有启动服务器、生成新提示词或新增模型查询。','',
'只允许生成阶段接受的改写进入优化面板。每个优化面板与完全相同来源、相同顺序的原文面板比较。接受标记来自生成阶段，未依据正式测试效果重新筛选。','',
'- 主分析枚举接受池中所有合法来源组合，不再抽样100个子集；避免额外子集抽样误差。组合彼此重叠，不是独立实验重复。',
'- 每条采样n=25/50/100，面板规模沿用m=1/2/5/10/20中不超过接受池大小的值。20条面板均不可用，不以原文或重复提示词补足。',
'- 复用原始逐提示词判定中对应真实m、n的记录，保留原来的多提示词阈值校正；面板任一提示词报警即判攻击。',
'- 每个面板评价40个攻击端点和20个正常采样面板；主达标标准为至少38/40检出且最多1/20误报，严格标准为40/40检出且0/20误报。',
'- 表中平均检测率是各面板检测率的均值；达标比例是满足检测率与误报率联合要求的面板比例，不等于平均检测率，也不是统计置信度。','',
'| 方法 | 接受改写 | 可用m |','|---|---:|---|']
for g,s in accepted.items():lines.append(f"| {names[g]} | {len(s)}/20 | {', '.join(str(m) for m in p['sizes'] if m<=len(s))} |")
lines+=['','## 单提示词：接受改写相对自己的原文','',
'| 方法 | n | 原文检测率 | 改写检测率 | 提升（百分点） | 原文误报率 | 改写误报率 | 改善/持平/退化来源 |','|---|---:|---:|---:|---:|---:|---:|---|']
for r in summary:
 if r['m']==1:lines.append(f"| {names[r['group']]} | {r['n']} | {100*r['original_tpr']:.3f}% | {100*r['optimized_tpr']:.3f}% | {100*r['delta_tpr']:+.3f} | {100*r['original_fpr']:.3f}% | {100*r['optimized_fpr']:.3f}% | {r['improved']}/{r['tied']}/{r['worse']} |")
lines+=['','## 全部接受子集的面板结果','',
'每一行的原文与改写使用同一组来源子集。预算是单端点查询上限m×n；实际顺序提前停止查询数保存在CSV中，未包含离线生成成本。','',
'| 方法 | m | n | 预算 | 组合数 | 原文检测率 | 改写检测率 | 提升pp | 原文/改写误报率 | 原文/改写达标比例 | 原文/改写严格达标比例 |','|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|']
for r in summary:lines.append(f"| {names[r['group']]} | {r['m']} | {r['n']} | {r['budget']} | {r['panels']} | {100*r['original_tpr']:.3f}% | {100*r['optimized_tpr']:.3f}% | {100*r['delta_tpr']:+.3f} | {100*r['original_fpr']:.3f}% / {100*r['optimized_fpr']:.3f}% | {100*r['original_target']:.2f}% / {100*r['optimized_target']:.2f}% | {100*r['original_strict']:.2f}% / {100*r['optimized_strict']:.2f}% |")
lines+=['','## 冻结开发排序的补充分析','',
'从原先冻结的各方法开发排序中删除未接受来源，不重新排序。优化组与原文组使用同一排序前缀。以下仅比较这些指定面板，不从全部测试子集中挑选表现最好的组合；预算最小值仍是测试矩阵上的探索性结果。','',
'| 方法 | 标准 | 同序原文最低预算 | 改写最低预算 |','|---|---|---|---|']
for g in accepted:
 for criterion,label in [('target','≥95%检出且≤5%误报'),('strict','全检出零误报')]:
  cells=[]
  for prefix in ['original_','optimized_']:
   eligible=[r for r in ranked if r['group']==g and r[prefix+criterion]]
   v=min(eligible,key=lambda r:(r['budget'],r['m'])) if eligible else None
   cells.append(f"{v['m']}×{v['n']}={v['budget']}；检出{v[prefix+'hits']}/40，误报{v[prefix+'fp']}/20" if v else '可用配置中未达到')
  lines.append(f"| {names[g]} | {label} | {' | '.join(cells)} |")
lines+=['','## 三种方法共同接受来源的对照','',f"共同接受来源共{len(common)}条：{', '.join(common)}。仅在这个共同来源集合中，三种方法的绝对检测率才具有直接同源可比性，但结论只适用于这个更小的交集。",'',
'| 方法 | m | n | 组合数 | 平均检测率 | 平均误报率 |','|---|---:|---:|---:|---:|---:|']
for r in common_rows:lines.append(f"| {names[r['group']]} | {r['m']} | {r['n']} | {r['panels']} | {100*r['mean_tpr']:.3f}% | {100*r['mean_fpr']:.3f}% |")
lines+=['','## 解释边界与复现','',
'- 这是生成阶段成功产出的条件性分析。不同方法接受池不同，不能用各自接受池的绝对检测率作公平方法排名；应结合配对提升、共同接受来源对照和接受率。',
'- 全来源回退分析衡量固定来源预算下整体效果；本报告衡量接受产物质量，两者互补。接受改写不保证真实检测改善。',
'- 既有40个攻击实例已用于先前分析，本次是补充探索，不是新种子确认；重叠子集及同一模型的正常采样面板不可当作独立模型重复。',
'- 未引入MCC或重新选取测试表现优异的提示词，没有外推到20条全部接受改写的面板。',
f"- AUDIT.json为PASS：核验归档SHA256及输入文件与原归档一致，复核{len(details)}个枚举配对配置和{len(ranked)}个排序配对配置，确认排除回退、来源一致、面板大小阈值一致。原始逐提示词判定此前已通过本地原始响应复算，本次在其上重组面板。",'',
'数据文件：[面板汇总](panel_summary.csv)、[全部面板](panel_details.csv)、[逐来源配对](single_prompt_pairs.csv)、[排序面板](ranked_paired_panels.csv)、[共同接受来源](common_accepted_sources.csv)、[审计记录](AUDIT.json)。','']
(O/'实验结果报告.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(accepted={g:len(s) for g,s in accepted.items()},common=common,panels=len(details),ranked=len(ranked))))
for r in summary:
 if r['n']==100 and r['m'] in [1,2]:print(r)
