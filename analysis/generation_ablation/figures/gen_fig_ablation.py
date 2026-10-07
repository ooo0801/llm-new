"""Reproducible descriptive report; all numbers come from audited frozen results."""
import csv,json
from pathlib import Path
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--data-dir', type=Path, required=True, help='Local experiment backup directory')
_data_dir = _parser.parse_args().data_dir.resolve()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=_data_dir;B=H/'generation-ablation-v1';R=B/'evaluation';F=H/'figures'
def read(f):return json.loads(f.read_text(encoding='utf-8'))
assert read(H/'LOCAL_AUDIT.json')['status']=='PASS'
p=read(R/'PLAN.json');d=read(R/'RESULTS.json')
styles=[('ordinary','Original','原始','#0072B2','o'),('micro_only','Micro only','仅微观','#009E73','^'),('macro_only','JS only','仅JS宏观','#CC79A7','D'),('joint_js','Micro + JS','微观＋JS','#D55E00','s')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300,'legend.frameon':False})
def export(fig,name):
 for ext in ['png','pdf']:fig.savefig(F/f'{name}.{ext}',bbox_inches='tight')
 plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(11,6.5),layout='constrained')
for col,n in enumerate(p['queries']):
 for g,label,_,color,marker in styles:
  rows=sorted([r for r in d['paired_panel_summary'] if r['group']==g and r['n']==n],key=lambda r:r['m'])
  for row,key in enumerate(['mean_tpr','mean_fpr']):
   axes[row,col].plot([r['budget'] for r in rows],[100*r[key] for r in rows],label=label,color=color,marker=marker)
 for row in [0,1]:
  upper=max(6,1+max(r['mean_fpr']*100 for r in d['paired_panel_summary'] if r['n']==n))
  axes[row,col].set(xscale='log',xlabel='Query cap (m x n)',title=f'n = {n}',ylim=(-1,103) if row==0 else (-.2,upper))
  axes[row,col].axhline(95 if row==0 else 5,color='gray',linestyle='--',linewidth=.8);axes[row,col].grid(alpha=.15)
axes[0,0].set_ylabel('Mean attack detection (%)');axes[1,0].set_ylabel('Mean false alarm (%)')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4)
export(fig,'paired_budget_curves')
single={(r['group'],r['n'],r['sources'][0]):r for r in d['panels'] if r['kind']=='paired' and r['m']==1}
fig,axes=plt.subplots(1,3,figsize=(10,3.5),layout='constrained')
comparisons=[]
for ax,n in zip(axes,p['queries']):
 x=[single['macro_only',n,s]['tpr']*100 for s in p['source_order']];y=[single['joint_js',n,s]['tpr']*100 for s in p['source_order']]
 ax.plot([0,100],[0,100],color='gray',linestyle='--');ax.scatter(x,y,color='#D55E00',alpha=.65,s=40)
 ax.set(xlabel='JS only detection (%)',ylabel='Joint detection (%)',title=f'n = {n}',xlim=(-3,103),ylim=(-3,103));ax.grid(alpha=.15)
 for s in p['source_order']:
  a=single['macro_only',n,s];b=single['joint_js',n,s]
  comparisons.append(dict(source_id=s,n=n,macro_hits=a['hits'],joint_hits=b['hits'],delta_hits=b['hits']-a['hits'],macro_fp=a['false_alarms'],joint_fp=b['false_alarms']))
export(fig,'joint_vs_macro_sources')
for name,rows in [('joint_vs_macro_sources',comparisons),('paired_panel_summary',d['paired_panel_summary']),('paired_prompts',d['paired_prompts']),('ranked_panels',[r for r in d['panels'] if r['kind']=='ranked'])]:
 with (H/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# 微观与JS宏观四组消融实验结果','',
'本报告对应 generation-ablation-v1。所有数字取自已冻结结果，并在本地通过逐响应、逐判定和逐面板复算。当前没有MCC选择；独立开发集排序仅为补充分析。','',
'## 核心结论','',
'本轮没有观察到微观＋JS联合优化相对仅JS宏观优化的稳定额外收益，不足以支持双层结构的必要性。三个采样预算下，两者单提示词平均检出率仅相差0.75、0.75和0.25个百分点；100次时20条来源中联合改善6条、持平8条、退化6条。接近不等于已证明统计等效。','',
'微观目标并非完全无效：100次时仅微观平均检出率68.00%，原文50.125%；仅JS为72.75%，联合73.00%。但是微观单独有效，与它在JS之上具有增量贡献，是两个不同的问题。','',
'同来源2条×100次面板中，仅JS平均检出率90.575%，联合89.575%，均无观测误报；联合达标子集比例42%，仅JS40%。平均检出率与达标比例方向不一致，不能选择性只报告有利指标。完整20条面板中，仅JS在25/50/100次都检出40/40，联合依次为38/40、39/40、40/40，均为0/20误报。','',
'独立开发排序后，主标准的观测最低预算为原文200、仅微观125、仅JS250、联合200；严格全检出零误报标准为原文200、仅微观1000、仅JS500、联合500。本轮联合没有比普通排序更省预算，且不能将排序优势解释为生成机制优势。此前实验的严格预算节省没有在本轮重新生成与新采样下复现，不能继续当作稳定结论。','',
'研究判断：应将“JS优化提高平均探针质量”与“微观组件具有不可替代贡献”分开。本轮支持前者的样本内效果，但尚不支持后者。后续可以把仅JS作为更简洁的候选基线，若保留微观则需要解释其作用条件；本次未启动新实验，也未依据测试结果重新筛选提示词。','',
'## 逐来源主分析','',
'每条原文分别对应三种优化结果；不接受的优化保留原文并纳入分析。下表是20条来源等权平均的单提示词检测率，括号内为平均误报率。每来源对应40个攻击端点及20个正常采样面板。','',
'| 方法 | n=25 | n=50 | n=100 |','|---|---:|---:|---:|']
for g,_,zh,_,_ in styles:
 cells=[]
 for n in p['queries']:
  a=[single[g,n,s] for s in p['source_order']]
  cells.append(f"{np.mean([v['tpr'] for v in a])*100:.3f}%（{np.mean([v['fpr'] for v in a])*100:.3f}%）")
 lines.append(f"| {zh} | {' | '.join(cells)} |")
lines+=['','## 核心对比：联合优化相对仅JS宏观','',
'| 每条查询数 | 检测率平均差（百分点） | 改善/持平/退化来源数 | 误报率平均差（百分点） |','|---:|---:|---|---:|']
for n in p['queries']:
 rows=[r for r in comparisons if r['n']==n]
 lines.append(f"| {n} | {np.mean([r['delta_hits'] for r in rows])*2.5:+.3f} | {sum(r['delta_hits']>0 for r in rows)}/{sum(r['delta_hits']==0 for r in rows)}/{sum(r['delta_hits']<0 for r in rows)} | {np.mean([r['joint_fp']-r['macro_fp'] for r in rows])*5:+.3f} |")
lines+=['','以上是同来源的观测差异，不是800个独立模型上的统计结论。未预设联合必优，也不能把持平直接当作等效证明。','',
'## 同来源面板预算矩阵','',
'四组使用相同来源子集：m=1枚举20个来源；m=2/5/10各100个固定子集；m=20一个完整面板。下表依次报告平均检测率、平均误报率、满足检测率≥95%且误报≤5%的子集比例。重叠子集不是独立重复。','',
'| m | n | 方法 | 检测率 | 误报率 | 达标子集比例 |','|---:|---:|---|---:|---:|---:|']
for m in p['sizes']:
 for n in p['queries']:
  for g,_,zh,_,_ in styles:
   r=next(v for v in d['paired_panel_summary'] if v['group']==g and v['m']==m and v['n']==n)
   lines.append(f"| {m} | {n} | {zh} | {100*r['mean_tpr']:.3f}% | {100*r['mean_fpr']:.3f}% | {100*r['target_fraction']:.1f}% |")
lines+=['','## 开发排序补充分析','',
'各组独立按开发攻击检测表现排序，排名在正式采样前冻结。下表是测试矩阵中的观测最低预算，不代表独立确认的最优配置。','',
'| 方法 | ≥38/40检出且≤1/20误报 | 40/40检出且0/20误报 |','|---|---|---|']
for g,_,zh,_,_ in styles:
 cells=[]
 for key in ['meets_target','strict']:
  rows=[r for r in d['panels'] if r['group']==g and r['kind']=='ranked' and r[key]]
  v=min(rows,key=lambda r:(r['budget'],r['m'])) if rows else None
  cells.append(f"{v['m']}×{v['n']}={v['budget']}；检出{v['hits']}/40，误报{v['false_alarms']}/20" if v else '没有达标配置')
 lines.append(f"| {zh} | {' | '.join(cells)} |")
lines+=['','## 各攻击强度：完整20条面板','',
'| 方法 | n | 类型/强度 | 检出 |','|---|---:|---|---:|']
for g,_,zh,_,_ in styles:
 for n in p['queries']:
  r=next(v for v in d['panels'] if v['group']==g and v['kind']=='paired' and v['m']==20 and v['n']==n)
  for attack,hits in r['attack_groups'].items():lines.append(f'| {zh} | {n} | {attack} | {hits}/5 |')
lines+=['','## 生成与完整性','',
'| 方法 | 接受改写 | 回退原文 | 成功保存任务耗时（秒） |','|---|---:|---:|---:|']
for g,_,zh,_,_ in styles[1:]:
 rows=[read(f) for f in (B/'generation/search').glob(g+'_*.json')]
 accepted=sum(r['accepted'] for r in rows)
 lines.append(f"| {zh} | {accepted}/20 | {20-accepted}/20 | {sum(r['seconds'] for r in rows):.1f} |")
audit=read(H/'LOCAL_AUDIT.json')
lines+=['',f"- 唯一文本{len(p['prompts'])}条，开发响应{audit['response_counts']['development']}条，正式响应{audit['response_counts']['test']}条；全部新采集。",
f"- 本地复算{audit['decisions_recomputed']}个提示词判定及{audit['panels_recomputed']}个面板，LOCAL_AUDIT.json为PASS。",
'- 生成任务、来源映射、失败回退、权重和代码快照已核验。仅微观完全跳过宏观模型和宏观门槛；仅宏观完全跳过微观梯度和微观重评分。',
'- 三种优化同来源使用相同搜索种子和候选预算；预算公平指相同搜索轮数、候选及重评分数量，不是相同耗时。中断未保存任务的耗时不计入上表。',
'- 宏观和联合均保留五族不退化门槛；因此联合对仅微观同时增加宏观目标和门槛，不能归因于单一因素。',
'- 普通组是20条原文；没有优化任务。采样相同文本去重复用，避免为同一原文人为制造组间差异。',
'- 既有攻击实例是探索性消融；20正常面板是同一正常模型的独立采样，不是20种部署环境。',
'- 本轮没有MCC指纹选择，没有证明跨模型、跨部署环境或新攻击种子的泛化，也没有进行独立重复生成。',
'- 主结论应优先使用同来源生成收益，再单独解释开发排序收益；测试上观察到的最小预算不等于总体可靠性保证。','',
'![同源预算曲线](figures/paired_budget_curves.png)','',
'![联合与仅宏观逐来源对比](figures/joint_vs_macro_sources.png)','']
(H/'实验结果报告.md').write_text('\n'.join(lines),encoding='utf-8')
print('Report, CSVs and PDF/PNG figures generated.')
