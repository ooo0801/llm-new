import pathlib,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=pathlib.Path(__file__).resolve().parent;R=B/'detection80_backup/runs/fingerprint-7b-mcc-vs-top-v1';F=B/'figures';F.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
a=read(B/'DETECTION80_LOCAL_AUDIT.json');assert a['status']=='PASS'
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
with (B/'detection80_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
 keys=['method','m','n','query_cap','hits','attack_total','tpr','false_alarms','normal_total','fpr'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
lines=['# 7B敏感提示词指纹：MCC与敏感度排序检测对照','',
'## 主要结果','',
'本轮没有观察到MCC相对敏感度排序的检测优势。在最低预算4条提示词×每条25次查询（100次查询）下，敏感度排序检出40/40攻击，MCC检出37/40；二者均无正常误报。其余五个预算配置中两者攻击检出均为40/40。MCC的8×50配置出现1/20正常误报，其余配置均为0/20。','',
'MCC提高了本实验定义下的组件覆盖率，但覆盖增益没有转化为本轮检测增益。4×25的配对McNemar未校正p=0.25，不能据此断言总体上排序显著优于MCC，也不能声称两种方法已被证明等效。','',
'## 固定设计与数据','',
'Qwen2.5-7B-Instruct，BF16、eager attention。原400条人工合成母池中100条用于校准；待优化池经用户预算修订从300→100→80。最终80条四类各20、20个子任务各4，固定随机种子嵌套抽样，不按得分挑选。61条接受改写且文本唯一，19条未接受；另28条额外历史搜索仅归档。接受改写是代理目标改善，不保证语义不变或独立检测成功。','',
'共同候选池统一计算7类微观分数、5种修改各5变体的JS均值及归一化联合得分；微观权重9.619061603348785。MCC按激活组件加权覆盖贪心选择，Top按联合敏感度排序；各分支4条是8条的前缀，跨分支总共15条唯一提示词。选择在测试训练、采样前冻结。','',
'检测使用首token分布的RESF启发式协议：实际temperature=0.7、top-k=50、top-p=0.9；正常参考温度网格0.5/0.7/0.9，MC每个网格成员、每个n为10000次。面板alpha=0.05，E规则份额0.2，P规则份额0.8，并按提示词数量分配。每个n单独终点判定，不声称12配置整体同时控制误报。','',
'40个新攻击端点=高斯4强度（0.001/0.0025/0.004/0.006）×5种子 + LoRA4步数（5/10/20/40）×5种子。测试LoRA使用独立的冻结32条合成常识训练数据；20个正常面板来自同一正常模型的独立采样，不是20个不同模型。全部端点各15条×80次，共72000条新响应；n25/50是80响应前缀，共享提示词共享采样。','',
'## 12组完整结果','',
'|选择方法|指纹条数m|每条查询n|查询预算|攻击检出|检测率|正常误报|误报率|',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for x in rows:lines.append(f"|{labels[x['method']]}|{x['m']}|{x['n']}|{x['query_cap']}|{x['hits']}/40|{100*x['tpr']:.1f}%|{x['false_alarms']}/20|{100*x['fpr']:.1f}%|")
lines+=['','查询预算是单端点检验的m×n上限；不包括生成、校准、MCC特征提取或测试模型训练成本。','',
'按预定样本目标“检出≥38/40且误报≤1/20”，本次测试网格中Top的最低达标预算为100（4×25）；MCC为200（4×50，另8×25也为200）。这只是本次测试上的描述性最低预算，不是另有独立验证集确认的最终部署配置。','',
'## 分层与配对结果','',
'MCC在4×25时漏检3个高斯端点，分别来自0.001、0.0025、0.004，每层4/5；0.006为5/5。该配置的20个LoRA均检出。Top在该配置及其他配置的所有8个攻击分层均为5/5。每层只有5个实例，不能据5/5认定普遍可靠。','',
'|m|n|MCC−Top检测率差|仅MCC命中|仅Top命中|配对bootstrap 95%区间|McNemar p（未校正）|','|---:|---:|---:|---:|---:|---|---:|']
for x in res['paired_comparisons']:
 ci=x['paired_bootstrap_percentiles'];lines.append(f"|{x['m']}|{x['n']}|{100*x['tpr_difference']:.1f}个百分点|{x['mcc_only_hits']}|{x['top_only_hits']}|[{100*ci[0]:.1f}, {100*ci[2]:.1f}]个百分点|{x['mcnemar_exact_unadjusted']:.3g}|")
lines+=['','分层配对bootstrap只在当前8个攻击分层的5个种子内重采样，不包含提示词生成、校准与模型选择的不确定性；零宽度区间仅说明当前观察配对差均为零。','',
'## 覆盖率与科研解释','',
'|分支|规模|候选组件并集加权覆盖率|平均联合敏感度|','|---|---:|---:|---:|']
for key,x in fp['panels'].items():lines.append(f"|{labels[x['method']]}|{x['size']}|{100*x['weighted_candidate_coverage']:.2f}%|{x['mean_sensitivity']:.3f}|")
lines+=['','MCC覆盖率更高，而Top平均敏感度更高。这说明两个选择目标确实产生了不同面板；本轮低预算检测更偏向高敏感度。组件覆盖率分母只是61个候选激活组件的并集，不能解释为模型全部参数或能力的覆盖率。','',
'本轮可以支持：冻结的敏感度排序指纹在当前7B、已知攻击家族的新实例上表现较好；尚不能支持MCC选择具有额外检测价值。本轮没有同源普通提示词指纹分支，所以也不能单独证明敏感提示词优化相对普通提示词的收益。','',
'40/40的Wilson 95%区间约为91.24%—100%；0/20正常误报的区间约为0—16.11%。因此观察达标不等于总体检测率已证明≥95%、总体误报率已证明≤5%。当前证据不外推到未知攻击家族、其他模型规模、其他解码条件或所有自然语言提示词。','',
'## 图与核验','',
'![预算与检测效果](figures/detection_budget.png)','',
'阴影为逐配置Wilson 95%区间；曲线点有重叠。误报面板纵轴包含区间上界，避免把0/20画成确定无误报。','',
'![覆盖率与敏感度](figures/coverage_sensitivity.png)','',
f"本地独立审计PASS：归档{a['files']}文件、校准依赖{a['calibration_files']}文件；80来源、61候选、1525宏观记录、4指纹、20测试适配器、72000响应、720面板判定、12配置。独立复算评分、贪心最大增益、参考MC缓存及全部E/P告警；未在本地GPU重算原始logits和激活特征。",
'','结果原始数据：detection80_backup/runs/fingerprint-7b-mcc-vs-top-v1；完整性报告：DETECTION80_LOCAL_AUDIT.json；机器可读汇总：detection80_results.csv；曲线同时提供PDF与PNG。',
'',f"完整归档SHA256：`{a['archive_sha256']}`。服务器关机状态另见DETECTION_SHUTDOWN_RESULT.json。"]
(B/'7B指纹选择与检测实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print('Report, CSV, PNG and PDF written')
