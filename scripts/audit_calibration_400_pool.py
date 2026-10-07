"""Independent answer-rule and split checks; no model scoring used for pool selection."""
import json,re,itertools
from fractions import Fraction as F
from collections import Counter
import xml.etree.ElementTree as ET
from calibration_400_pool import CATS,SUBTASKS,build,split

def expected(r):
    c=CATS.index(r['category']);s=SUBTASKS[c].index(r['subtask']);t=int(r['template'].split('_')[-1]);k=r['instance'];n=k+3;e=r['evidence']
    if c==0:
        if s==0:return [e['order'][0],e['order'][1],e['order'][2],e['order'][1],0][t]
        if s==1:
            # Enumerate truth assignments satisfying each premise, then test conclusion.
            worlds=list(itertools.product([False,True],repeat=3));ok=[]
            for a,b,c in worlds:
                premise=[(not a or b) and a,(not a or b) and not b,(not a or b) and b,(not b or a) and b,(not a or b) and (not b or c) and a][t]
                conclusion=[b,not a,a,a,c][t]
                if premise:ok.append(conclusion)
            return '能' if all(ok) else '不能'
        if s==2:
            A=set(e['A']);B=set(e['B']);v=[A.intersection(B),A.union(B),A.difference(B),B.difference(A),A.symmetric_difference(B)][t]
            return ','.join(str(x) for x in sorted(v))
        if s==3:
            names=e['names']
            values=[['猫','狗','鸟'],['帽子','外套','围巾'],['东','中','西'],['备份','更新','重启'],[1,2,3]][t]
            valid=[]
            for p in itertools.permutations(values):
                constraint=[p[0]=='猫' and p[1]!='狗',p[0]=='帽子' and p[1]!='外套',p[1]=='中' and p[0]!='西',p[2]=='重启' and p[0]!='更新',isinstance(p[0],int) and p[0]>p[1] and p[2]==2][t]
                if constraint:valid.append(p)
            assert len(valid)==1
            p=valid[0]
            return [p[2],p[2],names[p.index('西')] if t==2 else None,names[p.index('更新')] if t==3 else None,p[1]][t]
        return [['星期四','星期五','星期六','星期日'][k],'北','西',['检查','备份','升级','验收'][k],10+k][t]
    if c==1:
        if s==0:return [sum([n,n+7]),n+12-n,n*(n+2),F(n*(n+1),n),(n+2)*3-4][t]
        if s==1:return [F(1,n)+F(1,2*n),F(n,8)-F(1,8),F(n,5)*F(10,n+1),F(n,3)/F(n+1,6),F(n,(n+2))*((n+2)*3)][t]
        if s==2:
            candidates=[]
            for x in range(-100,101):
                if [x+n==n+8,n*x==n*(n+1),2*x-n==n+4,F(x,n)+2==5,3*(x-2)==3*n][t]:candidates.append(x)
            assert len(candidates)==1;return candidates[0]
        if s==3:return [n*(n+2),sum([n]*4),F(2*n*5,2),n*1000,n*60+15][t]
        return [6*n-4,F(4*n,4),10*n*3,F(100*n*8,10),-(-(n+5)//4)][t]
    if c==2:
        if s==0:return e['fields'][e['target']]
        if s==1:
            doc=e['document']
            if doc.startswith('<'):return ET.fromstring(doc).find(e['target']).text
            d=json.loads(doc)
            return d['city'] if t==0 else d['user']['id'] if t==1 else d[0]['city']
        if s==2:
            patterns=[r'在(.+?)工作',r'由(.+?)主持',r'《(.+?)》',r'快件编号(.+?)，',r'购买了(.+?)色外套']
            return re.search(patterns[t],e['document']).group(1)
        if s==3:return e['document'].split(e['delimiter'])[e['index']]
        matches=[q for q in e['records'] if q[e['selector']]==e['selector_value']];assert len(matches)==1;return matches[0][e['target']]
    if s==0:
        if t==0:return '偶数' if [-6,0,7,12][k]%2==0 else '奇数'
        if t==1:return ['负数','零','正数','正数'][k]
        if t==2:
            x=[2,9,13,21][k];return '质数' if all(x%d for d in range(2,x)) else '合数'
        if t==3:
            x=[1,4,8,16][k];return '是' if any(z*z==x for z in range(x+1)) else '否'
        return '能' if [3,10,18,25][k]%3==0 else '不能'
    if s==1:
        return [('通过' if [59,60,81,42][k]>=60 else '未通过'),('甲类' if ['A31','B20','C18','A99'][k].startswith('A') else '乙类'),
                ('冻结' if [-2,0,8,-1][k]<0 else '非冻结'),('允许' if k==0 else '拒绝'),['停止','等待','通行','等待'][k]][t]
    # Author labels are semantically reviewed, rather than claimed as an independent oracle.
    if s==2:
        text=e['text'];positive=['满意','精彩','出色','喜欢','开心','收获很大','享受'];negative=['失望','糟糕','厌烦','扫兴','难以忍受','恶劣']
        return '积极' if any(x in text for x in positive) else '消极' if any(x in text for x in negative) else '中性'
    if s==3:return ['人物','地点','动物','植物'][k]
    return ['请求','询问','陈述','感谢'][k]

def audit(rows,cal,opt):
    assert len(rows)==400 and len({r['id'] for r in rows})==400 and len({r['prompt'].strip() for r in rows})==400
    assert len(cal)==100 and len(opt)==300
    assert {r['id'] for r in cal}.isdisjoint(r['id'] for r in opt)
    assert {r['id'] for r in rows}=={r['id'] for r in cal+opt}
    for r in rows:assert str(expected(r))==r['expected_answer'],(r['id'],expected(r),r['expected_answer'])
    for pool,n,each in [(rows,100,20),(cal,25,5),(opt,75,15)]:
        assert Counter(r['category'] for r in pool)==dict.fromkeys(CATS,n)
        assert all(v==each for v in Counter((r['category'],r['subtask']) for r in pool).values())
    assert all(v==4 for v in Counter(r['template'] for r in rows).values())
    assert split(rows)==(cal,opt)
    return dict(status='PASS',answers=400,strata=20,templates=100,semantic_labels='author reviewed plus lexical sanity checks; not ground truth from external benchmarks')

if __name__=='__main__':
    rows=build();print(json.dumps(audit(rows,*split(rows))))
