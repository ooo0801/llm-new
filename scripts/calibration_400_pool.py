"""Prespecified synthetic pool. No public-dataset provenance is claimed.

Five task templates per subtask, four instances each. All answers come from
explicit rules; semantic labels have a stored author rationale, not model votes.
"""
import json, random, hashlib, itertools
from pathlib import Path
from fractions import Fraction
from collections import Counter

CATS = ['logic','math','extraction','classification']
SUBTASKS = [
 ['ordering','conditional','sets','assignment','time_space'],
 ['arithmetic','fractions','equations','geometry_units','word_problems'],
 ['fields','json_xml','facts','delimited','multi_record'],
 ['number_properties','rule_labels','sentiment','entity','intent']]

def build():
    rows=[]
    def add(c,s,t,k,p,a,rule,evidence,difficulty='easy'):
        rows.append(dict(id=f'{CATS[c]}_{s}_{t}_{k}',category=CATS[c],subtask=SUBTASKS[c][s],
          template=f'{CATS[c]}_{s}_{t}',instance=k,difficulty=difficulty,prompt=p,
          expected_answer=str(a),evaluator='exact_match',validation_rule=rule,evidence=evidence,
          provenance='synthetic_prespecified_v1',public_dataset=None))
    names=[['林悦','周宁','吴凡'],['陈雨','赵安','刘星'],['苏禾','何川','唐月'],['李青','王晨','郑平']]
    colors=['红','蓝','绿','黄']
    for k in range(4):
        a,b,c=names[k];n=k+3
        # Logic: order, implication, set membership, exhaustive assignment, temporal/spatial.
        ordered=[a,b,c]
        specs=[(f'{a}比{b}高，{b}比{c}高。谁最高？',a),
          (f'{c}比{b}晚到，{a}比{b}早到。谁第二个到？',b),
          (f'{b}得分高于{c}但低于{a}。谁得分最低？',c),
          (f'三本书按{a}、{b}、{c}的顺序从左到右放置。最右边书的左邻是哪本？',b),
          (f'{a}排在{b}前面，{c}排在{b}后面。{a}前面有几人？',0)]
        for t,(p,ans) in enumerate(specs):add(0,0,t,k,p+'只输出答案。',ans,'total_order',dict(order=ordered,query=t))
        specs=[(f'规则：持有{colors[k]}卡则可入场。{a}持有{colors[k]}卡。能否推出其可入场？','能'),
          (f'规则：开机则灯亮。{a}的设备灯不亮。能否推出其未开机？','能'),
          (f'规则：报名成功则收到短信。{a}收到了短信。能否仅由此推出报名成功？','不能'),
          (f'规则：仅持票者可以入场。{a}已经按规则入场。能否推出其持票？','能'),
          (f'规则：若{a}到场则{b}到场，若{b}到场则{c}到场。{a}已到场。能否推出{c}到场？','能')]
        for t,(p,ans) in enumerate(specs):add(0,1,t,k,p+'只答能或不能。',ans,'propositional_entailment',dict(case=t))
        A={n,n+1,n+3};B={n+1,n+2,n+3}
        specs=[('交集',sorted(A&B)),('并集',sorted(A|B)),('A中但不在B中',sorted(A-B)),('B中但不在A中',sorted(B-A)),('恰好属于其中一个集合',sorted(A^B))]
        for t,(q,ans) in enumerate(specs):add(0,2,t,k,f'A={sorted(A)}，B={sorted(B)}。求{q}的元素。升序输出，用英文逗号分隔。',','.join(map(str,ans)),'set_operation',dict(A=sorted(A),B=sorted(B),operation=t))
        pets=['猫','狗','鸟'];places=['东','中','西'];tasks=['备份','更新','重启']
        specs=[(f'{a}、{b}、{c}各养猫、狗、鸟中的一种且不重复。{a}养猫，{b}不养狗。{c}养什么？','狗'),
          (f'红蓝绿盒各装一件衣物：帽子、外套、围巾。红盒装帽子，蓝盒不装外套。绿盒装什么？记录号{k+1}。','外套'),
          (f'{a}、{b}、{c}各占东、中、西一个座位。{b}坐中间，{a}不坐西边。谁坐西边？',c),
          (f'{a}、{b}、{c}各负责备份、更新、重启一项任务。{c}负责重启，{a}不负责更新。谁负责更新？',b),
          (f'{a}、{b}、{c}分别拿1、2、3号牌且不重复。{a}的号码大于{b}，{c}拿2号。{b}拿几号？',1)]
        for t,(p,ans) in enumerate(specs):add(0,3,t,k,p+'只输出答案。',ans,'unique_assignment',dict(case=t,names=names[k]),'medium')
        weekdays=['星期一','星期二','星期三','星期四','星期五','星期六','星期日']
        specs=[(f'今天是{weekdays[k]}，三天后是星期几？',weekdays[(k+3)%7]),
          (f'{a}在{b}北边，{c}在{b}南边。{a}在{c}哪个方向？','北'),
          (f'从原点向东走{n}米，再向西走{n+2}米。最终在原点哪个方向？','西'),
          (f'任务顺序固定为检查、备份、升级、验收。第{k+1}个任务是什么？',['检查','备份','升级','验收'][k]),
          (f'{a}在上午{8+k}点出发，经过2小时到达。按24小时制，到达时是几点？',10+k)]
        for t,(p,ans) in enumerate(specs):add(0,4,t,k,p+'只输出答案，不加单位。',ans,'time_space_rule',dict(case=t,k=k))
        # Math: task structures differ by template; numeric variants remain explicitly synthetic.
        specs=[(f'{n}加{n+7}',2*n+7),(f'{n+12}减{n}',12),(f'{n}乘{n+2}',n*(n+2)),
          (f'{n*(n+1)}除以{n}',n+1),(f'({n}+2)乘3再减4',(n+2)*3-4)]
        for t,(expr,ans) in enumerate(specs):add(1,0,t,k,f'计算{expr}。只输出数字。',ans,'arithmetic',dict(case=t,n=n))
        specs=[(f'1/{n}与1/{2*n}相加',Fraction(3,2*n)),(f'{n}/8减去1/8',Fraction(n-1,8)),
          (f'{n}/5乘以10/{n+1}',Fraction(2*n,n+1)),(f'{n}/3除以{n+1}/6',Fraction(2*n,n+1)),
          (f'甲乙数量比为{n}:2，合计{(n+2)*3}，求甲的数量',Fraction(n*3))]
        for t,(expr,ans) in enumerate(specs):add(1,1,t,k,f'{expr}。只输出最简分数；分母为1时输出整数。',ans,'fraction_exact',dict(case=t,n=n))
        specs=[(f'x+{n}={n+8}',8),(f'{n}x={n*(n+1)}',n+1),(f'2x-{n}={n+4}',n+2),
          (f'x/{n}+2=5',3*n),(f'3(x-2)={3*n}',n+2)]
        for t,(eq,ans) in enumerate(specs):add(1,2,t,k,f'解方程{eq}，只输出x的数值。',ans,'substitution',dict(equation=eq,case=t,n=n))
        specs=[(f'长方形长{n+2}厘米、宽{n}厘米，面积是多少平方厘米？',n*(n+2)),
          (f'正方形边长{n}米，周长是多少米？',4*n),(f'三角形底{2*n}厘米、高5厘米，面积是多少平方厘米？',5*n),
          (f'{n}千米等于多少米？',1000*n),(f'{n}小时又15分钟，共多少分钟？',60*n+15)]
        for t,(p,ans) in enumerate(specs):add(1,3,t,k,p+'只输出数字。',ans,'units_geometry',dict(case=t,n=n))
        specs=[(f'有{n}盒笔，每盒6支，送出4支，还剩几支？',6*n-4),
          (f'{4*n}个苹果平均分给4人，每人几个？',n),(f'车速每小时{10*n}千米，行驶3小时共多少千米？',30*n),
          (f'商品原价{100*n}元，打八折后价格多少元？',80*n),(f'共有{n+5}人，每车最多坐4人，至少需要几辆车？',(n+8)//4)]
        for t,(p,ans) in enumerate(specs):add(1,4,t,k,p+'只输出数字。',ans,'word_problem_exact',dict(case=t,n=n),'medium')
        # Extraction: five input structures per subtask, traceable to exact fields/spans.
        city=['苏州','杭州','成都','厦门'][k];code=['R17','M28','Q39','T46'][k]
        fields={'姓名':a,'城市':city,'编号':code,'颜色':colors[k],'状态':['待发','已发','退回','签收'][k]}
        formats=['；'.join(f'{q}={v}' for q,v in fields.items()),' / '.join(f'{q}：{v}' for q,v in reversed(list(fields.items()))),
          '\n'.join(f'{q}: {v}' for q,v in fields.items()),'，'.join(f'【{q}】{v}' for q,v in fields.items()),' '.join(f'{q}({v})' for q,v in fields.items())]
        for t,q in enumerate(fields):add(2,0,t,k,f'记录：{formats[t]}。只提取{q}的值。',fields[q],'field_lookup',dict(fields=fields,target=q))
        cases=[(json.dumps({'name':a,'city':city},ensure_ascii=False),'city',city),
          (json.dumps({'user':{'id':code,'name':b}},ensure_ascii=False),'user.id',code),
          (f'<item><color>{colors[k]}</color><status>有效</status></item>','color',colors[k]),
          (json.dumps([{'id':1,'city':city},{'id':2,'city':'北京'}],ensure_ascii=False),'第一项的city',city),
          (f'<order id="{code}"><buyer>{c}</buyer></order>','buyer',c)]
        for t,(doc,q,ans) in enumerate(cases):add(2,1,t,k,f'文本：{doc}。提取{q}，只输出其值。',ans,'structured_lookup',dict(document=doc,target=q))
        cases=[(f'{a}在{city}工作，职位为编辑。','工作城市',city),(f'会议于10月{12+k}日举行，由{b}主持。','主持人',b),
          (f'{c}借了一本《{["山海记","远行录","植物笔记","古城故事"][k]}》，归还日为周五。','书名',['山海记','远行录','植物笔记','古城故事'][k]),
          (f'快件编号{code}，送往{city}，收件人为{a}。','快件编号',code),(f'{b}购买了{colors[k]}色外套和黑色鞋子。','外套颜色',colors[k])]
        for t,(doc,q,ans) in enumerate(cases):add(2,2,t,k,f'{doc}请提取{q}，只输出原文中的答案。',ans,'literal_span',dict(document=doc,target=q))
        cases=[(f'{a}|{city}|{code}','|',1),(f'{code},{b},{city}',',',0),(f'{colors[k]};{code};{c}',';',2),
          (f'{city}/{colors[k]}/{code}','/',1),(f'{a}#{code}#{city}','#',2)]
        for t,(doc,sep,index) in enumerate(cases):add(2,3,t,k,f'文本为“{doc}”，分隔符为“{sep}”。从1开始数，只输出第{index+1}段。',doc.split(sep)[index],'split_field',dict(document=doc,delimiter=sep,index=index))
        for t in range(5):
            data=[{'编号':f'{code}-{z}','姓名':names[k][z],'数量':n+z+t} for z in range(3)]
            target=(k+t)%3;q=['姓名','数量','编号','姓名','数量'][t]
            doc='；'.join(','.join(f'{key}={value}' for key,value in r.items()) for r in data)
            selector='编号' if q!='编号' else '姓名'
            add(2,4,t,k,f'{doc}。查找{selector}为{data[target][selector]}的记录，只输出其{q}。',data[target][q],
                'record_lookup',dict(records=data,selector=selector,selector_value=data[target][selector],target=q),'medium')
        # Classification: number/rule labels computable; semantic labels curated before model access.
        x=[-6,0,7,12][k]
        cases=[(f'整数{x}是奇数还是偶数？','偶数' if x%2==0 else '奇数'),
          (f'将{x}按正数、零、负数分类。','正数' if x>0 else '负数' if x<0 else '零'),
          (f'整数{[2,9,13,21][k]}是质数还是合数？',['质数','合数','质数','合数'][k]),
          (f'{[1,4,8,16][k]}是否为完全平方数？回答是或否。',['是','是','否','是'][k]),
          (f'{[3,10,18,25][k]}能否被3整除？回答能或不能。',['能','不能','能','不能'][k])]
        for t,(p,ans) in enumerate(cases):add(3,0,t,k,p+'只输出类别。',ans,'integer_property',dict(case=t,k=k,x=x))
        cases=[(f'规则：分数至少60为通过，否则为未通过。分数为{[59,60,81,42][k]}。',['未通过','通过','通过','未通过'][k]),
          (f'规则：编号以A开头归甲类，否则归乙类。编号为{["A31","B20","C18","A99"][k]}。',['甲类','乙类','乙类','甲类'][k]),
          (f'规则：温度低于0为冻结，否则为非冻结。温度为{[-2,0,8,-1][k]}度。',['冻结','非冻结','非冻结','冻结'][k]),
          (f'规则：只有同时持票且成年才为允许，否则为拒绝。此人{["持票且成年","持票但未成年","无票且成年","无票且未成年"][k]}。',['允许','拒绝','拒绝','拒绝'][k]),
          (f'规则：红色标为停止，绿色标为通行，其余标为等待。当前颜色为{colors[k]}。',['停止','等待','通行','等待'][k])]
        for t,(p,ans) in enumerate(cases):add(3,1,t,k,p+'只输出对应标签。',ans,'explicit_rule',dict(case=t,k=k))
        sentiment=[
          [('服务周到，我非常满意。','积极'),('产品坏了，我很失望。','消极'),('包裹今天到达。','中性'),('这里太吵，体验糟糕。','消极')],
          [('这部电影精彩极了。','积极'),('剧情拖沓，让我厌烦。','消极'),('影片时长九十分钟。','中性'),('演员表现出色，值得一看。','积极')],
          [('饭菜可口，我很喜欢。','积极'),('菜已经凉了，令人扫兴。','消极'),('菜单有十二道菜。','中性'),('这次用餐让我十分开心。','积极')],
          [('房间干净舒适，很满意。','积极'),('床单脏得难以忍受。','消极'),('房间在三楼。','中性'),('工作人员态度恶劣。','消极')],
          [('课程讲得清楚，收获很大。','积极'),('内容混乱，令人失望。','消极'),('课程安排在周三。','中性'),('讲解生动，我很享受。','积极')]]
        wrappers=['判断情感','给下面评价标注情感','阅读句子并判断态度','对文本进行情感分类','识别这句话的情感倾向']
        for t in range(5):
            sentence,ans=sentiment[t][k];add(3,2,t,k,f'{wrappers[t]}：{sentence}仅输出积极、消极或中性。',ans,'curated_semantic_label',dict(text=sentence,rationale='明确褒贬词或无褒贬的事实陈述'))
        entities=[ [('李白','人物'),('北京','地点'),('猫','动物'),('松树','植物')],
          [('杜甫','人物'),('上海','地点'),('海豚','动物'),('向日葵','植物')],
          [('鲁迅','人物'),('杭州','地点'),('老虎','动物'),('竹子','植物')],
          [('苏轼','人物'),('南京','地点'),('企鹅','动物'),('玫瑰','植物')],
          [('屈原','人物'),('成都','地点'),('大象','动物'),('荷花','植物')]]
        for t in range(5):
            word,ans=entities[t][k];add(3,3,t,k,f'{["判断词语类别","给实体分类","识别对象类型","为词语分组","标注实体类别"][t]}：{word}。从人物、地点、动物、植物中选一项，只输出类别。',ans,'curated_semantic_label',dict(entity=word,rationale='常用词的字面实体类别'))
        intentions=[ [('请帮我订明天的车票。','请求'),('会议几点开始？','询问'),('我已经提交材料。','陈述'),('谢谢你的帮助！','感谢')],
          [('麻烦把窗户关上。','请求'),('这本书多少钱？','询问'),('火车今天准点到达。','陈述'),('非常感谢你的提醒。','感谢')],
          [('请将报告发给我。','请求'),('图书馆在哪里？','询问'),('我住在学校附近。','陈述'),('多谢你替我值班。','感谢')],
          [('请为我保留座位。','请求'),('这趟车经过医院吗？','询问'),('他今天去了办公室。','陈述'),('感谢你耐心讲解。','感谢')],
          [('劳驾递给我一支笔。','请求'),('明天是否下雨？','询问'),('电脑放在桌子上。','陈述'),('谢谢你寄来的礼物。','感谢')]]
        for t in range(5):
            sentence,ans=intentions[t][k];add(3,4,t,k,f'{["判断句子意图","识别说话目的","给话语分类","标注沟通意图","判断表达类型"][t]}：{sentence}仅输出请求、询问、陈述或感谢。',ans,'curated_semantic_label',dict(text=sentence,rationale='显式请求词、疑问形式、事实陈述或感谢词'))
    rows.sort(key=lambda r:(CATS.index(r['category']),SUBTASKS[CATS.index(r['category'])].index(r['subtask']),r['id']))
    assert len(rows)==400 and len({r['prompt'] for r in rows})==400
    assert all(v==4 for v in Counter(r['template'] for r in rows).values())
    return rows

def split(rows):
    ids=set()
    for ci,cat in enumerate(CATS):
        for si,sub in enumerate(SUBTASKS[ci]):
            group=sorted([r for r in rows if r['category']==cat and r['subtask']==sub],key=lambda r:r['id'])
            assert len(group)==20
            ids.update(r['id'] for r in random.Random(930010000+ci*100+si).sample(group,5))
    cal=[r for r in rows if r['id'] in ids];opt=[r for r in rows if r['id'] not in ids]
    assert len(cal)==100 and len(opt)==300
    return cal,opt

def export(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=build();cal,opt=split(rows)
    for name,data in [('MOTHER_POOL',rows),('CALIBRATION_POOL',cal),('OPTIMIZATION_POOL',opt)]:
        (out/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*POOL.json'))}
    (out/'POOL_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(counts=[len(rows),len(cal),len(opt)],hashes=manifest)))

if __name__=='__main__':
    import sys
    export(sys.argv[1])
