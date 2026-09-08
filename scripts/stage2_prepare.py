"""Freeze attack/data design before any attack or candidate response is observed."""
import json
from pathlib import Path
from _bootstrap import project_path
from llm_integrity.stage1_r1 import canonical, digest, save_json


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"Refusing to change frozen design: {path}")
    else:
        save_json(path,value)


def design():
    variants=[]
    for split,seed_base,gaussian,steps,lr in [
        ("train",20261000,[.006,.012,.024],[5,10,20],1e-4),
        ("confirmation",20262000,[.006,.012,.024],[5,10,20],1e-4),
        ("heldout",20263000,[.008,.016,.032],[6,12,24],8e-5)]:
        seeds=range(1 if split=="train" else 3)
        for fi,family in enumerate(["gaussian_noise","finetuning"]):
            for li,label in enumerate(["weak","medium","strong"]):
                for si in seeds:
                    seed=seed_base+fi*100+si
                    if family=="gaussian_noise":
                        c={"method":"relative_gaussian","std_ratio":gaussian[li],"target_scope":"attention_ffn"}
                    else:
                        c={"method":"lora","rank":8,"alpha":16,"dropout":0.,"learning_rate":lr,
                           "steps":steps[li],"target_scope":"attention_ffn"}
                    variants.append({"variant_id":f"s2_{split}_{family}_{label}_s{si}","family":family,
                                     "split":split,"strength":label,"seed":seed,"configuration":c})
        for family in ["unstructured_pruning","structured_pruning","quantization"]:
            for li in (range(3) if split=="train" else [1]):
                c={}
                if family=="unstructured_pruning":
                    c={"method":"layerwise_magnitude","ratio":[.1,.2,.3][li],"target_scope":"attention_ffn"}
                    if split=="heldout":
                        c["ratio"]=.25
                elif family=="structured_pruning":
                    c={"structure":"ffn_channels","ratio":[.05,.1,.15][li],"selection":"magnitude",
                       "layer_scope":"all_layers","implementation":"mask"}
                    if split=="heldout":
                        c["ratio"]=.125
                else:
                    c={"method":["int8","nf4","fp4"][li],"compute_dtype":"bfloat16",
                       "target_scope":"full_model","double_quant":split=="heldout"}
                variants.append({"variant_id":f"s2_{split}_{family}_{li}","family":family,"split":split,
                                 "strength":"coverage","seed":seed_base+500+li,"configuration":c})
    return variants


def training_data():
    rows=[]
    # Authored, deterministic miniature SFT corpus; no benchmark-generalization claim.
    pairs=[("太阳系中最大的行星是什么？","太阳系中最大的行星是木星。"),
           ("水在标准大气压下的沸点是多少？","水在标准大气压下的沸点是100摄氏度。"),
           ("植物通过什么过程利用阳光制造养分？","植物通过光合作用利用阳光制造养分。"),
           ("地球的天然卫星叫什么？","地球的天然卫星叫月球。"),
           ("解释计算机中的只读文件。","只读文件允许读取内容，但不允许直接修改。"),
           ("为什么实验要记录随机种子？","记录随机种子有助于在相同环境中复现实验。"),
           ("简述训练集与测试集的区别。","训练集用于学习参数，测试集用于评估未见数据上的表现。"),
           ("为什么应保存原始数据？","保存原始数据便于复核处理过程并追溯结果。")]
    translations=[("Save the result.","保存结果。"),("Check the file path.","检查文件路径。"),
                  ("Read the instructions.","阅读说明。"),("The model is ready.","模型已准备好。"),
                  ("Compare the two values.","比较这两个值。"),("Keep the original copy.","保留原始副本。"),
                  ("The experiment has ended.","实验已经结束。"),("Use a separate test set.","使用独立的测试集。")]
    pairs += [(f"将英文翻译成中文：{a}",b) for a,b in translations]
    pairs += [(f"计算{a}+{b}，只输出整数。",str(a+b)) for a,b in [(17,28),(39,12),(24,31),(46,18),(57,23),(63,19),(72,16),(85,14)]]
    pairs += [(f"将词语“{word}”原样输出一次，不添加其他内容。",word) for word in ["山谷","河流","森林","海洋","星辰","晨光","雨滴","白云"]]
    for i,(p,a) in enumerate(pairs):
        rows.append({"id":f"s2_sft_{i:02}","prompt":p,"expected_answer":a,"source":"authored_stage2_mini_sft"})
    return rows


def utility_data():
    pairs=[("只输出计算结果：19+24。","43","math"),
           ("只输出计算结果：7×8。","56","math"),
           ("只输出计算结果：90-37。","53","math"),
           ("只输出计算结果：81÷9。","9","math"),
           ("小林比小王高，小王比小周高。谁最高？只回答名字。","小林","logic"),
           ("红盒比蓝盒重，蓝盒比绿盒重。哪个最轻？只回答盒子的名称。","绿盒","logic"),
           ("严格输出从2到5的整数，使用英文逗号分隔，不添加其他文字。","2,3,4,5","instruction"),
           ("请原样输出：春夏秋冬","春夏秋冬","instruction")]
    return [{"id":f"s2_utility_{i}","prompt":p,"expected_answer":a,"category":c,"evaluator":"exact"}
            for i,(p,a,c) in enumerate(pairs)]


def main():
    cfg=json.loads(project_path("configs/stage2_qwen15b.json").read_text())
    out=project_path(cfg["output"])
    variants=design()
    train=training_data()
    utility=utility_data()
    source=[json.loads(s) for s in project_path(cfg["source_prompts"]).read_text(encoding="utf-8").splitlines()]
    for a,b in [(train,utility),(train,source),(utility,source)]:
        if {r["prompt"] for r in a}&{r["prompt"] for r in b}:
            raise RuntimeError("Data role overlap")
    protocol={"schema":"stage2-design-v1","config_sha256":digest(project_path("configs/stage2_qwen15b.json")),
      "variants":variants,"lora_training":train,"independent_utility":utility,
      "utility_gate":{"minimum_intact_correct":6,"maximum_drop_on_intact_correct":.15,
                      "endpoint_policy":"invalid utility endpoints reported, never counted as successful subtle attacks"},
      "sampling":{"per_prompt_per_endpoint":8,"fit_intact":8,"confirmation_intact":8,"heldout_intact":8,"null_intact":8,
                  "max_union_prompts":40,"maximum_response_rows_excluding_search_task_guards":14720,
                  "note":"upper bound 40*(32+42*8); cache identical prompt strings across methods; no automatic expansion"},
      "comparison":{"first_stage_methods":["legacy","enhanced"],"six_proxy_searches":False,
                    "primary_proxy_js":True,"kl_secondary":True,"top_k":10,
                    "correlation_unit":"source-prompt clustered, family/strength stratified; no all-endpoint pooled significance"},
      "behavior_gate":{"gaussian_and_lora_each_strength":"at least 2/3 seeds raw p<=.05 and positive MMD2; worst-seed components separately reported",
                       "original_task":"intact >=6/8 correct and drop <=1/8 relative to source prompt",
                       "no_strength_pooling":True,"status":"exploratory limited-power screen"},
      "limitations":["8 sources and 8 responses/group are exploratory, not formal power/FPR certification",
                      "initial two-method phase cannot establish pure-behavior baseline or proxy-weight ablation conclusions",
                      "utility is an independent small exact-task set, not a broad capability benchmark",
                      "LoRA levels independently trained with linear schedules; not shared-trajectory checkpoints"]}
    freeze(out/"EXPERIMENT_DESIGN.json",protocol)
    print(json.dumps({"frozen":str(out/"EXPERIMENT_DESIGN.json"),"variants":len(variants),"training_rows":len(train)},ensure_ascii=False))


if __name__=="__main__":
    main()
