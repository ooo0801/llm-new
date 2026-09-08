"""Offline verification of v2 response provenance and strict task outcomes."""
import argparse
import json
from pathlib import Path
from _bootstrap import project_path
from llm_integrity.stage1_r1 import digest,save_json
from stage2_preflight import task_check


def verify(folder,previous):
    manifest=json.loads((folder/"PREFLIGHT_FROZEN.json").read_text(encoding="utf-8"))
    report=json.loads((folder/"PREFLIGHT_REPORT.json").read_text(encoding="utf-8"))
    rows=[json.loads(s) for s in (folder/"preflight_responses.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows)==manifest["maximum_responses"]==48
    assert report["responses_sha256"]==digest(folder/"preflight_responses.jsonl")
    assert report["manifest_sha256"]==digest(folder/"PREFLIGHT_FROZEN.json")
    expected=[(p,manifest["response_seed_base"]+i*10+j) for i,p in enumerate(manifest["pool"]) for j in range(4)]
    for r,(p,seed) in zip(rows,expected,strict=True):
        assert r["prompt_id"]==p["id"] and r["response_seed"]==seed
        assert r["manifest_sha256"]==report["manifest_sha256"]
        assert r["task_correct"]==task_check(p,r["response"],r["truncated"])
    v1=json.loads((previous/"PREFLIGHT_REPORT.json").read_text(encoding="utf-8"))
    original={r["id"]:r for r in v1["summaries"]}
    comparison=[]
    selected=[]
    for p,s in zip(manifest["pool"],report["summaries"],strict=True):
        rr=[r for r in rows if r["prompt_id"]==p["id"]]
        correct=sum(r["task_correct"] for r in rr)
        assert s["correct"]==correct and s["total"]==4
        fits=all(r["input_tokens"]<=128 for r in rr)
        assert fits==s["fits_search"]
        if correct>=3 and fits:
            selected.append(p)
        comparison.append({"source_prompt_id":p["source_prompt_id"],"category":p["category"],
                           "v1_correct":original[p["source_prompt_id"]]["correct"],
                           "v2_correct":correct,"wording_changed":p["wording_changed"],"fits_search":fits})
    assert report["passing_count"]==len(selected)
    assert report["selected"]==selected[:8]
    assert (report["status"]=="PASS")== (len(selected)>=8)
    by_category={}
    for category in sorted({c["category"] for c in comparison}):
        group=[c for c in comparison if c["category"]==category]
        by_category[category]={"v1_correct":sum(c["v1_correct"] for c in group),
            "v2_correct":sum(c["v2_correct"] for c in group),"responses_per_version":4*len(group),
            "v2_passing_prompts":sum(c["v2_correct"]>=3 and c["fits_search"] for c in group)}
    result={"integrity":"PASS","rows":len(rows),"passing_prompts":len(selected),"gate":report["status"],
            "comparison":comparison,"by_category":by_category,
            "scope":"offline hash/order/decision checks; no model re-generation; v1/v2 use different seeds; exploratory"}
    save_json(folder/"LOCAL_VERIFICATION.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--folder",type=Path,required=True)
    parser.add_argument("--previous",type=Path,required=True)
    args=parser.parse_args()
    verify(args.folder,args.previous)
