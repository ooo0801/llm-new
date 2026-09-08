import json
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from stage2_preflight_v2 import revised_pool


def test_revisions_preserve_contracts_and_operands():
    source=[json.loads(s) for s in (ROOT/"experiments/prompt-reconstruction-32b-h6/inputs/construction_prompts_k60.jsonl").read_text(encoding="utf-8").splitlines()]
    changes=json.loads((ROOT/"experiments/h9-stage2/prompt_revisions_v2.json").read_text(encoding="utf-8"))["replacements"]
    pool=revised_pool(source,changes)
    originals={r["id"]:r for r in source}
    assert len(pool)==12
    assert sum(r["wording_changed"] for r in pool)==9
    for row in pool:
        old=originals[row["source_prompt_id"]]
        for key in ["category","evaluator","expected_answer","expected_contains"]:
            assert row.get(key)==old.get(key)
        if row["category"]=="math":
            assert re.findall(r"\d+",row["prompt"])==re.findall(r"\d+",old["prompt"])
            assert row["expected_answer"] not in row["prompt"]
        if row["category"]=="instruction":
            assert row["prompt"]==old["prompt"]
        assert "original_prompt" in row
