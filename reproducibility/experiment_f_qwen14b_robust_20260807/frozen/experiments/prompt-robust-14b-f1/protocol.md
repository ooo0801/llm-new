# Experiment F1 Protocol: Multi-Seed Robust Selection and Independent Confirmation

Status: **FROZEN BEFORE ANY EXPERIMENT F1 MODEL ENDPOINT**

## Motivation and hypothesis

Experiment E technically completed its frozen joint confirmation but retained only 19/30 prompts and lost the summary category. The dominant failure was structured-pruning robustness (10/30 non-degraded), not task preservation or complete-block micro sensitivity. Experiment F1 treats all Experiment E outcomes as development evidence only. It does not edit Experiment E, its 30-row union, its seeds, or its thresholds.

F1 asks whether a wider, provenance-preserving portfolio can be reranked under several new development attack instances and then survive a completely independent joint confirmation.

**H-F1:** the frozen 30-prompt F1 selection will retain at least 23 prompts under the unchanged legacy gate on the independent confirmation split, and the retained set will contain all eight categories.

The strict 5/5 count remains a secondary descriptive endpoint. H-F1 is supported only when the technical endpoint gate, retained-count gate, and category-coverage gate all pass.

## Frozen candidate pool

- Source: the 64-row Experiment C portfolio produced before its development hard validation.
- Source evidence: `results/experiment_c_qwen14b_20260806/03_development/portfolio_candidates.jsonl`.
- The source contains 38 distinct original prompt IDs and all eight categories.
- Every row must preserve its original initial/optimized text, evaluator, task target, provenance, and prompt hashes.
- At most one candidate may be selected from each original source prompt.
- Exact optimized-text duplicates are forbidden.

The portfolio is prior development material. Neither Experiment C held-out results nor Experiment E outcomes are supplied to the F1 selection algorithm.

## Frozen model and endpoint definitions

- Model: `Qwen/Qwen2.5-14B-Instruct`.
- Revision: `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`.
- Dtype: BF16.
- Placement: one process, balanced layer sharding over GPUs 0/1/2, maximum 30 GiB per GPU, no CPU/disk offload.
- Task endpoints: deterministic greedy generation, no system prompt, unchanged frozen evaluator.
- Micro endpoint: complete 147-block Hutchinson score with four Rademacher probes.
- Macro endpoint: paired initial/optimized `macro_l2_raw` over the same five registered families.
- Family non-degeneration tolerance: optimized value no more than 1% below initial.

## Development split

- Micro seed: `2026080710`.
- Macro/bootstrap seed: `2026080714`.
- Three registered variants per family with attack seeds `2026080711`, `2026080712`, and `2026080713`.
- Unstructured pruning, NF4, Gaussian noise, and LoRA preserve the Experiment E family definitions while changing seeds.
- Structured pruning deliberately covers three preregistered configurations:
  1. random 5% FFN channels on a random half of layers;
  2. magnitude 5% FFN channels on all layers;
  3. random 5% attention heads on a random half of layers.
- The three development LoRA adapters are trained independently from the frozen isolated attack data.

For each candidate and family, compute the paired relative gain for every registered variant. The family robust gain is the median of those variant gains. A family is robustly non-degraded when its median relative gain is at least -1%.

A candidate is development-eligible only when both tasks pass, its optimized complete-block micro score is greater than its initial score, and its equal-family macro mean is positive. Eligible candidates are ranked lexicographically by:

1. structured-pruning robust non-degeneration;
2. number of robustly non-degraded families;
3. worst family median relative gain;
4. structured-pruning median relative gain;
5. overall macro relative gain;
6. micro relative gain;
7. prompt ID as a deterministic tie breaker.

Selection first takes three candidates per category in the fixed order code, instruction, knowledge, logic, reasoning, safety, structured, summary. It then fills to 30 from the global ranking. The one-candidate-per-source and unique-text rules apply throughout. If any category cannot supply three eligible unique-source candidates, or 30 total rows cannot be frozen, development is a preregistered No-Go and confirmation is not run.

## Independent confirmation split

- Micro seed: `2026080720`.
- Macro/bootstrap seed: `2026080723`.
- Two variants per family with seeds `2026080721` and `2026080722`.
- The confirmation LoRA adapters are trained only after the 30-row development selection is frozen.
- Confirmation structured pruning uses two held-out configurations:
  1. magnitude 5% FFN channels on a random half of layers;
  2. magnitude 5% attention heads on all layers.

The confirmation analysis uses the unchanged Experiment E per-prompt legacy gate:

1. both task endpoints pass;
2. optimized complete-block micro score is greater than initial;
3. optimized equal-weight five-family macro mean is greater than initial;
4. at least 3/5 family means are non-degraded within 1%.

Strict retention requires 5/5 families and remains descriptive. H-F1 requires at least 23 legacy-retained rows plus all eight categories in the retained subset.

## Leakage and repair rules

- Experiment E remains immutable and is not relabeled as F1 confirmation.
- Development endpoint results may select F1 prompts but may not change the frozen ranking, quotas, attacks, or confirmation gate.
- Confirmation results may not edit, replace, rerank, or repair the 30 frozen rows.
- A technical defect may be repaired only when candidates, seeds, attacks, evaluators, metrics, and thresholds are unchanged; failed logs remain preserved.
- A scientific F1 failure is archived as a No-Go. Any F2 construction must use a new protocol, new output paths, and new confirmation randomness.

## Output paths

- Protocol and frozen inputs: `experiments/prompt-robust-14b-f1/`.
- Raw results: `results/experiment_f_qwen14b_robust_20260807/`.
- Compact evidence: `reproducibility/experiment_f_qwen14b_robust_20260807/`.
- Human report: `to_human/experiment_f_qwen14b_robust_20260807/`.
