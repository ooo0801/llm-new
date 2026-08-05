# Research Findings

## Research Question

Can prompts optimized and frozen on Qwen2.5-7B transfer their task validity and sensitivity to Qwen2.5-14B without re-optimization?

## Current Understanding

The project already contains a closed V6 prompt-construction pipeline and V1/V2 fingerprint evidence. Cross-scale transfer has not been tested. Experiment A changes model scale and target tokenization only: prompt text, model revision, seeds, evaluator definitions, sensitivity methods, family configurations, and retention gates are frozen before target endpoint results.

## Key Results

Pending. No 14B endpoint result existed when the protocol was frozen.

## Patterns and Insights

Pending.

## Lessons and Constraints

- The V6 evidence file contains 22 evaluated candidates, not 16 directly usable rows. The transferred set is the 16 rows with `optimization.accepted == true`.
- Transfer uses `optimization.optimized_prompt`; its paired within-model baseline is `optimization.initial_prompt`.
- Three GPUs enable layer sharding, but there is no NVLink and GPU memory is not disk capacity.
- The frozen V6 test uses two seeds for each of five families. Its quantization endpoint is NF4 only; adding INT8 would be an unregistered extension.
- Existing V1/V2/V6 evidence and raw results remain immutable and outside Experiment A output paths.
- The formal V6 task script overrides the general sampling config and uses deterministic greedy generation without a system prompt.
- The formal Hutchinson script uses complete blockwise parameter coverage; the representative-tensor options in another config path do not control this entry point.

## Open Questions

- Whether the Hutchinson proxy remains tractable with a sharded 14B model.
- Whether 50-step rank-16 LoRA training fits and completes on the three-GPU topology.
- How many prompts retain at least 3/5 non-degraded families, and how many retain all 5/5.

## Optimization Trajectory

No prompt optimization is permitted in Experiment A. The trajectory records staged feasibility and the final retained count only.
