# Research Findings

## Research Question

Can prompts optimized and frozen on Qwen2.5-7B transfer their task validity and sensitivity to Qwen2.5-14B without re-optimization?

## Current Understanding

The project already contains a closed V6 prompt-construction pipeline and V1/V2 fingerprint evidence. Cross-scale transfer has not been tested. Experiment A changes model scale and target tokenization only: prompt text, model revision, seeds, evaluator definitions, sensitivity methods, family configurations, and retention gates are frozen before target endpoint results.

## Key Results

- All 16 prompt pairs passed the technical gate on the fixed 14B revision with three-GPU layer sharding and no CPU/disk offload.
- Original-task validity was preserved for 12/16 pairs (27/32 individual endpoints passed).
- Micro sensitivity increased for 14/16 pairs; equal-weight five-family macro sensitivity increased for 13/16 pairs.
- Eight prompts met the frozen legacy retention gate (task preserved, positive micro, positive macro, and at least 3/5 non-degraded families).
- Four prompts met the strict 5/5 gate.
- H-A1 required at least 12 legacy-retained prompts. The observed 8/16 refutes H-A1 under the preregistered protocol.
- The fixed V6 hybrid-calibration continuity diagnostic accepted 14/16, but it cannot replace the explicit transfer gates and therefore does not change the 8/16 conclusion.

## Patterns and Insights

- Transfer is partial rather than uniform: most prompts retain positive micro and macro direction, while task preservation and per-family consistency reduce the final retained set.
- Family robustness is heterogeneous. Pass counts were finetuning 13/16, NF4 12/16, structured pruning 12/16, Gaussian noise 8/16, and unstructured pruning 8/16.
- Four strict-transfer prompts survived every registered family: `logic_0ece81476a78`, `safety_8c6e8a477653`, `safety_54d2229faad7`, and `instruction_9109f3338c12`.
- The eight legacy-retained prompts were `knowledge_9edb61565cee`, `logic_0ece81476a78`, `safety_020dda637cac`, `instruction_43e8f0520c0e`, `safety_8c6e8a477653`, `safety_1e228e4d44b9`, `safety_54d2229faad7`, and `instruction_9109f3338c12`.
- Task failures account for four pair failures (two translation pairs, one code pair, and one logic pair). Other failures arose from negative sensitivity direction or fewer than 3/5 family passes.

## Lessons and Constraints

- The V6 evidence file contains 22 evaluated candidates, not 16 directly usable rows. The transferred set is the 16 rows with `optimization.accepted == true`.
- Transfer uses `optimization.optimized_prompt`; its paired within-model baseline is `optimization.initial_prompt`.
- Three GPUs enable layer sharding, but there is no NVLink and GPU memory is not disk capacity.
- The frozen V6 test uses two seeds for each of five families. Its quantization endpoint is NF4 only; adding INT8 would be an unregistered extension.
- Existing V1/V2/V6 evidence and raw results remain immutable and outside Experiment A output paths.
- The formal V6 task script overrides the general sampling config and uses deterministic greedy generation without a system prompt.
- The formal Hutchinson script uses complete blockwise parameter coverage; the representative-tensor options in another config path do not control this entry point.

## Open Questions

- Whether the family-specific failures persist at other model scales; answering this requires a separately preregistered experiment.
- Whether task evaluators should treat semantically equivalent translations as valid; changing that rule is outside Experiment A and cannot alter this result retrospectively.
- Whether a new prompt-construction procedure trained across multiple scales can improve the 8/16 legacy-retention count.

## Experiment A Failure Attribution (2026-08-06)

The frozen result decomposes as a sequential gate waterfall of 16/16 technically valid, 12/16 task-preserved, 10/16 also micro-positive, 9/16 also macro-positive, 8/16 legacy-retained, and 4/16 strict-retained. This is descriptive reuse of existing outputs; it does not change H-A1.

Across prompts, independent gate failures overlap: task failed for 4, micro direction for 2, macro direction for 3, and the legacy family-count requirement for 6. The latter is the most frequent non-task bottleneck. Gaussian noise and unstructured pruning each passed only 8/16, compared with structured pruning and NF4 at 12/16 and finetuning at 13/16.

Two task failures (`translation_a77d5aded0e9` and `logic_3f35493753d6`) would otherwise satisfy all legacy sensitivity gates, while `knowledge_075a65e696ce` is blocked only by the 2/5 family count. These are causal-attribution candidates, not grounds for retrospective evaluator or gate changes.

## Optimization Trajectory

No prompt optimization is permitted in Experiment A. The trajectory records staged feasibility and the final retained count only.

## Experiment B Independent-Seed Survivor Replication (2026-08-06)

The frozen Experiment A survivor cohorts were repeated with independently derived perturbation and Hutchinson seeds. All 8 prompts preserved their original task; 7/8 were micro-positive and 7/8 were macro-positive. Six of the 8 legacy survivors again met the legacy gate, exactly supporting H-B2. Three of the 4 original strict survivors again met the strict 5/5 gate, exactly supporting H-B1.

The stable legacy core is `logic_0ece81476a78`, `instruction_43e8f0520c0e`, `safety_8c6e8a477653`, `safety_1e228e4d44b9`, `safety_54d2229faad7`, and `instruction_9109f3338c12`. The repeated strict set among all 8 is `instruction_43e8f0520c0e`, `safety_8c6e8a477653`, `safety_54d2229faad7`, and `instruction_9109f3338c12`; among the original strict-4 source cohort, 3/4 repeated strict.

This supports a seed-stable 7B-to-14B core but does not rescue H-A1: only 8/16 exact prompts passed the first transfer run. Experiment C therefore tests a separate claim—whether the frozen construction method can create a held-out-valid 14B-specific complement without using Experiment A/B outcomes for selection.

## Experiment C Development Freeze (2026-08-06)

The model-aware 14B construction produced 38 proxy-positive source prompts and 64 deduplicated development candidates. Frozen hard validation completed 128 task endpoints, 102 complete-coverage micro scores, and 1,020 macro observations across ten variants from five perturbation families. Thirty-four candidates met the strict development selection rule, exceeding the preregistered minimum of four. No held-out test data was used for construction, ranking, calibration, or this freeze; the 34 are therefore a development-frozen set awaiting independent held-out confirmation rather than final accepted prompts.
