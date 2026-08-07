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

## Experiment C Independent Held-out Result (2026-08-06)

All 34 development-frozen candidates completed independent held-out validation. Both task endpoints passed for every candidate; 68/68 complete-block micro scores and 680/680 observations across the five registered perturbation families were produced. Twenty-nine candidates passed the frozen held-out acceptance rule, versus the preregistered minimum of four, so H-C1 is supported. The development-to-held-out retention rate is 29/34 (85.3%).

The accepted complement spans eight categories: 3 code, 5 instruction, 3 knowledge, 3 logic, 4 reasoning, 4 safety, 5 structured, and 2 summary prompts. Held-out outcomes were used only for final confirmation; they did not edit, rerank, or repair the development-frozen candidates or thresholds.

## Combined Qwen2.5-14B Fingerprint (2026-08-06)

The final provenance-preserving union contains 30 prompts: four independently repeated cross-model strict-core prompts from Experiments A and B, plus 26 non-overlapping 14B-specific prompts confirmed by Experiment C. Three Experiment C accepted rows (`instruction_43e8f0520c0e`, `safety_8c6e8a477653`, and `instruction_9109f3338c12`) overlapped the strict core and were deduplicated by source identity or exact prompt hash.

This union preserves the evidence scope of its components. It is not evidence that all 30 prompts were jointly rerun under an additional common attack set; making that stronger claim would require a separately preregistered experiment.

## Experiment E Joint Confirmation (Preregistered 2026-08-07)

Experiment E addresses the remaining claim gap without reconstructing or reranking prompts. The exact 30-row union is frozen as paired initial/optimized text and will be evaluated together with a new complete-micro seed and two new instances of every registered perturbation family. The primary gate requires at least 23 legacy-retained prompts and representation of all eight categories; the strict 5/5 subset remains a secondary endpoint. No Experiment E result may change the union, evaluators, seeds, family definitions, or thresholds.

## Experiment E Final Joint Result (2026-08-07)

Experiment E completed every preregistered endpoint without technical failure: 60/60 task endpoints, 60/60 complete-block micro endpoints, and 600/600 macro observations across two fixed variants of each of the five registered perturbation families. All 30 prompt pairs preserved their task and all 30 optimized prompts had positive micro sensitivity. Nineteen prompts had positive equal-weight macro sensitivity and satisfied the full legacy gate.

The observed 19/30 retained count is below the frozen requirement of 23/30. The retained prompts span code, instruction, knowledge, logic, reasoning, safety, and structured categories, but no summary prompt survived; the required eight-category coverage therefore also failed. H-E1 is refuted on both primary conditions. Three prompts met the secondary strict 5/5 gate.

The principal bottleneck was joint macro robustness rather than task validity or local gradient sensitivity. Family non-degeneration counts were 27/30 for unstructured pruning, 25/30 for Gaussian noise, 25/30 for finetuning, 23/30 for NF4 quantization, and 10/30 for structured pruning. The 19-row legacy-retained output is an operational subset under this single frozen seed set, not a repaired replacement for the preregistered 30-row union. No candidate, seed, evaluator, threshold, or attack definition was changed after observing results.

## Experiment F1 Multi-Seed Robust Selection (Preregistered 2026-08-07)

F1 preserves Experiment E as a refuted confirmatory result and returns to the full 64-row pre-development Experiment C portfolio. It uses three new development variants per family, including three distinct structured-pruning configurations, to rank candidates by structured-pruning non-degeneration, robust family count, worst-family median gain, and paired macro/micro gains. The frozen selection requires three unique-source candidates per category and 30 total rows. Two new variants per family and a new complete-micro seed are isolated for confirmation. H-F1 retains the unchanged primary gate of at least 23 legacy-retained prompts with all eight categories represented; confirmation outcomes cannot repair the frozen selection.

## Experiment F1 Development No-Go (2026-08-07)

F1 completed its entire development endpoint matrix without technical failure: 128 task records, 102 unique complete-block micro scores expanded to 128 pair endpoints, and 1,530 unique macro observations expanded to 1,920 pair observations across fifteen registered variants. Of 64 candidates, 63 preserved both tasks, 60 were micro-positive, 39 were macro-positive, 37 met the development eligibility rule, 42 were non-degraded under structured pruning, and 36 met the descriptive legacy rule.

The frozen quota-and-unique-source selector nevertheless produced only 29 of the required 30 rows. It found only two eligible unique-source logic candidates and one eligible unique-source summary candidate, below the frozen quota of three in each category. The provisional 29-row distribution was code 5, instruction 5, knowledge 3, logic 2, reasoning 4, safety 6, structured 3, and summary 1. This is a scientific development No-Go rather than a technical failure.

The quota was not lowered and no post-hoc candidate was inserted. Consequently the independent F1 confirmation was not run, and G1 stable-component/MCC extraction plus H1 stratified-MMD detection were not eligible to run under their conditional protocols. A future attempt would require a separately preregistered F2 that expands logic and summary construction before observing its new confirmation set.

## Experiment F2 Targeted Expansion (Preregistered 2026-08-07)

F2 acts only on the localized F1 source-diversity deficit. It adds 16 new clean logic sources and 16 new clean summary sources. Logic targets are exact and cover several reasoning forms; summary targets require a 25-character bound plus two content-specific keywords, preventing the weak one-keyword evaluator from accepting semantically corrupted instructions. The new sources are optimized with the frozen Experiment C construction calibration and training attacks, then all valid new candidates are combined with the untouched F1 pool and rescored under new development attacks.

The development selector, 30-row size, three-per-category quota, unique-source rule, structured-first ranking, 23/30 confirmation requirement, and eight-category coverage requirement are unchanged. Independent F2 confirmation uses disjoint seeds and held-out structured-pruning configurations. G2 MCC12 construction and H2 prompt-stratified MMD verification are strictly conditional on F2 and G2 passing, respectively.

### F2 Construction Smoke Observation

The two-source construction smoke completed technically and accepted both the logic and summary edits with positive proxy gain and task preservation. The logic edit still generated the exact target `甲`. The summary edit generated `固定随机种子后实验重复运行三次比对结果。`, satisfying the frozen 25-character and two-keyword rule.

The summary optimized prompt nevertheless changed `不超过25个汉字` to the linguistically unnatural `情形25个汉字`. Early formal logic edits likewise introduced unnatural punctuation or wording while preserving the exact answer. This demonstrates that the frozen evaluators guard generated-output task behavior but do not guarantee natural or semantically invariant prompt wording. The observation is retained as a limitation; F2 does not add a post-hoc naturalness gate, repair a candidate, or change the frozen construction and selection rules.

Among the first three formal logic sources, two were accepted and one failed the initial frozen task guard: the model generated `绿盒` while the preregistered exact target was `绿`. This rejected source remains rejected; F2 does not relax the evaluator to substring matching or edit the expected answer after observing the endpoint.

### F2 Targeted Construction Final Result

The frozen targeted construction completed all 32 sources technically. Twenty-eight sources produced accepted positive-proxy edits and four were rejected. The deterministic portfolio rule produced 52 candidates from 28 sources: 13 distinct new logic sources and 15 distinct new summary sources. Both exceed the preregistered minimum of three, so the construction gate passed.

Combining the 52 new candidates with the untouched 64-row F1 base yielded 116 development candidates from 66 distinct source prompts. The category counts are 7 code, 8 instruction, 7 knowledge, 29 logic, 8 reasoning, 12 safety, 10 structured, and 35 summary. No confirmation or test endpoint contributed to construction or pool assembly.

Development task validation completed 232/232 endpoints, with 231 passes. The sole failure was the optimized endpoint of old base candidate `safety_54d2229faad7__portfolio_01_r1_c1`: its corrupted text elicited advice about a flower-petal-style website rather than a refusal and omitted the frozen required marker `不能`. The candidate remains failed without replacement or evaluator repair.

### F2 Development Selection Result

F2 development completed all preregistered endpoints without a technical error: 232 task records, 182 unique complete-block micro scores expanded to 232 pair endpoints, and 2,730 unique macro observations expanded to 3,480 pair observations over fifteen registered attack variants. Of the 116 candidates, 51 met the frozen development eligibility rule.

The quota-and-unique-source selector froze exactly 30 candidates spanning all eight categories: 3 code, 4 instruction, 3 knowledge, 8 logic, 3 reasoning, 3 safety, 3 structured, and 3 summary prompts. The development gate therefore passed. The frozen 30-row set, rather than any confirmation outcome, defines the candidates entering independent confirmation.

Independent confirmation task validation passed all 60 initial/optimized endpoints. The two confirmation adapters also completed before an execution interruption during complete-block micro scoring. Seven unique micro records were durably written; after connectivity returned, the recovery entry point verified the development gate, frozen-set size, confirmation task summary, and adapter registry, then resumed by prompt identity without recomputing the completed records or changing scientific inputs.

The resumed confirmation micro stage subsequently completed all 60 unique prompt endpoints and expanded them to the expected 60 paired endpoints. The unchanged pipeline then advanced to the frozen two-variant-per-family confirmation macro stage; final retention is not evaluated until all 600 registered macro observations are complete.

### F2 Independent Confirmation Final Result

F2 completed every independent confirmation endpoint without technical failure: 60/60 deterministic task endpoints, 60/60 complete-block micro endpoints, and 600/600 finite macro observations across two held-out variants of each of the five registered attack families. All 30 candidates preserved their tasks and were micro-positive; 20 had positive equal-weight macro sensitivity.

Nineteen of the frozen 30 candidates satisfied the full legacy retention rule, below the preregistered requirement of 23. The retained rows did cover all eight categories (1 code, 4 instruction, 2 knowledge, 4 logic, 3 reasoning, 2 safety, 1 structured, and 2 summary), so category coverage passed while the primary retained-count gate failed. Ten prompts met the secondary strict 5/5 rule.

Family non-degeneration counts were 25/30 for finetuning, 24/30 for Gaussian noise, 25/30 for quantization, 21/30 for structured pruning, and 25/30 for unstructured pruning. Targeted expansion therefore fixed the F1 development feasibility problem but did not improve independent joint retention beyond the 19/30 result previously observed in Experiment E. H-F2 is refuted without repair. Because the F2 parent gate failed, G2 stable-component/MCC12 construction and H2 prompt-stratified MMD verification were not run.

The terminal evidence archive contains 50 checksum-covered files, including frozen inputs and configs, construction/development/confirmation reports and endpoints, adapter registries and training reports, and the pipeline log. LoRA tensor weights are intentionally excluded. The archive records `confirmation_no_go` and preserves the boundary that no candidate, threshold, seed, attack, evaluator, or downstream condition was changed after observing confirmation.

## Experiment G3/H3 Recovery Fingerprint (Preregistered 2026-08-08)

G3 preserves the F2 19/30 confirmation No-Go and treats its exact 19 legacy-retained rows as a new byte-frozen recovery pool. The source contains 19 unique prompt IDs and texts, covers all eight categories, and is locked by SHA-256 `6ecad275352bd532dd944154b569c6935ea849d5538ca7b0da2daf7ed7aa31e1`. No F2 failure is reclassified.

The only intentional protocol change from conditional G2 is candidate-pool feasibility: G3 requires the exact 19 rows rather than at least 23, while still selecting 12 prompts. Component thresholds, two-repeat Jaccard, 14-family build/audit calibration, saturation and audit-novelty gates, global-unweighted MCC objective and numerical audits remain unchanged. H3 is conditional on G3 and freezes new reference, target and attack seeds before any endpoint; it retains the intact-correct, 9/11 modified-detection and five-family-coverage gates.

Deterministic protocol preparation produced 19 candidate rows, 280 build prompts, 56 audit prompts and 12 registered H3 states. The nearest calibration/candidate trigram Jaccard was 0.0656 versus the frozen 0.72 ceiling. Candidate, build, audit and attack canonical hashes were frozen before model activation extraction.

### G3 Stable-Component and MCC12 Result

G3 passed every preregistered construction gate. Two-repeat activation profiles over all 280 build, 56 audit and 19 candidate prompts had minimum Jaccard 1.0, and profiling left the reference logits bit-identical. The empirical global universe contained 50,835 components. The final three cumulative batches added 0.37%, 0.57% and 0.40%, all below the frozen 2% ceiling; audit-only novelty was 2.58% versus 8% allowed.

Deterministic global-unweighted MCC selected 12 unique prompts spanning all eight categories. The selected set covered 16,508 global components, or 32.47% of the complete empirical universe and 95.57% of the 17,273 components reachable by the 19-prompt candidate pool. Immediate recomputation reproduced the selected IDs and trace, and numerical factorization error was zero. H-G3 is supported, making the independently seeded H3 verification eligible to run.
