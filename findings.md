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

### H3 Independent Stratified-MMD Result

H3 completed all 12 registered state verifications without technical failure. The primary prompt-stratified MMD test detected all 11 modified states (11/11), including at least one instance from each of unstructured pruning, structured pruning, Gaussian perturbation, quantization, and finetuning. Every modified-state decision had permutation p-value 0.001. Modified-state recall was therefore 1.0, with exact 95% interval [0.7151, 1.0].

The intact model was also flagged by the preregistered primary test: stratified statistic 0.12206106, p=0.001, effect size 1.6843. This violates the mandatory intact-correct gate, leaving 11/12 state decisions correct (accuracy 0.9167; exact 95% interval [0.6152, 0.9979]) and refuting H-H3 despite perfect modified-state detection and family coverage.

The frozen pooled-MMD ablation did not flag intact (statistic -0.008305, p=1.0), but it was explicitly secondary and cannot replace the primary statistic after outcomes are known. No threshold, statistic, seed, state, or prompt was changed. The final evidence is therefore: the G3 procedure successfully constructs a deterministic MCC12 from the frozen F2-retained pool, while this H3 prompt-stratified detector is not validated because it produced an intact false positive.

## H3 False-Positive Attribution and H4 Remediation (2026-08-08)

Exploratory reanalysis did not support a simple implementation-repair explanation. Refitting feature scaling symmetrically on the pooled intact samples left the primary statistic significant (p=0.002), and using whole generation-seed batches as the MMD exchangeable unit also left p=0.001. Surface, semantic and task feature blocks were each significant under pooled-symmetric scaling. Thus neither reference-only standardization nor independent within-prompt permutations alone explains the H3 decision.

A fresh three-GPU process then replayed all ten H3 reference seeds and all ten H3 intact-target seeds. Every one of the 240 responses matched the archived response byte-for-byte (240/240), ruling out weight drift, adapter leakage, GPU sharding changes and nondeterministic replay as the source. The remaining confound is design-level: H3 compared two independent banks of only ten stochastic generation seeds, so genuine sampling variation was inseparable from a model-state effect. The detector found a difference between the realized response banks, but that difference was not caused by a modified model.

H4 is a separately preregistered correction, not a reinterpretation of H3. It reuses the exact frozen G3 MCC12 but generates a new reference bank and every target with the same ten seeds. Its primary exact paired test flips all 12 prompt differences from one seed together and enumerates all 1,024 sign patterns. Three independent intact reloads guard specificity; eleven newly seeded modified states retain the 9/11 and five-family sensitivity gates. Legacy stratified and pooled MMD remain secondary ablations.

### H4 Final Paired Verification Result

H4 completed all 14 registered states without a technical error. The common-random-number design eliminated the H3 intact false positive: all three independently reloaded intact states produced a paired statistic of exactly 0, p=1.0, and remained unflagged. This supports the attribution that H3's false positive came from comparing two small independent stochastic seed banks rather than from model-weight drift, adapter leakage, GPU sharding, or irreproducible generation.

Sensitivity did not satisfy the frozen joint gate. The paired test detected both unstructured-pruning states, both structured-pruning states, both quantized states and both LoRA states; each had exact sign-flip p=0.001953125. It detected none of the three Gaussian perturbations, whose p-values were 0.46875, 0.6328125 and 0.576171875. Modified-state detection was therefore 8/11, below the required 9/11, and Gaussian family coverage was absent. H-H4 is refuted even though the specificity defect was corrected.

The result exposes a real tradeoff rather than an implementation failure: pairing removes nuisance sampling variation and restores intact specificity, but the frozen signed seed-block statistic lacks power for the registered subtle Gaussian changes. H3 remains a No-Go for false positives and H4 is an overall No-Go for sensitivity. No threshold, statistic, attack magnitude, seed, prompt or family rule was changed after observing endpoints. The 27-file, checksum-covered H4 archive preserves the complete fixed-instance evidence while excluding LoRA weight tensors.

## H5 Gaussian-Sensitivity Remediation (Preregistered 2026-08-08)

Exploratory H4 response-level attribution revealed that the three Gaussian states were not output-identical to reference: they changed 11, 15 and 18 of 120 paired responses and affected 7, 10 and 10 of ten generation-seed blocks. All three intact reloads changed 0/360 responses and 0/30 seed blocks. Thus H4's Gaussian failure arose from cancellation in the signed mean-feature statistic, not from an absence of observable output changes.

H5 freezes a non-directional paired detector before new endpoints. It compares each target response byte-for-byte with the response generated by the same prompt and new common seed, then treats a whole twelve-prompt generation seed as one block. The one-sided exact binomial null tolerates an intact block-mismatch probability of 0.10 and uses alpha 0.05/3 for the three intact reloads; with ten blocks, four positive blocks are required to flag a state. This provides a three-block margin relative to the weakest H4 Gaussian development observation while allowing isolated reproducibility noise.

Independent H5 verification uses new reference/target seeds and fourteen newly instantiated states. The frozen gate requires 3/3 intact specificity, at least 9/11 modified detections, all five modified families, and specifically 3/3 Gaussian detections. H4 data chose the method but cannot count toward H5 confirmation. Prompt-stratified and pooled MMD are secondary only, and no H5 outcome may alter the response comparator, null rate, alpha, four-block boundary, attack magnitude or family gates.

### H5 Final Independent Result

H5 completed all fourteen registered states without a technical error and supported H-H5. All three independently reloaded intact states had zero byte-different responses, zero positive seed blocks, statistic 0 and p=1.0. All eleven modified states were detected, producing 11/11 modified recall and 14/14 correct state decisions. The exact 95% interval for modified recall is [0.7151, 1.0], and the interval for total state accuracy is [0.7684, 1.0].

The dedicated Gaussian gate passed 3/3 on wholly new perturbation seeds. Each Gaussian state changed nine of ten seed blocks; response mismatch counts were 17/120 for FFN noise 0.00025, 16/120 for FFN noise 0.0005 and 21/120 for all-layer noise 0.001. Their one-sided exact block-binomial p-values were 9.1e-09. Both unstructured-pruning, both structured-pruning, both quantization and both LoRA instances changed all ten seed blocks and were also detected, so five-family coverage passed.

The resulting evidence chain is internally consistent: H3 was over-sensitive because independent stochastic banks confounded model state; H4 common-seed pairing restored intact specificity but its directional feature mean cancelled subtle Gaussian effects; H5 retained common-seed specificity while using non-directional response-block evidence, independently recovering Gaussian and overall sensitivity. The validated claim remains bounded to the fixed Qwen2.5-14B revision, runtime, MCC12, generation configuration, seeds and registered state instances. It does not establish a population detection rate or portability across hardware/software stacks.

## H6 Qwen2.5-32B Reconstruction Direction (Preregistered 2026-08-12)

H5 remains immutable and complete for its fixed 14B scope. H6 broadens the project to a fixed revision of `Qwen/Qwen2.5-32B-Instruct`, but first targets only model-specific sensitive-prompt reconstruction, independent confirmation, stable-component extraction and MCC12 construction. Integrity verification is deferred to a later protocol.

H6 reuses the byte-frozen 60 clean task texts as common task inputs so that scale changes are not confounded with a new source-task distribution. It assigns new H6 prompt IDs and retokenizes all text with the 32B tokenizer. No 14B gradients, calibration scales, responses, activations, component IDs, LoRA adapters or attack realizations are reused.

The 3x32 GiB host is expected to fit 32B BF16 inference by layer sharding, but full blockwise Hutchinson backward remains an explicit engineering risk. H6 therefore makes complete parameter coverage a pre-endpoint engineering gate and forbids CPU/disk offload or a quantized reference as a hidden repair. If the gradient gate fails for memory, the same frozen protocol can move to a larger GPU host without changing scientific inputs.

H6-P requires at least 30 development-frozen prompts and at least 19 independently confirmed prompts covering code, instruction, knowledge, logic, reasoning, safety, structured and summary. H6-G is conditional on H6-P and freezes the prior component thresholds as a direct-transfer test, while rebuilding the empirical universe and all component identities on 32B. The final MCC12 must also cover all eight core categories.

### H6 Operational Pause (2026-08-12)

The preregistration and recovery-capable runner were committed and pushed before any 32B model endpoint. The obsolete 14B model cache was removed after verifying that H5 code and non-weight evidence remained preserved. The frozen 32B revision download then reached approximately 17 GB through the revision-matching mirror before the user requested a temporary server shutdown.

The downloader and its automatic continuation process were terminated cleanly at 12:11 CST. No engineering-gate or scientific endpoint had started, all three GPUs were idle, and the partial Hugging Face cache was retained for resumable download. The last known server worktree was clean on commit `615a1d7ef81e9f2b12906e189e7280d24753881d`, branch `experiment/h6-qwen2.5-32b-reconstruction`, with the GitHub SSH remote configured. Recovery must rediscover the possibly changed SSH endpoint, validate project/GitHub access, and inspect retained cache before resuming the same fixed revision.

The same server returned at 14:37 CST with the three GPUs idle, 217 GB free, and the 17 GB partial cache intact. GitHub SSH authentication succeeded; repository transfer over port 22 later timed out, while GitHub's SSH-over-443 endpoint authenticated and transferred the exact branch successfully. The server fast-forwarded cleanly to `c1d9275d170e7e54897fa8f89613be80f6f56b0a`, H6 directed tests passed, and a push dry-run reported the branch up to date. The frozen-revision download resumed as PID 1872 with the single automatic continuation chain PID 1873; within the first two minutes the cache grew to 19 GB.

### H6 Download and Engineering Gate (2026-08-12)

The resumed download completed at 15:33 CST and validated the exact frozen revision, all 17 expected safetensor shards, 65,527,841,856 weight bytes, 64 transformer layers, and the hashed tokenizer/config metadata. The environment record confirmed Transformers 4.57.6, PyTorch 2.12.1+cu126, BF16 balanced placement across all three GPUs, eager attention, and no CPU or disk offload.

The preregistered engineering gate passed every stage. Two same-seed generations were byte-identical (`连接正常`); activation hooks exposed 10,613 components while leaving logits exactly unchanged; and the blockwise Hutchinson micro pass covered all 195 groups and all 32,763,876,352 parameters. Peak allocated memory was 28.87, 24.43, and 29.84 GB on GPUs 0, 1, and 2 respectively, so the fixed three-GPU host passed without quantization, offload, or scientific repair. The formal H6 pipeline then became eligible and started the registered training/development adapters before inner calibration and prompt construction.

The first registered training LoRA completed all 50 steps with finite final loss 0.3999 and was durably registered. Loading the second LoRA in the same long-lived Python process then OOMed because the prior Trainer/model references retained approximately 31.3 GiB on GPU 0; after process exit all GPU memory returned to baseline. This was an inter-variant lifecycle failure, not a model-fit or scientific-gate failure. Commit `b09dd81` isolates each of the six frozen LoRA variants in its own Python process and merges a reused adapter's persisted training report. It does not alter variant IDs, seeds, hyperparameters, data, model placement, or gates. Recovery reused the completed first adapter and the independently spawned second process loaded all 17 shards and resumed training normally.

The isolated second training adapter subsequently completed 50/50 steps with finite final loss 0.3677 and was registered beside the first. Its process exited before the first development adapter loaded all 17 shards in a fresh process, validating the repair across an actual variant boundary. Development adapter 1 then advanced through at least 20/50 steps with stable 21-23 GB per-GPU allocation; inner calibration has not yet started.

Both development adapters ultimately completed and registered. The inner calibration then completed 36/36 valid micro records and 84/84 valid macro records. Macro calibration used exact sequential variant/reference VJPs after the original dual-resident 32B path exceeded the aggregate 96 GiB GPU memory; the resulting family-specific scales and gradient clips were finite, and the combined calibration artifact was frozen before construction.

The first construction smoke produced no prompt endpoint. Its initializer rejected an internal mismatch: the H6 train manifest freezes two variants per family, while the inherited setting requested one search variant plus two disjoint anchors. Before any optimized prompt was observed, the configuration was corrected to the only disjoint two-way partition, one search plus one anchor per family. The 32B optimizer also adopts the already validated exact sequential-VJP execution for macro gradients and sequential reference-logit caching for reranking. These changes affect feasibility and model lifetime only; no source text, attack instance, seed, objective, threshold, task rule, development/confirmation split, or H6-P/H6-G gate changes.

The corrected one-prompt construction smoke subsequently passed end to end. It completed all three optimizer rounds without OOM or a model-lifecycle error, committed two rounds, produced one discrete edit, preserved the task guard, and reported positive proxy gain 14.1729. The smoke result is an engineering endpoint only and is excluded from the formal 60-row construction output. Formal candidate generation then started from an empty, separately persisted result file on commit `8033a0f`.

Formal generation reached its first 10/60 checkpoint with nine accepted discrete edits, one normal proxy-search rejection, and no technical failure. Accepted rows already include seven of the eight required core categories—code, instruction, knowledge, logic, reasoning, safety, and structured—while category coverage and the development gate remain unevaluated until all 60 construction rows and independent development endpoints are complete.

At 20/60, formal generation had produced seventeen accepted edits and three normal proxy-search rejections with zero technical failures. The accepted prefix now includes all eight core categories after the first summary acceptance. This remains an interim construction observation only: development still requires its frozen independent endpoints, at least 30 accepted rows, and complete category coverage before confirmation can begin.

Formal generation later persisted 43/60 valid unique prompt rows: 35 accepted edits, eight normal rejections, and zero technical failures. The optimizer and its post-construction guard then both disappeared without a logged Python, CUDA, or scientific-gate exception; the host itself had not rebooted, and the last optimizer log ended during a model reload after row 43 was already durable. The incomplete prompt had emitted no row and the formal summary did not yet exist.

Recovery validated all 43 JSONL rows, their unique prompt IDs, and the absence of technical failures before starting the same optimizer with `--resume`. Variant selection is deterministic from the frozen seed and prompt ID, so completed IDs are skipped without advancing shared sampling state. The resumed optimizer runs as PID 1537 and a separately detached guard as PID 1539; no prompt, attack, seed, calibration artifact, objective, threshold, evaluator, or H6 gate changed.

That detached recovery was also externally terminated while processing the still-uncommitted row 44. The durable file remained byte-safe at 43 rows, the log again contained no exception, and system evidence showed no cgroup memory event, kernel OOM, GPU Xid, host reboot, or exhausted storage. A dedicated recovery entry point now validates the 43-row checkpoint, runs the unchanged optimizer, asserts the frozen 60/60 technical summary, and then sources the original runner beginning at its first post-optimizer command. It is hosted in the server's `tmux` session `h6_pipeline` (pipeline PID 2122, optimizer PID 2128), records a timestamped exit code, and remained active across a fresh SSH session with normal three-GPU allocation. This is process supervision only; all scientific inputs and gates remain frozen.

The persistent recovery then completed the previously interrupted row 44. Its summary-category prompt was accepted with one committed edit and proxy-objective gain 1.3760, bringing the durable formal output to 44/60 with 36 accepted, eight normal rejections, and zero technical failures. The same `tmux` pipeline immediately began row 45, establishing that the supervision repair crossed the exact point of both prior external terminations.

The persistent pipeline remained active for 39 minutes and advanced to 46/60. Row 45 was an accepted code prompt with three committed rounds and proxy gain 1.2625; row 46 was a normal long-context rejection with no committed edit. The cumulative result is 37 accepted, nine normal rejections, and zero technical failures, with row 47 running under the unchanged protocol.

Row 47 was subsequently accepted in the logic category with one committed edit and proxy gain 2.6005. After nearly one hour of persistent execution, formal generation stood at 47/60 with 38 accepted, nine normal rejections, zero technical failures, and accepted examples in every required core category; row 48 was running.

Row 48 was accepted in the safety category with two committed rounds and proxy gain 1.9778. Formal generation reached 48/60 with 39 accepted, nine normal rejections, and zero technical failures. The persistent optimizer remained live through a normal low-memory model-switch interval and continued toward row 49.

Row 49 was accepted in the structured category with two committed rounds and proxy gain 6.9736. Formal generation reached 49/60 with 40 accepted, nine normal rejections, and zero technical failures after 99 minutes of persistent execution; row 50 was running.

At the 50/60 natural milestone, the translation row was accepted with two committed rounds and proxy gain 3.2737. The formal prefix contained 41 accepted edits, nine normal rejections, zero technical failures, and accepted examples in every required core category. The persistent pipeline continued directly to row 51; this remains construction evidence rather than an early development-gate decision.

Rows 51 and 52 advanced the durable output to 52/60. The knowledge row was accepted with two committed edits and proxy gain 12.9657, while the math row was a normal no-edit rejection. Cumulative counts were 42 accepted, ten normal rejections, and zero technical failures after 139 minutes under `tmux`; row 53 was running.

Row 53 was accepted in the reasoning category with two committed rounds and proxy gain 9.0324. Formal generation reached 53/60 with 43 accepted, ten normal rejections, and zero technical failures after 159 minutes of persistent execution; row 54 was running.

Row 54 was accepted in the instruction category with two committed edits and proxy gain 45.3375. Formal generation reached 54/60 with 44 accepted, ten normal rejections, and zero technical failures after three hours under `tmux`; row 55 was running.

Row 55 was accepted in the summary category with three committed edits and proxy gain 1.8917. Formal generation reached 55/60 with 45 accepted, ten normal rejections, and zero technical failures after 199 minutes under `tmux`; five rows remained before the complete construction summary and automatic development handoff.

Before a planned user disconnect, row 56 code was accepted with two committed edits and proxy gain 25.1984, while row 57 long-context was a normal no-edit rejection. The durable checkpoint is therefore 57/60 with 46 accepted, eleven normal rejections, and zero technical failures. Ordinary VS Code or SSH disconnection does not affect the active `tmux` pipeline. For a full server shutdown or instance restart, commit `af96ae6` generalizes the recovery entry point: it accepts any 1-60-row checkpoint only after verifying valid JSON, unique IDs, exact prefix identity against the frozen 60-prompt input, and absence of technical failures, then invokes the unchanged `--resume` optimizer and original downstream runner. At most the one in-progress, not-yet-written row can be recomputed after a shutdown; durable rows are skipped by prompt ID.

After the server became available again, the former `tmux` session and experiment process were absent while the repository remained clean on `af96ae6`, all GPUs were idle, and the 57-row checkpoint was intact. The generalized recovery entry point verified the exact 57-row frozen-input prefix, 46 accepted rows, and zero technical failures, then restarted the unchanged pipeline in `tmux` as pipeline PID 1356 and optimizer PID 1362. The model reloaded normally across all three GPUs and resumed row 58; no completed row or scientific setting changed.

The restarted pipeline durably advanced to 59/60 without a technical failure. Row 58 was an accepted logic prompt with two committed rounds, while row 59 safety was a normal initial-task-validation rejection. The cumulative construction prefix therefore contains 47 accepted edits and twelve normal rejections, and the same optimizer remains active on the final construction row. This is still construction-only evidence: no development or confirmation gate has been evaluated.

Formal construction then completed all 60 frozen inputs. Row 60 structured was accepted with one committed round, yielding 48 accepted edits, twelve normal rejections, 97 total committed rounds, and zero technical failures. The summary records 60 requested, 60 results, 60 technically complete, and technical/scientific construction pass; all eight core categories are represented in the accepted construction set. These counts make development eligible but do not themselves satisfy the independent H6-P gate.

Immediately after writing the detailed construction summary, the recovery wrapper exited with `DEV: unbound variable` before creating any development file. This was a post-construction shell-context omission: the wrapper sourced the frozen runner beginning at its first post-optimizer command but had not recreated the runner's `DEV`, confirmation, MCC, adapter, and LoRA helper context. Commit `e1facba` restores those exact runner declarations and adds a regression test. It changes no scientific input or gate; the validated 60-row construction output is the restart point.

The clean server fast-forwarded to `0aa98a4`, passed the eight H6 protocol tests, and restarted from the validated 60/60 checkpoint. The repaired wrapper crossed the prior exit point, built 95 proxy-accepted portfolio sources into 190 initial/optimized task-validation pairs representing 143 unique prompt strings, and entered independent development task validation under `tmux`. The first eight task endpoints passed, with normal three-GPU allocation and no logged error; development selection remains unevaluated until every frozen endpoint completes.

Development task validation completed all 190 initial/optimized endpoints: 185 passed and five failed their frozen task evaluators. The pipeline then automatically entered four-probe complete-block micro scoring for the 143 deduplicated unique prompt strings, retaining coverage of all 195 parameter groups per prompt. This task-pass rate is descriptive only; eligibility is determined later by the frozen joint task, micro, and macro rules.

Complete-block development micro scoring crossed its halfway point with 72/143 unique prompt records durably written. The same process remained live under `tmux`, continued through all 195 parameter groups for each completed prompt, and showed normal three-GPU allocation with no new exception. Macro endpoints and the joint development gate remain unobserved.

At the user's planned server-disconnect pause, complete-block micro scoring had durably reached 75/143 unique prompts. All 75 JSONL rows parsed, had unique prompt IDs, and contained no technical failure. The active prompt had not yet written a row; an intentional `Ctrl-C` stopped only the `h6_pipeline` session, released all three GPUs to 1 MiB, and produced wrapper exit code 0. Recovery must rediscover the endpoint, verify the clean `0aa98a4` server commit and 75-row prefix, then restart the same recovery entry point so `--resume` skips completed micro rows. No macro or development-gate endpoint has been observed.

After the server returned, the same endpoint exposed the clean `0aa98a4` repository, three idle GPUs, 71 GB free disk, working GitHub SSH-over-443 read access, and the exact valid 75/143 frozen micro prefix. The recovery entry point restarted in `tmux`, replayed only already-complete upstream summaries, explicitly skipped micro IDs `unique::0000` through `unique::0074`, and resumed full 195-group scoring at `unique::0075`. A fresh SSH session confirmed the new pipeline and scorer remained active with normal GPU allocation; no scientific setting changed.

The resumed complete-block micro stage reached 100/143 durably written unique prompts. The scorer remained live with 100% CPU activity, normal three-GPU allocation, and no new exception while continuing the full 195 parameter groups for prompt 101. Macro scoring and the joint development gate remain unobserved.

Development micro scoring subsequently completed all 143/143 unique prompts with full 195-group, four-probe coverage and handed off automatically to the ten frozen development attack variants. Macro sampling sequentially entered variant 8/10 after processing both finetuning, both Gaussian, both quantization, and the first structured-pruning variant. The macro process remained live with normal three-GPU allocation and no new exception. No joint eligibility or development gate has yet been emitted.

The independent development stage completed and passed its frozen gate. All ten registered attack variants produced 1,430 macro records over 143 unique prompts; strict joint validation accepted 74 of 95 proxy-accepted portfolio rows. The deterministic source-level selection retained 43 of 60 construction sources, exceeding the required 30, with category counts code 6, instruction 5, knowledge 5, logic 6, reasoning 5, safety 3, structured 4, and summary 4, plus math 3 and translation 2. All eight required core categories were present, so `DEVELOPMENT_GO` was emitted without changing thresholds.

The pipeline then began the preregistered independent confirmation stage. The first newly seeded confirmation LoRA completed 50/50 steps with finite training loss 0.4126; the second confirmation LoRA loaded normally and entered its own isolated 50-step process. No confirmation prompt endpoint or H6-P decision has yet been observed, and stable-component/MCC12 construction remains conditional on the frozen 19-prompt confirmation gate.

Before the next planned server shutdown, the second confirmation LoRA also completed 50/50 steps with finite loss 0.4082; both registered adapters and their reports were validated on disk. Confirmation preparation produced 86 initial/optimized task endpoints from the 43-prompt frozen development pool. An intentional pause then stopped task validation at an exact 23/86 prefix, with all 23 completed endpoints passing and no technical failure. The `h6_pipeline` wrapper exited with code 0, all GPUs returned to 1 MiB, 70 GB disk remained free, and the server worktree stayed clean on `0aa98a4`. Recovery must rediscover the endpoint, validate both adapters and the 23-row prefix, then restart the same wrapper so task validation resumes at row 24; no confirmation micro, macro, final gate, component, or MCC12 endpoint has been observed.

After the server returned, endpoint rediscovery verified the same clean `0aa98a4` repository, both registered confirmation adapters, all 86 frozen confirmation task inputs, and an exact unique 23-row result prefix matching frozen positions 1--23 by prompt ID; all 23 passed and none had a technical failure. Three GPUs were idle, 70 GB remained free, and GitHub SSH-over-443 remained readable. The unchanged recovery wrapper restarted as pipeline PID 1683, revalidated the complete 190-row development task file, explicitly skipped all 143 completed development micro IDs, and entered the preregistered ten-variant development macro replay before it can automatically resume confirmation at row 24. This replay changes no scientific input, result prefix, threshold, or gate.

The recovery wrapper completed the deterministic development macro replay at 1,430/1,430 records, reproduced `DEVELOPMENT_GO`, and reused both confirmation adapters rather than retraining them. Confirmation task validation then resumed from the frozen 23-row prefix and completed all 86/86 endpoints: every endpoint passed its task evaluator, all 86 prompt IDs were unique, and no technical failure was recorded. The pipeline automatically advanced to four-probe complete-block confirmation micro scoring over 86 unique prompts and durably wrote its first records with all 195 parameter groups; confirmation macro scoring and the final H6-P gate remain unobserved.

Confirmation complete-block micro scoring reached a durable 20/86 unique-prompt checkpoint, approximately one quarter of the frozen pool. All 20 rows had unique prompt IDs and no technical failure, and the live scorer proceeded to prompt 21 with normal three-GPU allocation. Each completed record still covers four probes across all 195 parameter groups; confirmation macro scoring, the H6-P decision, and MCC12 construction remain pending.

Confirmation complete-block micro scoring crossed halfway at a durable 44/86 unique prompts. All written records remained unique and free of technical failures, and the same scorer continued within the active prompt with normal three-GPU allocation and 70 GB free disk. The frozen four-probe, 195-group evaluation remains unchanged; confirmation macro endpoints and the final H6-P gate are still pending.

At the user's planned server-shutdown pause, confirmation complete-block micro scoring had durably reached 55/86 unique prompts. All 55 rows parsed, matched the first 55 frozen unique-prompt IDs in order, were mutually unique, and contained no technical failure; task validation remained complete at 86/86 with 86 passes. An intentional `Ctrl-C` stopped only the verified `h6_pipeline` session, the wrapper recorded exit code 0, all three GPUs returned to 1 MiB, disk retained 70 GB free, and the clean server stayed on `0aa98a4`. Resume must rediscover the possibly changed endpoint, revalidate the two confirmation adapters and exact 55-row prefix, then restart the unchanged recovery wrapper so micro scoring skips rows 1--55 and recomputes only the interrupted active prompt onward. Confirmation macro scoring, H6-P, and MCC12 remain unobserved.

After the server returned, endpoint rediscovery found the same clean `0aa98a4` repository, both intact confirmation adapters, 86/86 passing task records, and the exact ordered 55/86 confirmation micro prefix with no technical failure. Three GPUs were idle, 70 GB remained free, and GitHub SSH-over-443 was readable. The unchanged recovery wrapper restarted idempotently as pipeline PID 1589, validated construction 60/60 and development task validation 190/190, then entered the expected completed-development replay before confirmation micro scoring can skip rows 1--55 and resume row 56. No scientific input, threshold, or completed result changed.

The restarted wrapper completed its deterministic development replay, reused both confirmation adapters, preserved all 86 passing task records, and explicitly skipped confirmation micro IDs `unique::0000` through `unique::0054`. Confirmation micro scorer PID 2809 then entered row 56 (`unique::0055`) with the full four-probe, 195-group calculation and normal three-GPU allocation. The durable prefix remained exactly 55/86 with no duplicates or technical failures at the recovery boundary; no completed endpoint was recomputed or overwritten.

Resumed confirmation complete-block micro scoring passed the three-quarter point at a durable 72/86 unique-prompt records and entered prompt 73. Every written row remained unique and free of technical failures, while the same four-probe, 195-group scorer continued with normal three-GPU allocation and 70 GB free disk. Confirmation macro scoring, the final H6-P gate, and MCC12 construction remain pending.

Confirmation complete-block micro scoring then completed all 86/86 unique prompts with four probes and all 195 parameter groups per record; every row remained unique and free of technical failures. The pipeline automatically entered the ten frozen confirmation attack variants and sequentially reached variant 10/10 after both finetuning, both Gaussian-noise, both quantization, both structured-pruning, and the first unstructured-pruning variant. Macro process PID 4835 remained live with normal GPU allocation, but its 860-record output had not yet been finalized; H6-P and MCC12 therefore remain pending.

The ten confirmation variants completed all 860 macro records, after which strict held-out validation retained 39 of 43 frozen reconstruction sources. H6-P therefore passed its preregistered minimum of 19 with no technical errors and all eight core categories represented: code 5, instruction 5, knowledge 5, logic 4, reasoning 5, safety 3, structured 3, and summary 4, plus math 3 and translation 2. The pipeline then built the 32B-specific stable component universe from 280 build and 56 audit calibration prompts. Saturation and audit-novelty gates passed over 81,761 global components; deterministic global-unweighted selection produced MCC12 with zero missing required categories, factorization error `5.55e-17`, 28.53% final global coverage, and 86.25% selection efficiency. The fingerprint SHA-256 is `cea665b293acc49e6f4ec6c48781c4d613a3a6cc5287f7e542f6f87667f291d0`; the wrapper exited rc=0 and all GPUs returned to 1 MiB.

Terminal H6 evidence was copied into `reproducibility/fingerprint_h6_qwen32b_mcc_20260812` without modifying H5. The archive retains the 39-prompt confirmed pool, H6-P reports, frozen config and protocol, calibration and candidate manifests, activation audit, global component universe, MCC selection, final 3.1 MB fingerprint, a concise final report, and SHA-256 checksums. Raw activation profiles, checkpoints, adapters, and model shards remain excluded as declared large artifacts. The archive validator and the existing H6 protocol suite pass 9/9. This completes prompt-pool reconstruction and MCC12 construction only; modified-state detection verification remains a separate preregistered experiment.
