# R2 independent intact validation: frozen before new responses

Authorized task: advance R2 independent negative validation. No attacks, fine-tuning,
H6/MCC changes, threshold tuning, optional sampling extension or server shutdown.

## Budget and isolation

Use frozen Qwen2.5-0.5B-Instruct, 12 Stage1 prompts and 64-token generation budget.
Fresh Fit:48/prompt (576 total). Fresh Calibration:48/prompt (576 total).
Validation:60 independent24-reference/24-target comparisons per prompt (34,560).
Total maximum:35,712 responses. No old responses used for estimation. New unique
generation seeds start at2,600,000,000 and are checked against available Stage1 banks.
Sampling is batch1. Reference/target are interleaved within each independent unit.
Generation uses fixed seeds and PyTorch warn-only determinism because this CUDA
build lacks deterministic top-p cumsum. Do not claim bitwise regeneration from seeds.
Semantic encoding retains strict deterministic algorithms. The first strict-generation
launch failed before any response; preserve its v1 manifest/log and use a new v2 run.
Roles and units never reuse response identities/seeds; identical text by chance is
not leakage and must not be removed. All stopping decisions below are fixed-sample.

## Frozen features and statistics

Fit scalers, exclusions, global fallback and prompt bandwidths using ONLY fresh Fit.
Freeze these artifacts before Calibration and before any Validation generation.
Calibration consists of one24-vs24 technical smoke check per prompt; its p-values
cannot adjust alpha, kernel, raw combination, prompts or sample count. Stop only on
technical invalidity, not on an ordinary alpha=.05 rejection.

Use R1 separate raw bounded-max and unstandardized H8 MMD permutation channels,
999 permutations, plus-one p-values and1e-12 conservative tie tolerance. No OR.
Fresh BGE cache: fixed revision and weight hash, explicit512-token truncation,
FP32/eager attention, normalization, no text prefix, singleton encoding and fixed
runtime identity. Never import R1 vectors or batch dependent recomputations.

## Acceptance frozen BEFORE responses

1. Exactly60 validation units per prompt; all provenance/seed/count checks pass;
   no unevaluable unit in either channel. Missing units cannot count as negatives.
2. Report each prompt's rejection count and pointwise exact95% binomial interval.
   These are pointwise, not simultaneous assurances that all prompts satisfy5%.
3. Primary ENGINEERING gate: for each of the two channels, the uniform12-prompt
   macro rejection rate has an upper bound<=10%. Use Hoeffding with720 independent
   Bernoulli decisions conditional on frozen Fit, allowing heterogeneous prompt
   probabilities. Bound=observed+sqrt(log(2/.05)/(2*720)), clipped at1. The union
   bound supplies95% simultaneous coverage of the two macro upper bounds.
   This10% tolerance is NOT a change to the nominal test alpha=.05 and is not a
   claim that individual-prompt FPR<=5%. It is an explicit limited engineering gate.
4. Local safety check: one-sided binomial inflation tests againstp=.05, with
   Bonferroni cutoff.05/24 across12 prompts×2 channels. Any flag prevents a pass.
5. If a macro upper bound exceeds10% without a local flag, classify INCONCLUSIVE,
   not necessarily a defective detector. No sample extension or threshold revision.
6. R2 passing does not establish attack power, task retention or authorize Stage2;
   R3 still remains. Conditional independence/stationarity of frozen sampling is an
   assumption, not something seed uniqueness alone mathematically proves.

The heterogeneous macro bound avoids treating all prompts as sharing a common
binomial probability. Method references:
- https://people.eecs.berkeley.edu/~bartlett/courses/281b-sp08/12.pdf
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6405018/

## Recovery and evidence

Pre-data manifest binds config, runtime, model/tokenizer files, code and schedule.
Per-role append-only responses carry previous-record hashes and exact schedule keys.
Validate prefix before resume; reject changed config/code/assets or partial records.
Use an OS process lock to prevent concurrent writers. Fit/decisions/cache artifacts
are hash-bound. STATUS.json and stdout report phase, count and observed-speed ETA.
Final report distinguishes technical completion, inconclusive calibration and pass.
Never auto-shutdown this server. Read-only progress checks do not change experiment.
