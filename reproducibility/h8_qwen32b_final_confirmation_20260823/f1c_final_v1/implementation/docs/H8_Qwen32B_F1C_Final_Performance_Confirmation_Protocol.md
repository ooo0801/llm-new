# H8 F1-C Final Performance Confirmation Protocol

## Scope and authorization boundary

F1-C is a terminal, offline-only confirmation of the already frozen detector
`r60_q10_top2`.  Its complete data universe is the 12,720 formal responses
sealed by F1-B: 720 shared Reference responses, 7,200 intact Target responses,
and 4,800 responses from 40 preregistered held-out attack endpoints.  F1-C
must not generate a Qwen response, train or materialize an attack, replace a
response, derive a new seed namespace, change membership, or fit/tune any
feature, MMD, Score, sample-size, aggregation, or decision parameter.

## Fail-closed preflight

Before feature extraction, the runner verifies all of the following and stops
on the first mismatch:

1. the H8 branch and Git ancestry include the F1-B terminal archive and the
   frozen F1-C implementation;
2. the F1-B terminal audit is PASS and reports exactly 12,720 formal responses;
3. every entry of the 178-file F1-B terminal SHA256 index has the frozen size
   and SHA256, including all 41 response partitions;
4. every response matches its frozen F1-A response ID, schedule position,
   generation seed, role, prompt, unit, endpoint, retry record, materialization
   identity, and F1-B frozen identity;
5. the H6 MCC12 fingerprint, Frozen MMD manifest and inventory, Frozen Score
   manifest and inventory, selected detector payload, detector manifest, F1-A
   generation/attack/permutation/freshness/history/smoke manifests, and their
   payload hashes all match;
6. the frozen BGE model revision exists in the fixed offline cache, and the
   Python/NumPy/PyTorch/SentenceTransformers runtime equals the feature-fitting
   runtime.

The 41 formal response partition hashes are checked again after all F1-C
outputs are written.

## Deterministic 528-D feature layer

Responses are ordered by the frozen F1-A schedule.  Surface and normalized
512-D BGE embeddings are computed once per unique response text in sorted text
order; the five task dimensions are reconstructed only from the frozen H6
fingerprint metadata.  Output is little-endian float64 with family slices
`surface[0:11]`, `semantic[11:523]`, and `task[523:528]`.  Required QA is
12,720 rows, 528 dimensions, zero NaN/Inf/missing values, and one row per unique
response ID.  Final data never enter schema, scaler, exclusion-mask, or
bandwidth fitting.

## Frozen membership and measurement

All 100 evaluation units share the single 60-response-per-prompt Final
Reference bank.  Each unit contributes exactly 10 Target responses per prompt.
There are 60 intact units and ten endpoints in each of Gaussian, Pruning, LoRA,
and Quantization.  Membership comes directly from the F1-A generation and
permutation manifests and is never resampled.

For every unit and MCC12 prompt, the prompt-specific frozen family-balanced
scaler and RBF bandwidth produce generalized unequal-size unbiased MMD² for
60-versus-10 samples.  Negative raw MMD² is retained.  Only the frozen
`r60_q10` Score mapping applies the one-sided clamp:

- nondegenerate: `S=max(0,(D-m_j)/a_j)`;
- structurally degenerate: `S=max(0,D/a_global)`.

The observed statistic is the sum of the two largest scores in that unit.
Top-2 prompt identities are selected dynamically from all 12 scores and are
descriptive only.

## Frozen global permutation

For each unit, the F1-A manifest contains a frozen stream seed and the SHA256 of
its 11,988 prompt-level seeds.  The runner expands this compressed frozen
stream with the registered domain `h8-f1a-within-prompt-permutation-v1`, checks
the complete per-stream digest, and uses all 1,198,800 prompt seeds exactly
once.  No new root, domain, order, or seed is introduced.

Each of 999 iterations independently permutes labels inside each prompt's
70-response pool, preserves group sizes 60/10, recomputes MMD, frozen Score,
and dynamic Top-2, and never exchanges samples across prompts.  The decision is

`p_global=(1+count(T_perm>=T_obs))/1000`, alarm iff `p_global<=0.05`.

The p-value must lie on the 0.001 grid, and alarm must be exactly equivalent to
an exceedance count no greater than 49.

## Final estimands and interpretation

The primary intact result is false positives out of 60, FPR, and a two-sided
95% Clopper-Pearson conditional descriptive interval.  Each attack family is
reported separately as detected endpoints out of 10, held-out endpoint-panel
detection rate, and its conditional descriptive interval.  Detected out of 40
is secondary and cannot replace family-specific results.

All intervals and rates are explicitly conditional on the single frozen shared
Final Reference bank.  They do not include uncertainty from regenerating a new
Reference bank.  The deliberately heterogeneous attack panels are not treated
as iid draws from an attack superpopulation.

## Terminal artifacts

F1-C writes a feature cache and QA report; 100 unit-level records; 1,200
prompt-level descriptive records; observed MMD and Score arrays; permutation
MMD, Score, and Top-2 arrays; permutation audit; performance summary; final
report; and a terminal SHA256 index.  The compact Git archive excludes raw
F1-B banks and large NumPy caches but binds them through their terminal hashes.
After terminal PASS, execution stops without tuning, replacement, or follow-up
sampling.
