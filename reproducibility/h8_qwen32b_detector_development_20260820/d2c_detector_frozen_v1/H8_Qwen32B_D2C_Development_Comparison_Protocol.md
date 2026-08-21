# H8 D2-C Development Comparison Protocol

## Scope and authorization

D2-C is an offline detector-development comparison over the frozen D2-B1 bank. It is authorized to run global permutation tests for the twelve configurations frozen by D2-B0 and to select exactly one detector using the already frozen count-based lexicographic rule. It is not authorized to generate model responses, change membership, refit the MMD or score layers, inspect final/held-out data, or estimate formal FPR/TPR.

The only admissible response roles are:

- `detector_development_reference_only` (720 responses);
- `detector_development_intact_target_only` (1,200 responses);
- `detector_development_attack_only` (1,920 responses).

The D2-B0 smoke bank is forbidden. Every input file, frozen manifest, configuration rule, nested-membership artifact and permutation-seed artifact is loaded fail-closed by SHA256.

## Fixed measurement and score layers

For each of the twelve MCC fingerprints, raw text is transformed with the frozen deterministic 528-dimensional feature schema. The prompt-specific frozen family-balanced scaler and frozen RBF bandwidth are applied without refitting. Generalized unequal-size unbiased MMD squared is computed without clamping negative values. The frozen structure-specific score map converts each raw MMD value to the nonnegative score `S_j`.

No development response may influence a scaler, bandwidth, structural-degeneracy label, score center, score scale or global fallback scale.

## Frozen evaluation units and membership

The frozen nested-subset manifest defines thirteen evaluation units:

- five intact development blocks;
- two Gaussian endpoints;
- two pruning endpoints;
- two LoRA endpoints;
- two quantization endpoints.

For every prompt, `R40` is a subset of `R60` and `Q10` is a subset of `Q20`. The same frozen 60-response reference bank and 20-response target bank serve all four sample structures. D2-C resolves membership only by frozen response ID and rejects missing, duplicated or role-incompatible members.

## Global permutation detector

For each evaluation unit and sample structure, the observed statistic is computed for all twelve prompts. For every permutation, labels are shuffled independently within each prompt's pooled `R_j union Q_j`, preserving the frozen `N_R:N_Q` group sizes. Each raw permuted MMD value is passed through the same frozen score map.

Primary aggregation is limited to:

`T_r = sum of the r largest S_j`, for `r` in `{2,3,4}`.

The three Top-r variants reuse the same frozen permutation stream. The development p-value is:

`p = (1 + count(T_perm >= T_observed)) / 1000`,

using exactly 999 permutations and `alpha=0.05`. Max is reported only as a diagnostic. Local prompt p-values and Energy aggregation are forbidden.

## Frozen configuration selection

The twelve candidates are the Cartesian product of four sample structures and Top-2/3/4. Selection uses this immutable lexicographic rule:

1. minimize the false-positive count across five intact development units;
2. maximize the minimum detected-endpoint count across Gaussian, pruning, LoRA and quantization, with each count in 0--2;
3. maximize total detected endpoints across all eight development attacks;
4. prefer `q10` over `q20`;
5. prefer `r40` over `r60`;
6. prefer Top-2, then Top-3, then Top-4.

The five intact units and two endpoints per family are development counts only. They are not formal estimates of FPR or TPR. No single Gaussian result may directly determine the detector.

## Output and stopping boundary

If every numerical, membership, provenance and hash gate passes, D2-C freezes:

- all 156 configuration-by-unit results;
- the 12-row comparison table;
- the selected sample structure, Top-r aggregation, 999-permutation global test and alpha;
- hashes of the response bank, features, nested membership, permutation streams, MMD layer and score layer.

The detector state becomes `frozen_development_selected`, while formal performance confirmation remains `not_performed`. After archival and push, execution stops. Fresh final held-out or confirmation sampling requires separate authorization.
