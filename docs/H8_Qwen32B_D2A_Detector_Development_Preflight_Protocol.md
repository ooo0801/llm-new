# H8 D2-A Detector Development Preflight

## Boundary

D2-A is code and manifest preflight only. It performs no model inference and no detector comparison. The frozen MMD measurement layer and frozen D-to-S score layer are fail-closed dependencies. Scalers, bandwidths, structural-degeneracy labels, `m_j`, `a_j`, and `a_global` may not be refitted or replaced.

All proposed D2 development responses are permanently ineligible for later formal Reference, final held-out evaluation, or final confirmation. D2-A creates schedules and membership IDs, not responses.

## Candidate detectors

The four sample structures remain `r40_q10`, `r40_q20`, `r60_q10`, and `r60_q20`. Primary aggregation is restricted to:

`T_r = sum of the r largest values among S_1,...,S_12`, for `r in {2,3,4}`.

This yields exactly twelve configurations. Max score is diagnostic only. Energy is neither implemented nor compared. Per-prompt p-values are not detector inputs.

## Model-level global permutation

For a fixed structure and Top-r, each fingerprint independently permutes labels within its own pooled `R_j union Q_j`, preserving `N_R:N_Q`. Every permutation recomputes the frozen unbiased MMD, maps it through the frozen score parameters, and applies the same Top-r aggregation. No measurement or score parameter is fitted inside a permutation.

Development uses `B_dev=999`, `alpha=0.05`, and

`p_global = (1 + count(T_perm >= T_obs)) / 1000`.

## Proposed development banks

- Intact development Reference: 60 new responses per prompt (720 total).
- Intact development Target: 100 new responses per prompt (1,200 total), partitioned into five disjoint 20-response evaluation blocks.
- Fresh development attacks: Gaussian, Pruning, LoRA, and Quantization; two new endpoint IDs per family and 20 responses per endpoint/prompt (1,920 total).

The maximum proposed total is 3,840 responses, but D2-A authorizes zero. Attack instance IDs and future materialization contracts use the H8-D2 namespace and forbid reuse of any H6 construction or confirmation endpoint.

Before any future response is generated, the manifests freeze response IDs, uint32 generation seeds, attack instance IDs, R40/R60 and Q10/Q20 memberships, evaluation-unit IDs, permutation streams, and configuration IDs. One 60-response Reference bank and one 20-response Target block serve all four sample structures: `R40 subset R60` and `Q10 subset Q20`.

## Development selection rule

Using development attack endpoints only:

1. maximize the minimum detection rate across the four attack families;
2. then maximize the mean detection rate across the four families;
3. then prefer q10 to q20;
4. then prefer r40 to r60;
5. then use the fixed Top-r order Top-2, Top-3, Top-4.

The last order is a preregistered deterministic tie-break, not a performance claim. A single Gaussian endpoint can never directly determine the configuration. Development intact results are reported as sanity diagnostics and do not modify this attack-only selection rule.

## D2-A terminal state

On PASS, the report must retain: `new_model_responses=0`, `sample_size=not_selected`, `aggregation=not_selected`, and `detector=not_frozen`. Formal sampling requires a separate user-approved authorization artifact.

