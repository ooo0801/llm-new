# H8 F1-A Fresh Held-out Final Confirmation Protocol Freeze / Preflight

## 1. Purpose and phase boundary

F1-A freezes the future final-confirmation design for the already selected detector
`r60_q10_top2`. It may generate only the six explicitly marked `smoke_only` responses.
It must not generate any of the 12,720 formal responses or compute formal detector
statistics. Formal sampling requires a new authorization artifact that binds the final
PASS preflight report and every manifest hash.

## 2. Frozen detector

The D2-C frozen detector is loaded fail-closed from `DETECTOR_FROZEN_MANIFEST.json`.
The measurement and score layers remain frozen. The following values cannot change:

- `N_R=60`, `N_Q=10`;
- Top-2 sum aggregation;
- model-level within-prompt global permutation with `B=999`;
- `p_global=(1 + #\{T_perm >= T_obs\})/1000`;
- `alpha=0.05`.

No per-prompt local p-value is used as an aggregation input. No scaler, bandwidth,
MMD center/scale parameter, sample size, aggregation, or threshold may be refitted.

## 3. Frozen formal response design

The future formal bank contains exactly 12,720 new Qwen2.5-32B responses at batch 1:

- 60 `final_reference_only` responses per prompt: 720 total;
- 60 independent `final_heldout_intact_target_only` units, each containing 10 responses
  per prompt: 7,200 total;
- four attack families with 10 fresh held-out endpoints per family, each endpoint
  containing 10 responses per prompt: 4,800 total.

The three roles are mutually exclusive. They may be used only by the frozen final
detector and are ineligible for measurement fitting, score fitting, detector selection,
or development analysis. The 60 reference rounds are deterministically interleaved
through the intact-acquisition superblocks. Attack endpoints are executed as isolated
model-state workers so one endpoint cannot contaminate another.

## 4. Fresh attack endpoints

The endpoint manifest freezes 10 Gaussian, 10 pruning, 10 LoRA, and 10 quantization
identities before any formal output is observed. Every identity binds family,
configuration, materialization seed, LoRA training seed where applicable, and a planned
identity SHA256. Endpoint IDs, configurations, and seeds must not overlap H6 or D2.
Quantization configurations are pairwise distinct because changing a seed alone does not
create a new deterministic quantized model. Five INT8 endpoints therefore use distinct
frozen outlier thresholds in addition to five distinct 4-bit configurations.

Preflight freshness is a configuration/seed/identity guarantee, not a claim that all 40
model payloads have already been materialized. At formal materialization, Gaussian and
pruning state sketches, LoRA adapter hashes/training reports, and quantization load/state
reports must match the frozen identity and be sealed before that endpoint's first answer.
An endpoint that fails materialization cannot be replaced after results are visible.

## 5. Seed and response identity isolation

All 12,720 generation seeds are derived before the first formal response by a frozen,
domain-separated SHA256-to-uint32 algorithm with collision counters. The derivation uses
the available historical H6, M0, D1, D2 generation, smoke, materialization, and training
seed inventory as a forbidden set. All formal response IDs use the `h8-f1-` namespace
and are checked against available historical IDs. Materialization seeds, LoRA training
seeds, formal generation seeds, smoke seeds, and 64-bit CPU permutation streams are
separate namespaces.

Technical failures permit only a same-seed retry. A successful record is immutable.
A legal first-token EOS empty response is retained and cannot be replaced.

## 6. Final detector output, estimand, and confidence intervals

The Primary estimand is the conditional performance of the fully frozen detector
package, including this one frozen shared Reference bank. F1-A does not estimate the
additional variability that would arise if the 60-response-per-prompt Reference bank
were regenerated. The Reference bank is therefore part of the detector identity rather
than an independently replicated evaluation unit.

Each of the 100 evaluation units produces exactly one `p_global` and one decision
`p_global<=0.05`. The final report will contain:

- false-positive count among 60 intact units, conditional FPR, and a two-sided 95%
  conditional Clopper-Pearson interval;
- detected count out of 10 and held-out endpoint-panel detection rate separately for
  Gaussian, pruning, LoRA, and quantization, with a two-sided 95% conditional
  Clopper-Pearson interval;
- overall detected count out of 40 as a secondary total that cannot replace the four
  family-specific results.

The 60-per-prompt Reference bank is shared across all evaluation units. The requested
Clopper-Pearson intervals are therefore labelled conditional descriptive binomial
intervals for this frozen bank. They must not be described as unconditional exact
coverage across repeated Reference-bank construction. Shared-Reference dependence
across units and the omission of between-Reference-bank uncertainty must remain visible
in every final report and table.

The ten endpoints within an attack family intentionally use heterogeneous frozen
configurations. Their detected/10 result is a finite preregistered held-out endpoint
panel rate. `TPR` may appear only as an explicitly labelled operational alias; F1-A does
not claim that the ten endpoints are iid draws from an attack superpopulation.

An optional Reference-resampling sensitivity diagnostic may be added later only as a
secondary analysis. It cannot be presented as a replacement for independently generated
Reference banks and cannot modify any Primary decision or interval.

## 7. Tiny smoke

F1-A authorizes exactly six `smoke_only` responses: one Reference-role plumbing check,
one intact-target-role plumbing check, and one separate smoke endpoint from each attack
family. Smoke endpoints, seeds, responses, and adapters are disjoint from the 40 formal
endpoints and 12,720 formal schedule. Smoke verifies offline snapshot loading, role and
seed metadata, EOS/token recording, same-seed retry handling, materialization evidence,
immutable output, and GPU cleanup. It must not extract final features, run MMD, perform
global permutation, or report detector performance.

## 8. Stop conditions

Any frozen hash mismatch, seed/ID/configuration overlap, role/count mismatch, unsupported
attack materialization, technical retry with a changed seed, non-clean GPU state, or
attempt to invoke formal sampling without a separate authorization is an immediate
NO-GO. After preflight and six-response smoke audit, execution stops.
