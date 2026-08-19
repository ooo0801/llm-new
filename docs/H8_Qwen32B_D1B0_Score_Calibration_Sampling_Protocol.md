# H8 D1-B0 Score Calibration Sampling Protocol Freeze / Runner Preflight

## 1. Authorization boundary

D1-B0 freezes manifests and validates the sampling runner. Formal 2,400-response
sampling is not authorized. Only 24 new `score_calibration_smoke_only` responses
may be generated. The runner must fail closed if invoked in formal mode while
`formal_sampling_authorized=false`.

The H8 frozen MMD measurement manifest and D1-A score schema are immutable
dependencies. Every invocation verifies their byte and payload hashes before any
model load or response generation.

## 2. Formal bank design

The formal generation manifest contains 200 rounds × 12 MCC12 fingerprints =
2,400 response-level requests at batch size 1. Every round contains every prompt
exactly once. Even rounds have role `score_calibration_fit_only`; odd rounds have
role `score_calibration_stability_audit_only`. Thus every prompt receives exactly
100 fit and 100 audit responses. Prompt order within each round is fixed by a
domain-separated SHA256 ordering rule.

Both roles have formal-reference, held-out, Target, and attack eligibility set to
false. Only the fit role may later enter `m_j`, `a_j`, and `a_global` fitting. Audit
responses are restricted to independent D1-C stability evaluation.

## 3. Seed namespaces

All 2,400 model generation seeds are unique uint32 values derived before the first
response with domain-separated SHA256 plus deterministic collision rejection.
They must not overlap any available frozen H6/H7/H8/M0 response seed. Fit and audit
seed sets are disjoint.

The 24 smoke generation seeds use a separate domain/root and are disjoint from the
formal schedule and repository history. CPU pseudo-split/resampling seeds use a
third reserved namespace and are not generated or consumed in D1-B0. A model
generation seed must never be reused as a CPU split seed.

## 4. One bank, four later structures

D1-B generates one role-specific response bank, not four sample-structure data
sets. D1-C will construct `r40_q10`, `r40_q20`, `r60_q10`, and `r60_q20` null
statistics independently from the same role-specific bank using frozen,
without-replacement pseudo splits. Statistics and score parameters from different
structures must never be pooled.

## 5. Response and retry contract

Every response stores role, round, prompt and response identity, generation seed,
attempt history, rendered-prompt hash and token IDs, completion token IDs, decoded
text, stop reason, EOS metadata, and complete frozen provenance. A legal first-token
EOS empty response is retained. Technical failures may only retry the same response
with the same seed; seed replacement is forbidden.

## 6. D1-B0 smoke

The smoke manifest contains 12 prompts × 2 independent generation seeds. Each
record has actual role `score_calibration_smoke_only` and an `intended_bank_role`
of fit or audit. Smoke IDs and seeds cannot appear in the 2,400-request formal
manifest and smoke responses can never enter either bank.

## 7. Stop condition

After manifest audit, CPU tests, 24-response smoke, metadata/role/seed audit, and GPU
cleanup, D1-B0 writes a preflight report and stops. A later explicit authorization
and a new committed authorization state are required before formal sampling.
