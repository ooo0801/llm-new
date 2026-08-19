# H8 Qwen2.5-32B D1-B1 Formal Score Calibration Sampling Protocol

## Scope and authorization

This phase is authorized only to generate the 2,400 responses already enumerated in the frozen D1-B0 formal generation manifest. It does not authorize score-parameter fitting, sample-size selection, Top-r/Max/Energy, formal Reference, held-out, Target, or attack generation.

D1-B1 uses a separate authorization artifact. The D1-B0 configuration remains preflight-only and must continue to reject formal sampling.

## Immutable inputs

- The final D1-B0 PASS report, never the `failed_before_generation` diagnostic report.
- Formal generation manifest file SHA256 `e6db0eda9d373c7d62f2a2efeacf4ca127476a0ab91d104665bcaa7354577002`.
- Frozen MMD measurement manifest and all payload hashes.
- D1-A score schema.
- H6 MCC12 fingerprint artifact.
- Qwen2.5-32B-Instruct snapshot revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`.
- Frozen tokenizer, chat template, dual EOS list, decoding configuration, and runtime identity.

## Exact schedule

The existing manifest is executed without derivation or replacement: 200 rounds × 12 prompts, batch size 1. Even rounds have role `score_calibration_fit_only`; odd rounds have role `score_calibration_stability_audit_only`. Each prompt must finish with exactly 100 fit and 100 audit responses. Formal generation seeds must not be changed, regenerated, or pooled with later CPU resampling seeds.

## Fail-closed resumability

Responses form a strict schedule prefix. Each persisted response binds `response_id`, schedule position, round, prompt, role, seed, manifest-request hash, frozen provenance hash, payload hash, and a chained record hash. Before resume, every response and attempt event is revalidated against the frozen manifest. A valid successful response is immutable and skipped; the next generation request is the first schedule item without a valid persisted success.

Attempt indices are cumulative across processes. Technical failure permits only same-seed retry within the frozen attempt budget. A success event without its response record, a gap, duplicate, future event, changed seed, broken hash, or non-prefix response is a hard stop requiring review.

## Runtime gate

The worker runs with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` against the fixed local snapshot. Before loading and before every next response it verifies the frozen artifacts and live model/tokenizer/runtime identity. Snapshot identity is rechecked at every round boundary. A legal first-token EOS empty completion is retained.

## Completion audit and stop

Completion requires exactly 2,400 responses, 1,200 per role, 100+100 per prompt, unique response IDs and seeds, zero fit/audit and formal/smoke seed overlap, exact round-role agreement, complete fields, same-seed retry compliance, consistent provenance/hashes, and complete GPU cleanup.

The final report must state that score fitting, sample-size selection, Top-r selection, formal Reference, held-out, and attack generation were not performed. D1-C remains forbidden until separately approved.
