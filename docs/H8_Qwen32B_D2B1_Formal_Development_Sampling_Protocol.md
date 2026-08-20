# H8 Qwen2.5-32B D2-B1 Formal Development Sampling Protocol

## 1. Scope and authorization boundary

D2-B1 is a response-generation and integrity-audit phase only. It generates exactly 3,840
development responses from the immutable D2-A generation manifests:

- 720 `detector_development_reference_only` responses;
- 1,200 `detector_development_intact_target_only` responses;
- 1,920 `detector_development_attack_only` responses.

This phase must not compute MMD, scores, global permutations, development comparison, or detector
selection. The terminal state remains `sample_size=not_selected`, `aggregation=not_selected`, and
`detector=not_frozen`.

## 2. Frozen dependencies

Before the first response and on every resume, the runner fails closed unless all of the following
remain byte-identical to the D2-B1 authorization artifact:

1. the frozen MMD measurement manifest and all payloads;
2. the frozen score-calibration manifest and all payloads;
3. the D2-A configuration, reference, intact-target, attack, nested-subset, and attack-instance
   manifests;
4. the D2-B0 revised count-based configuration-selection rule;
5. the D2-B0 terminal PASS report, final SHA-256 index, freshness/materialization audit, eight
   endpoint materialization records, and eight smoke response records;
6. the H6 MCC12 fingerprint, Qwen2.5-32B snapshot, tokenizer, chat template, EOS list, generation
   configuration, and runtime identity;
7. the versioned D2-B1 runner, module, protocol, and authorization artifact.

The D2-B0 GitHub archive commit must be an ancestor of the execution commit. Local, server, and
GitHub branch heads are checked externally immediately before formal generation; the execution
repository must then remain tracked-clean.

## 3. Immutable schedule and response roles

The three D2-A generation manifests are not rewritten. D2-B1 grants independent authorization to
their existing 3,840 response IDs, uint32 generation seeds, prompt orders, endpoint IDs, and global
schedule positions. The combined schedule is strictly ordered by `schedule_position=0..3839`.

The frozen counts are:

- Reference: 60 responses per prompt, 12 prompts, total 720;
- Intact Target: 100 responses per prompt, 12 prompts, total 1,200;
- Attack: 20 responses per endpoint and prompt, 8 endpoints and 12 prompts, total 1,920.

All response IDs and generation seeds must be globally unique. The three role-specific seed sets
must be pairwise disjoint. D2-B0 smoke response IDs and seeds are permanently ineligible and must
have zero overlap with the D2-B1 bank.

## 4. Nested membership

The D2-A nested-subset manifest remains immutable. For each prompt, `R40` is a subset of `R60`.
For every one of the five intact units and eight attack endpoints, `Q10` is a subset of `Q20`.
D2-B1 generates the single maximal Reference and Target banks only; it does not generate separate
banks for the four candidate sample structures.

## 5. Attack identity reuse

The eight D2-B0 endpoints are reused without changing configuration or materialization seed:

- Gaussian ×2;
- Pruning ×2;
- LoRA ×2;
- Quantization ×2 (FP4 and NF4).

Each attack worker rematerializes or reloads its registered endpoint in a fresh process. Before any
formal response for that endpoint, the canonical materialization payload hash must equal the
corresponding D2-B0 materialization hash. LoRA adapter payload hashes, in-memory state sketches and
attack reports, and quantization load/state identities are thereby bound to the D2-B0 evidence.
A mismatch is a hard NO-GO; it is never repaired by changing the attack.

## 6. Generation and retry rules

- Batch size is 1.
- Each response uses its frozen response-level generation seed.
- A legal first-token EOS and empty decoded text are retained as successful responses.
- Only generation exceptions, OOMs, worker failures, or missing records are technical failures.
- A technical retry must use the same response ID and generation seed.
- The cumulative retry budget is two attempts, including attempts made before an interruption.
- A successful record is immutable and is never regenerated or overwritten.

Every successful response records the rendered-prompt hash, input token IDs, completion token IDs,
decoded text, stop reason, token count, attempts, role/endpoint identity, frozen provenance, worker
runtime identity, materialization binding, and hash-chain fields.

## 7. Fail-closed resume

Successful responses form one append-only global prefix of the frozen schedule. Each record binds
its manifest request hash and the three-manifest set hash, and is chained to its predecessor. On
resume, the runner verifies every existing record, attempt event, seed, response ID, endpoint,
role, eligibility flag, provenance hash, payload hash, and chain hash. Attempt events may exist only
for the completed prefix or the immediately next unfinished response. Generation resumes at the
first missing global schedule position.

The intact Reference and Intact Target banks run in a base-model worker. Every attack endpoint runs
in its own worker. GPU cleanup must pass after each worker and at terminal completion.

## 8. Terminal integrity audit

The only permitted terminal analysis checks:

- exact totals 3,840 / 720 / 1,200 / 1,920;
- per-prompt and per-endpoint counts;
- globally unique response IDs and seeds, pairwise role isolation, and zero D2-B0 smoke overlap;
- exact nested membership;
- required-field and provenance completeness;
- same-seed retry compliance and technical-failure counts;
- stop-reason, legal-empty-response, and token-length summaries;
- all eight D2-B0 materialization bindings;
- response, attempt-event, provenance, report, and archive SHA-256 values;
- final GPU worker cleanup.

The terminal report must explicitly state:

- `development_comparison_performed=false`;
- `detector_statistics_computed=false`;
- `sample_size=not_selected`;
- `aggregation=not_selected`;
- `detector=not_frozen`.

After PASS or FAIL, stop and wait for separate authorization. No global permutation or detector
configuration comparison is implied by D2-B1 completion.
