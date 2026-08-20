# H8 D2-B0 Development Sampling Authorization Preflight

## Scope and hard boundary

D2-B0 may revise and freeze the detector-development selection rule, audit fresh-attack provenance, materialize the eight already registered D2-A attack endpoints, and generate exactly one `detector_development_attack_smoke_only` response per endpoint. It may not generate any of the 3,840 formal development responses, read development outcomes for configuration selection, estimate formal FPR/TPR, select a sample structure or Top-r aggregation, or freeze the detector.

The frozen MMD measurement layer, frozen score layer, D2-A generation schedules, nested memberships, and global-permutation seeds are immutable inputs. D2-B0 smoke seeds must be response-level unique, uint32, and disjoint from every proposed D2-A formal generation seed and every available frozen repository seed.

## Revised configuration-selection rule

The candidate dictionary remains exactly the Cartesian product of four sample structures (`r40_q10`, `r40_q20`, `r60_q10`, `r60_q20`) and Top-2/Top-3/Top-4, for twelve configurations. Max remains diagnostic only; Energy is absent; local prompt-level p-values never enter aggregation.

After a separately authorized future development run, configuration selection uses this fixed lexicographic rule:

1. minimize the false-positive count among the five Development Intact evaluation units;
2. maximize the minimum detected-endpoint count across Gaussian, Pruning, LoRA, and Quantization, where each count is an integer from zero to two;
3. maximize the total detected count across all eight development attack endpoints;
4. prefer `q10` to `q20`;
5. prefer `r40` to `r60`;
6. use the frozen Top-r order Top-2, then Top-3, then Top-4.

The five intact outcomes and two endpoints per family are configuration-selection development counts only. They are not formal FPR/TPR estimates, and percentages such as 0%, 50%, or 100% must not be described as stable operating characteristics. Formal FPR/TPR remain reserved for fresh held-out confirmation.

## Fresh-attack provenance

Freshness is audited against all versioned H6 train, development, and confirmation attack manifests. For every D2 endpoint the audit compares explicit identifier, normalized family, full configuration, materialization/training seed, and any available artifact hash. A claim of zero ID overlap is permitted only if H6 explicit IDs are actually recovered. Missing historical model-state artifacts must be reported as unavailable rather than inferred to be different.

The two quantization endpoints must be realized exactly as registered BitsAndBytes FP4/BF16/no-double-quant and NF4/FP16/no-double-quant loads. `is_loaded_in_4bit`, quantized module types/counts, requested/realized method, compute dtype, double-quant state, model memory footprint, and a state/provenance hash are mandatory. A BF16/FP16 fallback is a failure.

Gaussian and pruning endpoints must record the registered configuration and materialization seed, selected tensor/parameter counts, changed count/fraction, deterministic before/after state-sketch hashes, and a hash of the attack report including mask/selection details.

LoRA adapters must be trained from the frozen `data/h8_d2_fresh_isolated_attack_training_data_v1.jsonl` only. Exact and normalized MCC12 IDs, prompts, archived responses, and H6 confirmed prompt forms are forbidden. The data file, training seed, requested/completed steps, adapter config, and adapter weights are hashed. Adapters that do not complete the registered 40 or 60 steps are failures.

## Eight-response smoke

Each of the eight endpoints receives exactly one independent smoke request. Smoke records save response ID, endpoint ID/family, generation seed, materialization seed, prompt identity, rendered-prompt and token hashes/IDs, completion token IDs, decoded text, stop reason, attempt history, exact model/tokenizer/template/EOS/runtime provenance, and endpoint materialization hashes. Legal first-token EOS is retained. Technical retries may reuse only the same seed.

Smoke responses have `formal_development_eligible=false`, `nested_q10_q20_eligible=false`, and `configuration_selection_eligible=false`. No MMD, score, Top-r statistic, permutation p-value, detection label, or configuration comparison is computed from them.

Each endpoint executes in an isolated subprocess. The parent process audits GPU cleanup after every endpoint and after the complete run. Existing successful endpoint records cannot be overwritten.

## Terminal report

PASS requires eight unique smoke responses, exact endpoint/configuration realization, zero smoke/formal seed overlap, LoRA data isolation, provenance/hash consistency, all D2 CPU tests, and complete GPU cleanup. The terminal report must state:

- `formal_development_responses=0`;
- `attack_smoke_only_responses=8`;
- `development_comparison_performed=false`;
- `sample_size=not_selected`;
- `aggregation=not_selected`;
- `detector=not_frozen`.

After the report is archived, execution stops and waits for an explicit authorization before any of the 3,840 formal development responses are generated.
