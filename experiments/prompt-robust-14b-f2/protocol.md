# Experiment F2 Protocol: Targeted Logic/Summary Expansion, Re-optimization, and Independent Confirmation

Status: **FROZEN BEFORE ANY F2 MODEL ENDPOINT**

## Motivation and hypothesis

F1 completed every development endpoint but could freeze only 29/30 rows. The failure was localized before confirmation: logic supplied two rather than three eligible unique sources and summary supplied one rather than three. F2 preserves F1 as a No-Go and tests a new, targeted construction intervention instead of altering F1 candidates or gates.

**H-F2:** after targeted expansion, construction, and multi-seed development selection, the frozen 30-prompt set will retain at least 23 prompts under an independent confirmation split, and the retained set will contain all eight categories.

The unchanged strict 5/5 count is secondary. H-F2 is supported only if construction, development, confirmation technical completeness, retained count, and category coverage all pass.

## Stage 0: targeted source expansion

- Add exactly 16 new logic sources and 16 new summary sources.
- Each source has a deterministic ID and clean UTF-8 Chinese text and must not exactly duplicate an initial or optimized F1 text.
- Logic uses exact-answer evaluators over varied transitive, implication, ordering, and classification tasks.
- Summary requires at most 25 non-whitespace characters and two preregistered semantic keywords. This strengthens the existing task guard against fluent but semantically corrupted optimized instructions.
- The source file, evaluators, targets, model, seeds, construction method, attacks, ranking, quotas, and confirmation gate are frozen before construction endpoints.

## Stage 1: model-aware re-optimization

- Model: `Qwen/Qwen2.5-14B-Instruct@cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, BF16, balanced GPUs 0/1/2, no CPU/disk offload.
- Reuse only Experiment C's frozen training/calibration evidence: calibration SHA-256 `66944c94...d066ad0c` and training-registry SHA-256 `288791e6...e385f5f`.
- Run the frozen Experiment C v4 discrete joint optimizer for three rounds, with task guard enabled, maximum edit ratio 0.25, PPL ratio at most 2, and at least 3/5 non-degraded construction families.
- Retain at most two distinct task-valid positive-proxy candidates per new source using the already frozen portfolio rule.
- Construction advances only if each expanded category supplies candidates from at least three different new sources. A failure is a terminal construction No-Go; no observed result may trigger another optimization pass in F2.

## Stage 2: combined-pool development screening

- Assemble all 64 untouched F1 base portfolio candidates plus every valid F2 expansion portfolio candidate.
- Exact prompt-ID and optimized-text duplicates are forbidden.
- Development uses new seeds `2026080811`–`2026080813`, micro seed `2026080810`, and bootstrap seed `2026080814`.
- Three variants per family are used. Structured pruning covers three configurations not used in F1: random 5% FFN/all layers, random 5% heads/all layers, and magnitude 5% heads/random half of layers.
- A candidate is eligible only when both task endpoints pass, complete-block micro gain is positive, and equal-family macro gain is positive.
- Rank lexicographically by structured non-degeneration, number of non-degraded families, worst family median relative gain, structured median relative gain, macro relative gain, micro relative gain, and prompt ID.
- Select three candidates per category in fixed order `code, instruction, knowledge, logic, reasoning, safety, structured, summary`, then fill globally to 30.
- At most one candidate per original source and one per optimized-text hash may be selected.
- If any category lacks three eligible unique sources or 30 rows cannot be frozen, development is a terminal No-Go.

## Stage 3: independent confirmation

- Confirmation begins only after the exact 30 rows are frozen.
- It uses new seeds `2026080821` and `2026080822`, micro seed `2026080820`, and bootstrap seed `2026080823`.
- Structured pruning is held out from F2 development and uses two stricter 6% configurations: magnitude FFN/all layers and random heads/random half of layers.
- The unchanged legacy gate requires: both tasks pass; positive complete-block micro gain; positive equal-family macro gain; and at least 3/5 family medians non-degraded within 1%.
- H-F2 requires at least 23/30 legacy-retained rows and all eight categories in the retained subset.
- Confirmation cannot repair, replace, rerank, or re-optimize any frozen row.

## Conditional final fingerprint

Only if H-F2 is supported, G2 constructs the final MCC12 fingerprint using the unchanged 7B stable-component thresholds and frozen saturation/audit gates. Only if G2 passes, H2 performs the final prompt-stratified MMD verification using a preregistered intact state and eleven independently materialized modified states. Their protocols and configuration are committed with F2 before any F2 endpoint.

## Output paths

- Frozen protocol/inputs: `experiments/prompt-robust-14b-f2/`.
- Raw F2 results: `results/experiment_f2_qwen14b_targeted_20260808/`.
- Conditional final fingerprint: `results/fingerprint_g2_qwen14b_mcc_20260808/`.
- Compact evidence: matching paths under `reproducibility/` and `to_human/`.
