# 7B fingerprint selection: MCC versus top sensitivity

This is the frozen 2026-10-03 experiment, on branch `experiment/resf-token-small-20260917`.
Read [the Chinese report](7B指纹选择与检测实验报告.md) and [the results CSV](detection80_results.csv).

## Finding

At four prompts and 25 samples per prompt, top sensitivity detects 40/40 attack instances;
MCC detects 37/40. Both have 0/20 normal-panel alarms. Other configurations detect
40/40; MCC at eight prompts and 50 samples has 1/20 normal alarms, all others 0/20.
MCC improves the defined activation-component coverage but does not improve detection
in this experiment. The lowest-budget paired McNemar p-value is 0.25 (unadjusted).
Neither statistical superiority nor equivalence is established.

## Scope and provenance

- Qwen2.5-7B-Instruct, BF16 and eager attention; model identity is in `calibration/MODEL_MANIFEST.json`.
- The mother pool is **synthetic**, not a public benchmark: 400 prompts, four categories.
  100 are calibration prompts; the original optimization pool has 300.
- User budget amendments reduce the optimization scope 300 -> 100 -> 80.
  `PLAN.json` retains the original 300 source identities; **SCOPE_80.json is the active scope**.
  It samples four of each subtask's five members in SCOPE_100, independently of scores.
- 80 optimized sources yield 61 accepted, unique candidate texts; 19 unaccepted sources
  are excluded from fingerprint selection. Another 28 historical searches are archived
  outside the active scope. `GENERATION_SUMMARY.json` includes original-record hashes.
- Acceptance optimizes proxy sensitivity. No strict semantic-preservation or perplexity
  gate was used; acceptance is not proof of task preservation or detection effectiveness.
- MCC and top sensitivity use the same 61 candidates. Each method's four-prompt panel
  is a prefix of its eight-prompt panel. Their union contains 15 unique prompts.
- New tests comprise 20 Gaussian and 20 LoRA instances with independent seeds and
  independent synthetic LoRA training text, plus 20 normal sampling panels from one
  normal model. These are known attack families, not unseen-family validation.
- 15 prompts x 80 responses x 60 endpoints = 72,000 sampled first tokens.
  The 25/50 query configurations reuse prefixes. There are 720 panel decisions and
  12 fixed method/size/query configurations.

## Included evidence

`RESULTS.json`, `DECISIONS.json`, `FINGERPRINTS.json`, `SCORES.json`, candidate/source
mappings, calibration rules and pool, micro/macro scores, coverage profiles, normal
reference distributions and Monte Carlo caches, and adapter training reports.
Full generation traces, raw responses, model weights and adapter weights are excluded
from Git; their complete archive is preserved separately:

- `fingerprint-detection80-final.tar.gz` (1,614,760,979 bytes)
- SHA256 `7d7b158316a0b8918579a09b89b735bb0f2c5ec60982c0331bd26d866c9cd996`

`DETECTION80_LOCAL_AUDIT.json` records the completed full-backup audit: all 2,855
archived files and 6,545 calibration dependency files were hash-verified locally,
including all 20 test adapters. Independent CPU checks recomputed candidate scores,
selection checks, all 15 MC caches, 72,000 response seed bindings, 720 panel decisions,
and all 12 summaries. `DETECTION80_CORE_AUDIT.json` remains as the earlier core-only
checkpoint. GPU logits and activation features were not recomputed locally.

## Code and reproduction

The repository's `scripts/run_7b_fingerprint_comparison.py` contains the scientific
pipeline; `scripts/fingerprint_comparison_stats.py` implements the frozen summaries.
The calibration scripts published with this commit reproduce the 400/100/300 design.
`execution/single_postprocess_v1.py` is the actual single-GPU downstream entrypoint;
it imports the frozen run's code snapshot and filters candidates by SCOPE_80.

The archived execution scripts expect `/root/autodl-tmp/token-integrity`, its `env`,
`models/qwen7b`, `runs/calibration-7b-100-geo-v1`, and the restored run/code_snapshot.
They are resume scripts, **not a fresh-install command**. Do not run the original
pipeline's `main` blindly: it contains the historical 300-source generation loop.
For a fresh replication, provision the pinned model/environment, calibration adapters,
and code snapshot, then use the documented active scope. Do not mix historical responses.

Quick CPU checks and plot regeneration, from repository root:

```bash
python -m pytest tests/test_mcc.py tests/test_resf_token.py -q
python reproducibility/7b_mcc_vs_top_20261003/verify_core_results.py
python reproducibility/7b_mcc_vs_top_20261003/figures/gen_fig_detection80.py
```

`execution/audit_detection80.py` is the full-archive audit, requiring the complete
archive, manifest, saved pre-test fingerprints and calibration backup in the original
workspace layout. It is distinct from the lightweight published-evidence check.

The observational 40/40 result has a Wilson 95% lower bound of about 91.2%; 0/20
false alarms has an upper bound of about 16.1%. These data do not establish population
TPR >=95% and FPR <=5%, nor validate 32B models or arbitrary decoding conditions.
