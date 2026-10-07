# 7B same-source original-prompt control

This adds the original texts corresponding to the 15 already selected sensitive prompts. The MCC/Top 4/8 panels, 40 attack instances and detection protocol are fixed; no selection or training is repeated. Queries per prompt: 25/50/80.

The sensitive arm is the frozen preceding experiment in `../7b_mcc_vs_top_20261003`. Model/package/adapter hashes and recomputed sensitive reference distributions passed the reuse gate. The original arm uses new reference calibration and 72,000 fresh responses. Local independent audit passed source mapping, complete archive and parent hashes, MC regeneration, all 720 decisions and 12 summaries.

This is a post-hoc, same-selected-source diagnostic, not an independent confirmatory experiment or a comparison against an independently optimized ordinary-prompt selection method. Twenty normal panels are repeated sampling from one model. See the Chinese report for limitations and paired uncertainty.

Core results, reference distributions, MC caches and execution/audit/report scripts are included. Complete raw response archive is backed up locally; SHA256 is recorded in PAIRED_ORIGINAL_LOCAL_AUDIT.json. Model and adapter weights are not duplicated here. Execution imports the preceding run's frozen code snapshot and existing adapters. Audit/report scripts expect the documented local backup layout; they are not standalone downloads of the model.
