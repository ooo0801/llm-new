# Constraints

- Model revision, candidates, seeds, evaluators, attack definitions, tolerances, ranking, quotas, and gates are frozen before their endpoints.
- Negative results are not repaired by lowering thresholds or inserting post-hoc candidates.
- F1 confirmation is conditional on a 30-row development freeze.
- G1 is conditional on H-F1 support; H1 is conditional on H-F1 and H-G1 support.
- Existing Experiment A-E and V1/V2/V6 evidence remains immutable.
- Large LoRA tensors and activation checkpoints may be excluded from compact releases only when registries, realization metadata, logs, and explicit exclusions are retained.
