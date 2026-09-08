# H9 Stage 1 Gaussian escalation extension

The primary frozen Gaussian grid (`0.0005`, `0.0015`, `0.003`) produced strictly
increasing realized BF16 weight changes but did not reach the predeclared response
sensitivity gates. This versioned adaptive calibration extension therefore evaluates
`0.006`, `0.012`, and `0.024`, again with three attack seeds.

The primary report remains immutable. This extension reuses its intact bank and LoRA
artifacts read-only, uses new Gaussian endpoint files under a distinct output root, and
keeps the original statistical and task-integrity thresholds unchanged. It is an
attack-calibration extension, not an independent confirmatory test; the selected level
must later be frozen before H6 optimization experiments.

