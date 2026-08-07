# Experiment H3 Protocol: Recovery MCC12 Prompt-Stratified MMD Verification

Status: **CONDITIONAL ON H-G3 SUPPORT; PREREGISTERED BEFORE ANY G3/H3 MODEL ENDPOINT**

**H-H3:** the final G3 recovery MCC12 leaves the intact state unflagged, detects at least 9 of 11 registered modified states, and detects at least one instance in every modified family.

For each selected prompt, reference responses use ten seeds `2026081300` through `2026081309`; target responses use ten seeds `2026081400` through `2026081409`. Generation samples at temperature 0.7, top-p 0.9 and top-k 50 with no system prompt. Frozen surface, semantic and task features use `BAAI/bge-small-zh-v1.5@7999e1d3359715c523056ef9478215996d62a620` on CPU.

The primary statistic is mean within-prompt unbiased RBF MMD using prompt-specific median bandwidth and exactly 999 within-stratum permutations at alpha 0.05. Pooled MMD is an ablation only.

The registered states comprise one intact, two unstructured-pruning, two structured-pruning, three Gaussian-noise, two quantization and two independently trained LoRA instances. H-H3 requires the intact decision to be correct, at least 9/11 modified detections, and nonzero detection in all five modified families. Claims are restricted to these fixed registered instances. A failed state or family gate is reported without attack replacement, seed changes or threshold repair.
