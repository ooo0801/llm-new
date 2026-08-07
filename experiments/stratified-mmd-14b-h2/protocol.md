# Experiment H2 Protocol: Final Prompt-Stratified MMD Verification

Status: **CONDITIONAL ON H-F2 AND H-G2 SUPPORT; FROZEN BEFORE ANY F2/H2 MODEL ENDPOINT**

**H-H2:** the final G2 MCC12 fingerprint leaves the intact state unflagged, detects at least 9/11 registered modified states, and detects at least one instance in every modified family.

For each prompt, reference responses use ten seeds `2026081000`–`2026081009`; target responses use ten seeds `2026081100`–`2026081109`. Generation samples at temperature 0.7, top-p 0.9, top-k 50, with no system prompt. Frozen surface, semantic and task features use `BAAI/bge-small-zh-v1.5@7999e1d...` on CPU.

The primary statistic is mean within-prompt unbiased RBF MMD using prompt-specific median bandwidth and exactly 999 within-stratum permutations at alpha 0.05. Pooled MMD is an ablation only.

The registered states comprise one intact, two unstructured-pruning, two structured-pruning, three Gaussian-noise, two quantization and two independently trained LoRA instances. H-H2 requires the intact decision, at least 9/11 modified detections, and nonzero detection in all five modified families. Claims are restricted to these fixed registered instances.
