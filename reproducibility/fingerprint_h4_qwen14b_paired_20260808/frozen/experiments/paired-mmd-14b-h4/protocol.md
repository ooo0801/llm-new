# Experiment H4 Protocol: Common-Random-Number Paired Verification

Status: **PREREGISTERED AFTER H3 DIAGNOSIS AND BEFORE ANY H4 MODEL ENDPOINT**

H3 remains a No-Go. Its intact false positive was reproduced under reference-only scaling, pooled-symmetric scaling and seed-block MMD. A fresh-process engineering replay reproduced all 240 H3 intact responses exactly, excluding weight, GPU-sharding and generation drift. H4 therefore tests a new claim rather than repairing H3.

**H-H4:** common-random-number pairing of the frozen G3 MCC12 leaves all three independently reloaded intact states unflagged, detects at least 9 of 11 newly instantiated modified states, and detects at least one instance in every modified family.

The byte-frozen G3 MCC12 is reused without prompt or component changes. A new reference bank uses seeds `2026081600` through `2026081609`. Every target state uses those exact same ten seeds. The primary test uses pooled-symmetric feature scaling followed by a paired sign-flip statistic; all 12 prompt differences sharing one generation seed are flipped together. With ten seed blocks, all 1,024 sign patterns are enumerated exactly at alpha 0.05. The legacy prompt-stratified and pooled MMD tests are secondary ablations only.

The registered targets are three fresh intact reloads plus eleven newly seeded modified states: two unstructured-pruning, two structured-pruning, three Gaussian-noise, two quantization and two newly trained LoRA instances. The primary gate requires 3/3 intact correct, at least 9/11 modified detected, and nonzero detection in all five modified families. No state, seed, feature, threshold or decision rule may be replaced after an H4 endpoint is observed.
