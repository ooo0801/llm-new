# H8 D2-C Development Comparison

The frozen rule selected `r60_q10_top2` uniquely. Counts below are development evidence, not formal FPR or TPR estimates.

| Configuration | Intact FP count (of 5) | Minimum family detections (of 2) | Total detected endpoints (of 8) | Gaussian | Pruning | LoRA | Quantization |
|---|---:|---:|---:|---:|---:|---:|---:|
| r40_q10_top2 | 0 | 0 | 5 | 1 | 2 | 0 | 2 |
| r40_q10_top3 | 1 | 0 | 5 | 1 | 2 | 0 | 2 |
| r40_q10_top4 | 1 | 0 | 5 | 1 | 2 | 0 | 2 |
| r40_q20_top2 | 0 | 0 | 4 | 0 | 2 | 0 | 2 |
| r40_q20_top3 | 0 | 0 | 4 | 0 | 2 | 0 | 2 |
| r40_q20_top4 | 0 | 0 | 4 | 0 | 2 | 0 | 2 |
| r60_q10_top2 | 0 | 1 | 6 | 1 | 2 | 1 | 2 |
| r60_q10_top3 | 1 | 0 | 5 | 1 | 2 | 0 | 2 |
| r60_q10_top4 | 1 | 0 | 5 | 1 | 2 | 0 | 2 |
| r60_q20_top2 | 0 | 0 | 5 | 1 | 2 | 0 | 2 |
| r60_q20_top3 | 1 | 0 | 5 | 1 | 2 | 0 | 2 |
| r60_q20_top4 | 1 | 0 | 5 | 1 | 2 | 0 | 2 |

Primary proof: `reproducibility/h8_qwen32b_detector_development_20260820/d2c_detector_frozen_v1/D2C_12_CONFIGURATION_COMPARISON.json`.
