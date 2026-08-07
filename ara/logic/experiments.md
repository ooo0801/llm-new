# Experiments

## F1: Multi-seed robust portfolio selection

- Input: 64 pre-development Experiment C candidates.
- Development: three new variants per each of five perturbation families; 128 task endpoints; complete-block micro scoring; equal-family macro scoring.
- Selection: structured-pruning-first robust ranking, unique sources, at least three rows per category, 30 rows total.
- Confirmation: two isolated new variants per family, conditional on development Go.
- Terminal outcome: development No-Go at 29/30 rows.

## G1: Direct stable-component threshold migration

- Conditional on F1 confirmation support.
- Uses unchanged 7B activation thresholds, two repeats, frozen saturation/audit gates, and global-unweighted MCC12.
- Outcome: not run because F1 did not reach confirmation.

## H1: Prompt-stratified MMD

- Conditional on F1 and G1 support.
- Uses ten reference and ten target repetitions, prompt-wise RBF bandwidths, 999 within-stratum permutations, one intact and eleven modified states.
- Outcome: not run because its parent gates were not eligible.
