# Architecture

The experiment chain is a gated DAG:

1. F1 development endpoint matrix and frozen balanced selection.
2. F1 independent confirmation only if 30 rows satisfy development selection.
3. G1 activation stability, empirical component universe, and MCC12 only if H-F1 is supported.
4. H1 reference fingerprint and modified-state verification only if H-G1 is supported.
5. Every terminal Go or No-Go is archived with frozen inputs, machine reports, logs, and checksums.
