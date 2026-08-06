# Experiment B Final Report

- H-B1 strict replication: 3/4 (supported)
- H-B2 legacy replication: 6/8 (supported)
- Task preserved: 8/8
- Positive micro: 7/8
- Positive macro: 7/8

Global-magnitude pruning and NF4 are deterministic fresh-materialization repeats; the other three attack families use independent seeds.

## Prompt results

| Prompt | A strict | Task | Micro | Macro | Families | Legacy | Strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| knowledge_9edb61565cee | 0 | 1 | 1 | 0 | 2/5 | 0 | 0 |
| logic_0ece81476a78 | 1 | 1 | 1 | 1 | 4/5 | 1 | 0 |
| safety_020dda637cac | 0 | 1 | 0 | 1 | 5/5 | 0 | 0 |
| instruction_43e8f0520c0e | 0 | 1 | 1 | 1 | 5/5 | 1 | 1 |
| safety_8c6e8a477653 | 1 | 1 | 1 | 1 | 5/5 | 1 | 1 |
| safety_1e228e4d44b9 | 0 | 1 | 1 | 1 | 4/5 | 1 | 0 |
| safety_54d2229faad7 | 1 | 1 | 1 | 1 | 5/5 | 1 | 1 |
| instruction_9109f3338c12 | 1 | 1 | 1 | 1 | 5/5 | 1 | 1 |
