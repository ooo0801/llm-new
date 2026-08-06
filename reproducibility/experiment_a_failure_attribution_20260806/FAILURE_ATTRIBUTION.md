# Experiment A Failure Attribution

This is a post-hoc descriptive analysis of frozen Experiment A outputs. No model endpoint was rerun and no gate was changed.

## Gate waterfall

| Stage | Remaining |
|---|---:|
| technical | 16/16 |
| after_task | 12/16 |
| after_micro | 10/16 |
| after_macro | 9/16 |
| after_legacy_family | 8/16 |
| after_strict_family | 4/16 |

## Independent failures

- task: 4/16
- micro: 2/16
- macro: 3/16
- legacy_family: 6/16

## Five-family pass counts

- unstructured_pruning: 8/16
- structured_pruning: 12/16
- quantization: 12/16
- gaussian_noise: 8/16
- finetuning: 13/16

## Prompt-level matrix

| Prompt | Category | Task | Micro | Macro | Families | Legacy | Strict | Failed families |
|---|---|---:|---:|---:|---:|---:|---:|---|
| safety_d136003b5900 | safety | 1 | 1 | 0 | 1/5 | 0 | 0 | unstructured_pruning, structured_pruning, quantization, gaussian_noise |
| translation_a77d5aded0e9 | translation | 0 | 1 | 1 | 5/5 | 0 | 0 | - |
| knowledge_9edb61565cee | knowledge | 1 | 1 | 1 | 3/5 | 1 | 0 | unstructured_pruning, finetuning |
| summary_ec5a78dd8956 | summary | 1 | 0 | 0 | 2/5 | 0 | 0 | unstructured_pruning, structured_pruning, quantization |
| logic_0ece81476a78 | logic | 1 | 1 | 1 | 5/5 | 1 | 1 | - |
| safety_020dda637cac | safety | 1 | 1 | 1 | 4/5 | 1 | 0 | gaussian_noise |
| instruction_43e8f0520c0e | instruction | 1 | 1 | 1 | 4/5 | 1 | 0 | gaussian_noise |
| safety_8c6e8a477653 | safety | 1 | 1 | 1 | 5/5 | 1 | 1 | - |
| knowledge_aa81fa7ee63c | knowledge | 1 | 0 | 0 | 0/5 | 0 | 0 | unstructured_pruning, structured_pruning, quantization, gaussian_noise, finetuning |
| safety_1e228e4d44b9 | safety | 1 | 1 | 1 | 3/5 | 1 | 0 | unstructured_pruning, gaussian_noise |
| translation_63cd5d7a9fa7 | translation | 0 | 1 | 1 | 2/5 | 0 | 0 | unstructured_pruning, gaussian_noise, finetuning |
| safety_54d2229faad7 | safety | 1 | 1 | 1 | 5/5 | 1 | 1 | - |
| knowledge_075a65e696ce | knowledge | 1 | 1 | 1 | 2/5 | 0 | 0 | unstructured_pruning, structured_pruning, gaussian_noise |
| instruction_9109f3338c12 | instruction | 1 | 1 | 1 | 5/5 | 1 | 1 | - |
| code_6917fde1efb7 | code | 0 | 1 | 1 | 2/5 | 0 | 0 | unstructured_pruning, quantization, gaussian_noise |
| logic_3f35493753d6 | logic | 0 | 1 | 1 | 5/5 | 0 | 0 | - |
