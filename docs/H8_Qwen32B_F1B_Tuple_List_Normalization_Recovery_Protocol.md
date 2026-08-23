# H8 Qwen2.5-32B F1-B tuple/list normalization recovery protocol

## Scope and authorization

This amendment implements the user's explicit authorization to recover the interrupted F1-B formal sampling run after a false fail-closed rejection. It does not modify the frozen F1-B runner, original authorization, generation manifest, attack endpoints, model snapshot, seeds, response IDs, prompt order, decoding configuration, frozen detector, or the 12,583-response immutable prefix.

The recovery is bounded to schedule positions 12,583 through 12,719 and to these two pre-registered partitions, in order:

1. `h8f1_final_quantization_09_seed2646328052`
2. `h8f1_final_quantization_10_seed0331532823`

No other response may be generated.

## Defect and semantic repair

The original runner seals the materialization payload as JSON. On resume, the sealed JSON represents `notes` and `quantized_module_examples` as lists, while the live Python `quantization_load_report.to_dict()` represents the same values as tuples. The canonical materialization SHA256 is identical, but direct Python dictionary equality treats tuple and list containers as unequal and stops before generation.

The recovery wrapper performs one operation only: it converts the live materialization identity payload through a deterministic JSON round trip before the original runner compares it with the sealed JSON payload. It requires the canonical identity SHA256 to remain unchanged. Any value, ordering, model-state sketch, quantization configuration, endpoint identity, runtime identity, or canonical hash difference remains a hard failure.

The normalized fields are frozen to:

- `materialization.quantization_load_report.notes`
- `materialization.quantization_load_report.quantized_module_examples`

The original frozen runner is imported and used for model loading, materialization, generation, record validation, same-seed retry, fsync writes, GPU cleanup, and final integrity audit. The wrapper does not implement generation itself.

## Recovery gate

Before each worker starts, the wrapper must:

- fail-closed load the original F1-B authorization and every frozen F1-A/F1-B dependency;
- verify the original frozen identity and model snapshot listing SHA256;
- verify the recovery runner/module/protocol hashes and implementation commit;
- verify strict offline mode and `/root/autodl-tmp/huggingface` cache identity;
- validate the immutable pause snapshot SHA256;
- verify the exact byte hashes of all non-recovery partitions;
- verify the exact first 103 Quantization-09 response and attempt-event lines against the pause snapshot;
- validate all current response records against the frozen schedule, identity, seed, role, and materialization identity;
- require a contiguous global schedule prefix, unique response IDs, and unique generation seeds;
- require zero detector-statistics computation and clean GPUs.

After an interruption, the same gate permits only a longer valid contiguous prefix within the two bounded partitions. Previously successful records remain immutable and are never regenerated.

## Provenance continuity

Quantization-09 already has a sealed partition provenance file from the original execution. The recovery worker preserves its original `execution_commit` field so that existing and newly appended records share one immutable partition provenance hash. The separately hashed recovery authorization and recovery execution report disclose the wrapper commit and the exact repaired schedule range.

Quantization-10 has no pre-recovery response, materialization, or provenance. Its first recovery worker seals the original F1-B materialization/provenance payload before generating its first response. If interrupted, the same JSON-normalized comparison is used on resume.

## Prohibited operations

The recovery must not compute features, MMD, scores, global permutations, FPR, attack detection, or any F1-C statistic. It must not modify measurement/score/detector artifacts, replace endpoints, derive seeds, change decoding, or generate beyond the existing 12,720-response manifest.

After exactly 12,720 valid responses are present, the wrapper writes a compact recovery provenance report and invokes the original frozen F1-B final integrity audit. It then stops.
