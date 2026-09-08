"""Versioned, offline audit. Reuses responses; does not change historical results."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import run_h9_stage1_attack_stat_calibration as run

ROOT = run.ROOT
DEST = ROOT / 'results/stage1_audit_20260905'
CONFIGS = [
    ROOT / 'configs/h9_stage1_qwen05b_attack_stat_calibration.yaml',
    ROOT / 'configs/h9_stage1_qwen05b_gaussian_escalation.yaml',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    configs = [run.load_config(p) for p in CONFIGS]
    sources = [run.output_root(c) for c in configs]
    prompts = run.read_jsonl(run.resolve_path(configs[0]['inputs']['prompts']))
    bank = {}
    checks = []
    for source, cfg in zip(sources, configs):
        for name in ['intact'] + [e['endpoint_id'] for e in run.attack_endpoints(cfg)]:
            path = source / 'responses' / (name + '.jsonl')
            rows = run.read_jsonl(path)
            n = 48 if name == 'intact' else 24
            keys = [(r['prompt_id'], r['response_index']) for r in rows]
            expected = {(p['id'], i) for p in prompts for i in range(n)}
            errors = []
            if len(rows) != len(expected) or len(set(keys)) != len(rows) or set(keys) != expected:
                errors.append('missing/duplicate/unexpected response keys')
            for row in rows:
                if hashlib.sha256(row['response'].encode()).hexdigest() != row['response_sha256']:
                    errors.append('response text hash mismatch')
                if row['generation_seed'] != 910000 + row['prompt_index'] * 1000 + row['response_index']:
                    errors.append('generation seed mismatch')
                if row['endpoint_id'] != name:
                    errors.append('endpoint ID mismatch')
            checks.append({'file': str(path), 'rows': len(rows), 'sha256': digest(path), 'errors': errors})
            if errors:
                raise RuntimeError(checks[-1])
            bank[str(path)] = rows
    run.write_json(DEST / 'integrity.json', {'status': 'PASS', 'files': checks})
    texts = sorted({r['response'] for rows in bank.values() for r in rows})
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(configs[0]['semantic_encoder']['name'],
                                  revision=configs[0]['semantic_encoder']['revision'],
                                  device='cuda', local_files_only=True)
    embedding_path = DEST / 'canonical_semantic_embeddings.npy'
    embeddings = np.asarray(encoder.encode(texts, batch_size=32, normalize_embeddings=True), dtype=np.float64)
    np.save(embedding_path, embeddings)
    run.write_json(DEST / 'canonical_texts.json', texts)
    mapping = {text: index for index, text in enumerate(texts)}

    # Test existing encoder's batch invariance on identical deterministic responses.
    probe = ['1,2,3,4,5,6'] * 40 + texts[:50] + ['1,2,3,4,5,6'] * 24
    probe_vectors = np.asarray(encoder.encode(probe, batch_size=32, normalize_embeddings=True), dtype=np.float64)
    same = probe_vectors[[i for i, t in enumerate(probe) if t == '1,2,3,4,5,6']]
    run.write_json(DEST / 'embedding_batch_audit.json', {
        'identical_text': '1,2,3,4,5,6',
        'distinct_vectors_for_identical_text': int(len(np.unique(same, axis=0))),
        'maximum_coordinate_difference': float(np.max(np.abs(same - same[0]))),
        'canonical_unique_text_count': len(texts),
        'canonical_embedding_sha256': digest(embedding_path),
        'rule': 'One cached embedding per exact response text; shared by both audit analyses.',
    })

    def cached_semantic(self, values):
        return embeddings[[mapping[t] for t in values]]

    run.FeatureExtractor._semantic = cached_semantic
    reports = []
    for label, source, cfg in zip(['primary', 'extension'], sources, configs):
        target = DEST / label
        target.mkdir(exist_ok=True)
        for folder in ['responses', 'audits']:
            link = target / folder
            if not link.exists():
                link.symlink_to(source / folder, target_is_directory=True)
        corrected = copy.deepcopy(cfg)
        corrected['output']['root'] = str(target)
        corrected['audit'] = {
            'parent_config_sha256': digest(CONFIGS[len(reports)]),
            'script_sha256': digest(Path(__file__)),
            'runner_sha256': digest(Path(run.__file__)),
            'canonical_embedding_sha256': digest(embedding_path),
            'interpretation': 'Exploratory calibration only; no independent FPR or confirmatory power estimate.',
        }
        frozen_path = target / 'audit_config.json'
        run.write_json(frozen_path, corrected)
        report = run.analyze(corrected, frozen_path)
        reports.append(report)
        # Strict seed robustness, alongside original majority/median aggregation.
        rows = run.read_jsonl(target / 'analysis/prompt_endpoint_results.jsonl')
        nested = []
        for family in ['gaussian', 'lora']:
            for strength in ['weak', 'medium', 'strong']:
                per_prompt = []
                for prompt in prompts:
                    group = [r for r in rows if (r['family'], r['strength'], r['prompt_id']) ==
                             (family, strength, prompt['id'])]
                    values = [r['raw_composite_z'] for r in group if r['raw_composite_z'] is not None]
                    per_prompt.append({
                        'prompt_id': prompt['id'],
                        'raw_evaluable_seeds': len(values),
                        'raw_z_q25_across_seeds': float(np.quantile(values, .25)) if values else None,
                        'raw_detected_all_three_seeds': all(r['raw_detected'] is True for r in group),
                        'max_task_drop_across_seeds': max(r['task_pass_drop'] for r in group),
                    })
                nested.append({'family': family, 'strength': strength, 'prompts': per_prompt,
                               'raw_all_three_seed_detected_count': sum(p['raw_detected_all_three_seeds'] for p in per_prompt)})
        run.write_json(target / 'analysis/seed_robustness.json', nested)
    a = run.read_jsonl(DEST / 'primary/analysis/prompt_endpoint_results.jsonl')
    b = run.read_jsonl(DEST / 'extension/analysis/prompt_endpoint_results.jsonl')
    map_b = {(r['endpoint_id'], r['prompt_id']): r for r in b if r['family'] == 'lora'}
    differences = []
    for row in a:
        if row['family'] == 'lora':
            other = map_b[(row['endpoint_id'], row['prompt_id'])]
            for key in ['h8_mmd2', 'h8_mmd_z', 'raw_composite_z']:
                if row[key] is not None and other[key] is not None:
                    differences.append(abs(row[key] - other[key]))
    run.write_json(DEST / 'shared_data_invariance.json', {
        'status': 'PASS' if max(differences) == 0 else 'FAIL',
        'max_absolute_statistic_difference_on_reused_lora_data': max(differences),
    })
    summary = {
        'data_integrity': 'PASS',
        'new_generation_responses': 8352,
        'model_generation_rerun': False,
        'primary': [{k: v for k, v in s.items() if k != 'per_prompt'} for s in reports[0]['strength_summaries']],
        'extension': [{k: v for k, v in s.items() if k != 'per_prompt'} for s in reports[1]['strength_summaries']],
        'decisions': [r['family_decisions'] for r in reports],
        'null_audit': reports[0]['null_calibration_audit'],
        'limitations': [
            'Null fit/evaluation splits recycle the same 48 intact responses; no independent FPR validation.',
            'Scaler and bandwidth fit on intact bank also reused for null/testing; future independent fit bank required.',
            'Raw composite max-z is not a standard-normal effect or a task accuracy statistic.',
            '12 prompts were selected on 32B and not optimized on the 0.5B target.',
            'LoRA training and heldout datasets are the same copy-code task template with different identifiers.',
            'Training steps imply different linear learning-rate schedules; not checkpoints from a single trajectory.',
            'Gaussian implementation casts full FP32 perturbation once to BF16; legacy implementation casts random draws before scaling.',
            'Aggregate task drop can hide severe prompt-specific failures; numeric/JSON/code evaluators are permissive.',
            'Statistical comparison was repaired after inspecting results; treat as exploratory, not preregistered confirmation.',
        ],
    }
    run.write_json(DEST / 'AUDIT_SUMMARY.json', summary)
    print('AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
