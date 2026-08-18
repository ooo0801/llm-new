from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from llm_integrity.h8_precalibration import build_h8_feature_schema
from run_h8_m0gh_scaler_bandwidth import deterministic_feature_extract, feature_qa


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "h8_m0gh_offline_scaler_bandwidth.yaml"
PROTOCOL_PATH = ROOT / "docs" / "H8_Qwen32B_M0GH_Offline_Protocol.md"


def test_m0gh_config_forbids_all_sampling_and_formal_data() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["model_sampling_authorized"] is False
    assert config["candidate_artifacts_only"] is True
    assert all(value is True for value in config["forbidden_operations"].values())
    assert config["input"]["required_data_role"] == "mmd_precalibration_fit_only"
    assert config["input"]["required_response_count"] == 1200
    assert config["input"]["required_prompt_count"] == 12
    assert config["input"]["required_responses_per_prompt"] == 100


def test_m0gh_provenance_distinguishes_code_execution_and_archive_commits() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    provenance = config["provenance"]
    assert provenance["sampling_code_commit"].startswith("0360e70")
    assert provenance["sampling_execution_commit"].startswith("b5bbde8")
    assert provenance["sampling_archive_commit"].startswith("c67fa06")
    assert len(set((provenance["sampling_code_commit"], provenance["sampling_execution_commit"], provenance["sampling_archive_commit"]))) == 3
    protocol = ROOT / provenance["h8_protocol"]
    h8_config = ROOT / provenance["h8_config"]
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == provenance["h8_protocol_sha256"]
    assert hashlib.sha256(h8_config.read_bytes()).hexdigest() == provenance["h8_config_sha256"]


def test_m0gh_runner_contains_no_model_generation_entrypoint() -> None:
    source = (ROOT / "scripts" / "run_h8_m0gh_scaler_bandwidth.py").read_text(encoding="utf-8")
    forbidden = ("load_model(", "generate_one(", ".generate(", "run_fingerprint", "run_integrity")
    assert not any(token in source for token in forbidden)
    assert "FeatureExtractor(" in source
    assert "h8_bandwidth_candidate_stability_suite(" in source


def test_feature_qa_reports_required_per_prompt_diagnostics() -> None:
    schema = build_h8_feature_schema(2)
    records = []
    for index in range(4):
        records.append(
            {
                "prompt_id": "a" if index < 2 else "b",
                "raw_response": "same" if index % 2 == 0 else "different",
                "completion_token_ids": [index % 2, 1],
                "stop_reason": "eos" if index % 2 == 0 else "length",
                "response_token_count_including_eos": index + 2,
            }
        )
    matrix = np.arange(4 * schema.dimension, dtype=np.float64).reshape(4, schema.dimension)
    report = feature_qa(records, matrix, schema, 1e-8)
    assert report["response_count"] == 4
    assert report["prompt_count"] == 2
    for prompt in report["prompts"].values():
        assert prompt["raw_response_unique_count"] == 2
        assert prompt["completion_token_sequence_unique_count"] == 2
        assert prompt["feature_vector_unique_count"] == 2
        assert prompt["feature_dimensions"] == {
            "surface": 11,
            "semantic": 2,
            "task": 5,
            "total": 18,
        }
        assert prompt["nan_value_count"] == 0
        assert prompt["inf_value_count"] == 0
        assert prompt["eos_stop_fraction"] == 0.5
        assert prompt["length_stop_fraction"] == 0.5


def test_deterministic_feature_extract_encodes_each_unique_text_once() -> None:
    class FakeExtractor:
        def __init__(self) -> None:
            self.semantic_calls: list[list[str]] = []

        def transform(
            self,
            texts,
            rows=None,
            include_surface=True,
            include_semantic=True,
            include_task=True,
        ):
            text_list = list(texts)
            if include_semantic:
                self.semantic_calls.append(text_list)
                return np.asarray(
                    [[len(text), len(text) + 0.5] for text in text_list], dtype=np.float64
                )
            return np.asarray(
                [[float(index % 2)] for index, _ in enumerate(text_list)], dtype=np.float64
            )

    records = [
        {"raw_response": "same", "task_metadata": {}, "category": "x"},
        {"raw_response": "other", "task_metadata": {}, "category": "x"},
        {"raw_response": "same", "task_metadata": {}, "category": "x"},
    ]
    extractor = FakeExtractor()
    matrix = deterministic_feature_extract(extractor, records)
    assert extractor.semantic_calls == [["other", "same"]]
    assert np.array_equal(matrix[0, :2], matrix[2, :2])


def test_protocol_freezes_degeneracy_fallback_boundaries() -> None:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "global_degenerate_fallback" in text
    assert "禁止 cross-prompt" in text
    assert "禁止改用" in text
    assert "MMD pseudo trials" in text


def test_m0f_archive_marks_running_progress_as_transient() -> None:
    archive_path = (
        ROOT
        / "reproducibility"
        / "h8_qwen32b_mmd_precalibration_20260818"
        / "m0f"
        / "H8_M0F_CALIBRATION_SAMPLING_ARCHIVE.json"
    )
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    files = archive["result_files"]
    assert files["sampling_progress_classification"] == "transient_snapshot"
    assert files["sampling_progress_status_is_final"] is False
    basis = archive["execution_basis"]
    assert basis["sampling_code_commit"].startswith("0360e70")
    assert basis["sampling_execution_commit"].startswith("b5bbde8")
    assert basis["sampling_archive_commit"].startswith("c67fa06")
