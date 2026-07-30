from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .io import write_json


@dataclass
class FingerprintEntry:
    prompt_id: str
    prompt: str
    category: str | None = None
    sensitivity: float | None = None
    components: list[str] = field(default_factory=list)
    reference_responses: list[str] = field(default_factory=list)
    reference_response_seeds: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelFingerprint:
    model_name: str
    model_revision: str
    selection_method: str
    entries: list[FingerprintEntry]
    generation_config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.1"

    def save(self, path: str | Path) -> None:
        write_json(path, asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> "ModelFingerprint":
        import json

        value = json.loads(Path(path).read_text(encoding="utf-8"))
        value["entries"] = [FingerprintEntry(**entry) for entry in value["entries"]]
        return cls(**value)
