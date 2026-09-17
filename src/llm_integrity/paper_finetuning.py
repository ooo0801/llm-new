from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .modeling import load_model, render_prompt


@dataclass(frozen=True)
class LoraTrainingReport:
    variant_id: str
    family: str
    method: str
    base_model: str
    adapter_path: str
    data_path: str
    data_sha256: str
    training_rows: int
    requested_steps: int
    completed_steps: int
    rank: int
    alpha: int
    dropout: float
    learning_rate: float
    target_scope: str
    target_modules: tuple[str, ...]
    seed: int
    max_length: int
    trainable_parameters: int
    total_parameters: int
    trainable_ratio: float
    training_loss: float | None


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)

    rows = [
        json.loads(line)
        for line in source.open(encoding="utf-8")
        if line.strip()
    ]

    if not rows:
        raise ValueError(
            f"LoRA training dataset is empty: {source}"
        )

    return rows


def _sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()

    with source.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def _answer_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (list, dict)):
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    return str(value).strip()


def target_modules_for_scope(
    target_scope: str,
) -> tuple[str, ...]:
    scope = target_scope.strip().lower()

    attention_modules = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )

    ffn_modules = (
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    if scope == "attention":
        return attention_modules

    if scope in {
        "ffn",
        "mlp",
    }:
        return ffn_modules

    if scope in {
        "attention_ffn",
        "all_linear",
    }:
        return attention_modules + ffn_modules

    raise ValueError(
        "Unsupported LoRA target_scope: "
        f"{target_scope}"
    )


def _validate_manifest_variant(
    variant: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    family = str(variant.get("family", "")).lower()

    if family != "finetuning":
        raise ValueError(
            "Expected a finetuning manifest variant, "
            f"received family={family!r}"
        )

    configuration = dict(
        variant.get("configuration", {})
    )

    method = str(
        configuration.get("method", "")
    ).lower()

    if method != "lora":
        raise ValueError(
            "Only real LoRA fine-tuning is supported, "
            f"received method={method!r}"
        )

    variant_id = str(
        variant.get("variant_id", "")
    ).strip()

    if not variant_id:
        raise ValueError(
            "Manifest variant has no variant_id"
        )

    return variant_id, configuration


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch
    from transformers import set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)


def _build_supervised_dataset(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    max_length: int,
    system_prompt: str | None,
):
    from datasets import Dataset

    records: list[dict[str, list[int]]] = []

    for row in rows:
        prompt = str(
            row.get("prompt", "")
        ).strip()

        answer = _answer_text(
            row.get("expected_answer")
        )

        if not prompt or not answer:
            continue

        rendered_prompt = render_prompt(
            tokenizer,
            prompt,
            system_prompt,
        )

        eos_token = tokenizer.eos_token or ""
        full_text = (
            rendered_prompt
            + answer
            + eos_token
        )

        prompt_encoding = tokenizer(
            rendered_prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

        full_encoding = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

        input_ids = list(
            full_encoding["input_ids"]
        )
        attention_mask = list(
            full_encoding["attention_mask"]
        )
        labels = input_ids.copy()

        prompt_length = min(
            len(prompt_encoding["input_ids"]),
            len(labels),
        )

        labels[:prompt_length] = (
            [-100] * prompt_length
        )

        # 若截断后答案完全消失，则该记录不能产生监督损失。
        if all(label == -100 for label in labels):
            continue

        records.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    if not records:
        raise ValueError(
            "No usable supervised LoRA records "
            "remain after tokenization"
        )

    return Dataset.from_list(records)


def train_lora_manifest_variant(
    model_config: Mapping[str, Any],
    variant: Mapping[str, Any],
    data_path: str | Path = (
        "data/attack_train_lora.jsonl"
    ),
    output_root: str | Path = (
        "results/paper_aligned_qwen_7b/"
        "finetuning_adapters"
    ),
    max_length: int = 256,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    system_prompt: str | None = None,
) -> LoraTrainingReport:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    variant_id, configuration = (
        _validate_manifest_variant(variant)
    )

    seed = int(
        variant.get("seed", 42)
    )
    rank = int(
        configuration.get("rank", 8)
    )
    alpha = int(
        configuration.get("alpha", 2 * rank)
    )
    dropout = float(
        configuration.get("dropout", 0.05)
    )
    learning_rate = float(
        configuration.get(
            "learning_rate",
            5e-5,
        )
    )
    requested_steps = int(
        configuration.get("steps", 50)
    )
    target_scope = str(
        configuration.get(
            "target_scope",
            "attention_ffn",
        )
    )
    target_modules = target_modules_for_scope(
        target_scope
    )

    if rank <= 0:
        raise ValueError(
            "LoRA rank must be positive"
        )

    if alpha <= 0:
        raise ValueError(
            "LoRA alpha must be positive"
        )

    if requested_steps <= 0:
        raise ValueError(
            "LoRA steps must be positive"
        )

    if not 0.0 <= dropout < 1.0:
        raise ValueError(
            "LoRA dropout must satisfy "
            "0 <= dropout < 1"
        )

    _set_seed(seed)

    source = Path(data_path)
    rows = _read_jsonl(source)

    # 再次执行防御性过滤，确保训练记录具有监督答案。
    rows = [
        row
        for row in rows
        if str(row.get("prompt", "")).strip()
        and _answer_text(
            row.get("expected_answer")
        )
    ]

    if not rows:
        raise ValueError(
            "No usable prompt-answer pairs "
            f"in {source}"
        )

    local_model_config = deepcopy(
        dict(model_config)
    )

    # LoRA攻击使用原始浮点模型，不在训练时叠加量化攻击。
    local_model_config["quantization"] = "none"

    bundle = load_model(
        local_model_config
    )

    try:
        dataset = _build_supervised_dataset(
            tokenizer=bundle.tokenizer,
            rows=rows,
            max_length=max_length,
            system_prompt=system_prompt,
        )

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(
            bundle.model,
            lora_config,
        )
        bundle.model = model

        if hasattr(
            model,
            "enable_input_require_grads",
        ):
            model.enable_input_require_grads()

        if hasattr(
            model,
            "gradient_checkpointing_enable",
        ):
            model.gradient_checkpointing_enable()

        if hasattr(model, "config"):
            model.config.use_cache = False

        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        if trainable_parameters <= 0:
            raise RuntimeError(
                "LoRA created no trainable parameters"
            )

        adapter_path = (
            Path(output_root) / variant_id
        )
        adapter_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        use_bf16 = bool(
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )
        use_fp16 = bool(
            torch.cuda.is_available()
            and not use_bf16
        )

        training_arguments = TrainingArguments(
            output_dir=str(adapter_path),
            max_steps=requested_steps,
            per_device_train_batch_size=(
                batch_size
            ),
            gradient_accumulation_steps=(
                gradient_accumulation_steps
            ),
            learning_rate=learning_rate,
            lr_scheduler_type=str(configuration.get("lr_scheduler_type", "linear")),
            weight_decay=0.0,
            warmup_ratio=0.0,
            logging_steps=max(
                1,
                min(10, requested_steps),
            ),
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            gradient_checkpointing=True,
            bf16=use_bf16,
            fp16=use_fp16,
            seed=seed,
            data_seed=seed,
            dataloader_num_workers=0,
            optim="adamw_torch",
        )

        collator = DataCollatorForSeq2Seq(
            tokenizer=bundle.tokenizer,
            model=None,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        )

        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=dataset,
            data_collator=collator,
        )

        training_output = trainer.train()

        model.save_pretrained(
            adapter_path
        )
        bundle.tokenizer.save_pretrained(
            adapter_path
        )

        completed_steps = int(
            training_output.global_step
        )

        training_loss_value = (
            training_output.metrics.get(
                "train_loss"
            )
        )
        training_loss = (
            float(training_loss_value)
            if training_loss_value is not None
            else None
        )

        report = LoraTrainingReport(
            variant_id=variant_id,
            family="finetuning",
            method="lora",
            base_model=str(
                local_model_config["name"]
            ),
            adapter_path=str(adapter_path),
            data_path=str(source),
            data_sha256=_sha256(source),
            training_rows=len(dataset),
            requested_steps=requested_steps,
            completed_steps=completed_steps,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            learning_rate=learning_rate,
            target_scope=target_scope,
            target_modules=target_modules,
            seed=seed,
            max_length=max_length,
            trainable_parameters=(
                trainable_parameters
            ),
            total_parameters=total_parameters,
            trainable_ratio=(
                trainable_parameters
                / total_parameters
            ),
            training_loss=training_loss,
        )

        report_path = (
            adapter_path
            / "training_report.json"
        )
        report_path.write_text(
            json.dumps(
                asdict(report),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        return report

    finally:
        bundle.close()
