from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config
from llm_integrity.io import read_jsonl
from llm_integrity.modeling import load_model, render_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a controlled LoRA-modified model for integrity tests")
    parser.add_argument("--config", default=str(ROOT / "configs/main_qwen_1.5b.yaml"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", default="results/adapters/lora_weak")
    args = parser.parse_args()
    config = load_config(args.config)
    bundle = load_model(config["model"])
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    rows = read_jsonl(project_path(config["data"]["candidate"]))[:200]
    texts = [render_prompt(bundle.tokenizer, row["prompt"]) + str(row.get("expected_answer", "")) for row in rows]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return bundle.tokenizer(batch["text"], truncation=True, max_length=256)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(bundle.model, lora)
    output = project_path(args.output)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output),
            max_steps=args.steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(bundle.tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(output)
    print(json.dumps({"adapter": str(output), "steps": args.steps}, ensure_ascii=False))


if __name__ == "__main__":
    main()
