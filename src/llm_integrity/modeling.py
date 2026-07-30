from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    name: str
    device: Any
    revision: str = "main"

    def close(self) -> None:
        del self.model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _torch_dtype(name: str):
    import torch

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "auto": "auto",
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def load_model(model_config: Mapping[str, Any]) -> ModelBundle:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    name = str(model_config["name"])
    revision = str(model_config.get("revision", "main"))
    quantization = str(model_config.get("quantization", "none")).lower()
    kwargs: dict[str, Any] = {
        "revision": revision,
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "low_cpu_mem_usage": True,
        "device_map": model_config.get("device_map", "auto"),
        "torch_dtype": _torch_dtype(str(model_config.get("dtype", "bfloat16"))),
    }
    if bool(model_config.get("eager_attention", False)):
        kwargs["attn_implementation"] = "eager"
    if quantization in {"int8", "8bit"}:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization in {"int4", "4bit"}:
        compute_dtype = _torch_dtype(str(model_config.get("compute_dtype", "bfloat16")))
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=str(model_config.get("quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(model_config.get("double_quant", True)),
        )
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        revision=revision,
        trust_remote_code=kwargs["trust_remote_code"],
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    device = next(model.parameters()).device
    if torch.cuda.is_available() and device.type == "cpu" and kwargs["device_map"] is None:
        model.to("cuda")
        device = next(model.parameters()).device
    return ModelBundle(model=model, tokenizer=tokenizer, name=name, device=device, revision=revision)


def render_prompt(tokenizer: Any, prompt: str, system_prompt: str | None = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def tokenize_prompts(
    bundle: ModelBundle,
    prompts: Iterable[str],
    max_length: int = 512,
    system_prompt: str | None = None,
):
    texts = [render_prompt(bundle.tokenizer, p, system_prompt) for p in prompts]
    encoded = bundle.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {key: value.to(bundle.device) for key, value in encoded.items()}


def generate_texts(
    bundle: ModelBundle,
    prompts: list[str],
    generation_config: Mapping[str, Any],
    seed: int | None = None,
) -> list[str]:
    import torch

    if seed is not None:
        from .io import set_seed

        set_seed(seed)

    encoded = tokenize_prompts(
        bundle,
        prompts,
        int(generation_config.get("max_input_tokens", 512)),
        generation_config.get("system_prompt"),
    )
    input_length = encoded["input_ids"].shape[1]
    do_sample = bool(generation_config.get("do_sample", True))
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation_config.get("max_new_tokens", 128)),
        "do_sample": do_sample,
        "pad_token_id": bundle.tokenizer.pad_token_id,
        "eos_token_id": bundle.tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs.update(
            temperature=float(generation_config.get("temperature", 0.7)),
            top_p=float(generation_config.get("top_p", 0.9)),
            top_k=int(generation_config.get("top_k", 50)),
        )
    with torch.inference_mode():
        generated = bundle.model.generate(**encoded, **kwargs)
    return bundle.tokenizer.batch_decode(generated[:, input_length:], skip_special_tokens=True)


def next_token_logits(bundle: ModelBundle, prompts: list[str], max_length: int = 512):
    import torch

    encoded = tokenize_prompts(bundle, prompts, max_length=max_length)
    with torch.inference_mode():
        outputs = bundle.model(**encoded, use_cache=False, return_dict=True)

    attention_mask = encoded["attention_mask"].bool()
    token_positions = torch.arange(
        attention_mask.shape[1],
        device=attention_mask.device,
    ).unsqueeze(0).expand_as(attention_mask)
    last_positions = token_positions.masked_fill(
        ~attention_mask,
        -1,
    ).max(dim=1).values
    row = torch.arange(
        outputs.logits.shape[0],
        device=outputs.logits.device,
    )
    return outputs.logits[row, last_positions].float().cpu()



def model_metadata(bundle: ModelBundle) -> dict[str, Any]:
    config = bundle.model.config
    return {
        "name": bundle.name,
        "revision": bundle.revision,
        "model_type": getattr(config, "model_type", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "dtype": str(next(bundle.model.parameters()).dtype),
        "device": str(bundle.device),
    }
