"""QLoRA fine-tune Gemma 4 on a (prompt, teacher_response) JSONL corpus.

Designed to run locally on a single GPU or as a Vertex AI Custom Training job
without modification: input/output paths accept gs:// URIs when the runtime
mounts them via Cloud Storage FUSE (Vertex managed datasets) or local paths.

Usage:
    python tuning/qlora_gemma4.py \
        --train tuning/datasets/cloudrun/v1/train.jsonl \
        --eval  tuning/datasets/cloudrun/v1/eval.jsonl \
        --base  google/gemma-4-2b-it \
        --out   tuning/adapters/cloudrun/v1/ \
        --epochs 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel, Field


class LoraHParams(BaseModel):
    # Defaults match Google's official Gemma 4 QLoRA notebook
    # (ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora):
    # alpha=16 and target_modules="all-linear" outperform the QLoRA-paper
    # alpha=2r / attention-only defaults on Gemma 4's expanded MLP blocks.
    r: int = 16
    alpha: int = 16
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: list[str] = Field(
        default_factory=lambda: ["lm_head", "embed_tokens"]
    )


class TrainHParams(BaseModel):
    # Defaults match the Google Gemma 4 notebook's SFTConfig.
    epochs: int = 3
    lr: float = 5e-5
    batch_size: int = 4
    gradient_accumulation: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_seq_len: int = 2048
    lr_scheduler: str = "constant"
    optim: str = "adamw_torch_fused"
    max_grad_norm: float = 0.3
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    seed: int = 42
    bf16: bool = True


class QuantHParams(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


# ---------- builders ----------

def build_bnb_config(q: QuantHParams):
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=q.load_in_4bit,
        bnb_4bit_quant_type=q.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=getattr(torch, q.bnb_4bit_compute_dtype),
    )


def build_peft_config(l: LoraHParams):
    from peft import LoraConfig
    return LoraConfig(
        r=l.r,
        lora_alpha=l.alpha,
        lora_dropout=l.dropout,
        target_modules=l.target_modules,
        bias=l.bias,
        task_type=l.task_type,
        modules_to_save=l.modules_to_save,
    )


def load_base_model(model_id: str, quant: QuantHParams):
    # Note: Gemma 4 is multimodal. We deliberately load it via
    # AutoModelForCausalLM (text-only) because vLLM dynamic LoRA does NOT
    # support multimodal models. Training and serving must agree on the
    # text-only path. Source: Vertex Llama 3.2 vLLM doc + Gemma 4 model card.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=build_bnb_config(quant),
        device_map="auto",
        attn_implementation="eager",
    )
    return model, tokenizer


def load_jsonl_dataset(train_path: str, eval_path: str):
    from datasets import load_dataset
    return load_dataset(
        "json",
        data_files={"train": train_path, "eval": eval_path},
    )


def build_trainer(
    model,
    tokenizer,
    dataset,
    peft_cfg,
    train: TrainHParams,
    out_dir: str,
):
    from trl import SFTConfig, SFTTrainer
    sft_cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=train.epochs,
        per_device_train_batch_size=train.batch_size,
        per_device_eval_batch_size=train.batch_size,
        gradient_accumulation_steps=train.gradient_accumulation,
        learning_rate=train.lr,
        warmup_ratio=train.warmup_ratio,
        weight_decay=train.weight_decay,
        lr_scheduler_type=train.lr_scheduler,
        optim=train.optim,
        max_grad_norm=train.max_grad_norm,
        bf16=train.bf16,
        logging_steps=train.logging_steps,
        eval_strategy="steps",
        eval_steps=train.eval_steps,
        save_steps=train.save_steps,
        save_total_limit=2,
        seed=train.seed,
        max_seq_length=train.max_seq_len,
        packing=False,
        report_to=[],
    )
    return SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        peft_config=peft_cfg,
        tokenizer=tokenizer,
    )


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train.jsonl (local or gs://)")
    ap.add_argument("--eval", required=True, help="eval.jsonl (local or gs://)")
    ap.add_argument("--base", default="google/gemma-4-2b-it")
    ap.add_argument("--out", required=True, help="adapter output dir (local or gs://)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--gradient-accumulation", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lora = LoraHParams(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    train = TrainHParams(
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        max_seq_len=args.max_seq_len, seed=args.seed,
    )
    quant = QuantHParams()

    out_dir = args.out
    if not out_dir.startswith("gs://"):
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading base model {args.base} in 4-bit NF4…", flush=True)
    model, tokenizer = load_base_model(args.base, quant)

    print(f"[2/4] loading dataset (train={args.train}, eval={args.eval})…", flush=True)
    dataset = load_jsonl_dataset(args.train, args.eval)

    print(f"[3/4] building SFTTrainer (epochs={train.epochs}, "
          f"bs={train.batch_size}x{train.gradient_accumulation}, lr={train.lr})…", flush=True)
    peft_cfg = build_peft_config(lora)
    trainer = build_trainer(model, tokenizer, dataset, peft_cfg, train, out_dir)

    print(f"[4/4] starting training; adapter will be written to {out_dir}", flush=True)
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"done. adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
