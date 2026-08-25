#!/usr/bin/env python
"""QLoRA fine-tune of the generator on the grounded FinanceBench SFT set.

PORTABLE cloud script — runs where a GPU + CUDA torch exist (Kaggle T4x2 free,
Vast.ai 4090 spot ~$0.20/h, Colab). NOT run on the dev host (no GPU torch, tight
disk). Deps: `pip install -U transformers peft trl bitsandbytes accelerate datasets`.

Pipeline fit: consumes `build_trainset.py` output; produces a LoRA adapter you
merge and export to GGUF (`llama.cpp/convert_hf_to_gguf.py` + `llama-quantize`),
then drop into your engine's `[[model]]` config and re-run `run_full.py` to score
the fine-tuned model on the same FinanceBench harness (the reward signal).

    python train_qlora.py \
        --base Qwen/Qwen3.5-4B \
        --train benchmarks/financebench/trainset/train.jsonl \
        --val   benchmarks/financebench/trainset/val.jsonl \
        --out   out/qwen35-4b-fin-lora --epochs 3

Swap --base for the distill's HF repo once you confirm it ships safetensors;
otherwise fine-tune base Qwen3.5-4B and layer the distill's gains via data.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="HF model id (safetensors) to fine-tune")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val")
    ap.add_argument("--out", default="out/lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    args = ap.parse_args()

    # Imports are inside main so the file parses on the dev host (no torch there);
    # they resolve on the GPU box where deps are installed.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=quant, device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    data_files = {"train": args.train}
    if args.val:
        data_files["validation"] = args.val
    ds = load_dataset("json", data_files=data_files)

    def to_text(ex: dict) -> dict:
        # Render our {"messages":[...]} rows with the model's own chat template so
        # training matches the served prompt exactly.
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}

    ds = ds.map(to_text, remove_columns=ds["train"].column_names)

    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    sft_cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if args.val else "no",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        peft_config=peft_cfg,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"saved LoRA adapter to {args.out}")
    print("next: merge (peft merge_and_unload) -> convert_hf_to_gguf.py -> "
          "llama-quantize Q4_K_M -> engine [[model]] -> run_full.py")


if __name__ == "__main__":
    main()
