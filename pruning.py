"""
Pruning Gemma-2-2B (LoRA finetuned) — Multi-GPU DDP
==================================================
Structured magnitude pruning on Linear layers.

Launch:
accelerate launch --config_file accelerate_config.yaml prune.py
"""

import argparse
import os
import torch
import torch.nn.utils.prune as prune
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ============================================================
# Config
# ============================================================
MODEL_NAME = "google/gemma-2-9b-it"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints", "gemma", "final")
PRUNED_DIR = os.path.join(SCRIPT_DIR, "checkpoints", "gemma_pruned")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

PRUNE_RATIO = 0.3  # 30% weights pruned
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 512


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prune_ratio", type=float, default=PRUNE_RATIO)
    return parser.parse_args()


# ============================================================
# Pruning function
# ============================================================
def prune_model(model, amount=0.3):
    """
    Apply structured pruning to all Linear layers
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.ln_structured(
                module,
                name="weight",
                amount=amount,
                n=2,
                dim=0,  # prune output neurons
            )
            prune.remove(module, "weight")  # make permanent
    return model


def count_sparsity(model):
    total, zero = 0, 0
    for p in model.parameters():
        total += p.numel()
        zero += torch.sum(p == 0).item()
    return zero / total


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        print("=" * 60)
        print(f"Pruning ratio: {args.prune_ratio}")
        print("=" * 60)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

    # Load finetuned model
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR,
        torch_dtype=torch.bfloat16,
    )

    model.config.use_cache = False

    # Apply pruning
    model = prune_model(model, args.prune_ratio)

    sparsity = count_sparsity(model)
    if local_rank == 0:
        print(f"Sparsity after pruning: {sparsity * 100:.2f}%")

    # ============================================================
    # Dataset (for evaluation)
    # ============================================================
    val_file = os.path.join(DATA_DIR, "val.jsonl")

    dataset = load_dataset(
        "json",
        data_files={"validation": val_file},
    )

    # ============================================================
    # Eval config
    # ============================================================
    training_args = SFTConfig(
        output_dir=PRUNED_DIR,
        per_device_eval_batch_size=BATCH_SIZE,
        bf16=True,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        eval_dataset=dataset["validation"],
    )

    # Evaluate pruned model
    eval_metrics = trainer.evaluate()

    # Save model
    os.makedirs(PRUNED_DIR, exist_ok=True)
    trainer.save_model(PRUNED_DIR)

    if local_rank == 0:
        tokenizer.save_pretrained(PRUNED_DIR)

        print(f"\n{'=' * 60}")
        print(f"Pruned model saved to {PRUNED_DIR}")
        print(f"Eval loss after pruning: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
