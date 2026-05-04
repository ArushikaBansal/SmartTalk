"""
Finetuning sub-3B models on Smart Home Commands (Multi-GPU DDP)
================================================================
Full-precision LoRA with DDP across 8 GPUs.

Launch: accelerate launch --config_file accelerate_config.yaml finetune.py
"""

import argparse
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "Efficient-Large-Model/gemma-2-2b-it"  # ungated mirror of google/gemma-2-2b-it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "checkpoints", "gemma")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Training hyperparameters — 8x B200 GPUs
NUM_EPOCHS = 2
BATCH_SIZE = 16              # per GPU
GRAD_ACCUM_STEPS = 2         # effective batch = 16 * 8 GPUs * 2 = 256
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"

# LoRA hyperparameters
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=-1, help="Override max training steps (for testing)")
    parser.add_argument("--data_suffix", type=str, default="", help="Suffix for data files, e.g. '_tiny' for train_tiny.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        print("=" * 60)
        print("Gemma-2-2B-IT LoRA Finetuning (8-GPU DDP)")
        print(f"Effective batch size: {BATCH_SIZE} x 8 x {GRAD_ACCUM_STEPS} = {BATCH_SIZE * 8 * GRAD_ACCUM_STEPS}")
        if args.max_steps > 0:
            print(f"TEST MODE: max_steps={args.max_steps}, data_suffix='{args.data_suffix}'")
        print("=" * 60)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Model — full bf16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Gemma 2 needs eager attention
    )
    model.config.use_cache = False

    # 3. LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 4. Dataset
    # Gemma 2 does NOT support a "system" role in its chat template.
    # We merge the system prompt into the first user message.
    def merge_system_into_user(example):
        messages = example["messages"]
        new_messages = []
        system_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                if system_content:
                    msg = dict(msg)  # copy
                    msg["content"] = system_content + "\n\n" + msg["content"]
                    system_content = ""
                new_messages.append(msg)
            else:
                new_messages.append(msg)
        example["messages"] = new_messages
        return example

    train_file = os.path.join(DATA_DIR, f"train{args.data_suffix}.jsonl")
    val_file = os.path.join(DATA_DIR, f"val{args.data_suffix}.jsonl")
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": val_file,
        },
    )
    dataset = dataset.map(merge_system_into_user)
    if local_rank == 0:
        print(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")

    # 5. Training config
    is_test = args.max_steps > 0
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        max_steps=args.max_steps,  # -1 means use num_train_epochs
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=50 if not is_test else 0,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        logging_steps=1 if is_test else 10,
        eval_strategy="no" if is_test else "steps",
        eval_steps=200,
        save_strategy="no" if is_test else "steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=False,
        # metric_for_best_model="eval_loss",
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        peft_config=lora_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    if local_rank == 0:
        trainer.model.print_trainable_parameters()

    # Train
    train_result = trainer.train()

    # Save model (all ranks must participate)
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    if local_rank == 0:
        tokenizer.save_pretrained(final_dir)

    # Evaluate (all ranks must participate)
    eval_metrics = trainer.evaluate()

    # Logging and plots (rank 0 only)
    if local_rank == 0:
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        # Plot training loss vs steps
        log_history = trainer.state.log_history
        train_steps = [h["step"] for h in log_history if "loss" in h]
        train_losses = [h["loss"] for h in log_history if "loss" in h]
        eval_steps_plot = [h["step"] for h in log_history if "eval_loss" in h]
        eval_losses_plot = [h["eval_loss"] for h in log_history if "eval_loss" in h]

        results_dir = os.path.join(SCRIPT_DIR, "results")
        os.makedirs(results_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_losses, label="Train Loss", color="#2196F3", linewidth=1.5)
        if eval_losses_plot:
            plt.plot(eval_steps_plot, eval_losses_plot, label="Eval Loss", color="#FF5722", linewidth=1.5, marker="o", markersize=4)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Gemma-2-2B LoRA Finetuning — Training Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "gemma_training_loss.png"), dpi=150)
        plt.close()
        print(f"  ✓ Training loss plot saved to results/gemma_training_loss.png")

        print(f"\n{'=' * 60}")
        print(f"Training complete! Model saved to {final_dir}")
        print(f"Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
        print(f"Eval loss:  {eval_metrics.get('eval_loss', 'N/A'):.4f}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
