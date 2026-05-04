"""
Evaluation Script for Finetuned Smart Home Models
===================================================
Evaluates finetuned LLaMA and Gemma models on the test set.
Metrics:
  - Exact JSON Match: output == expected (after normalization)
  - Valid JSON Rate: % of outputs that are valid JSON
  - Room-level F1: precision/recall/F1 of predicted rooms
  - Device-Value Accuracy: % of device-value pairs correctly predicted
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEM_PROMPT = (
    "You are a smart home assistant for a 2BHK house. "
    "Given a natural language command, output the corresponding device states as a JSON object. "
    "The house has these rooms: bathroom, bedroom, balcony, dining_room, hall, kitchen, living_room, study_room. "
    "Available devices include: ac, blinds, computer, exhaust, fan, geyser, lights, music_system, tv. "
    "Respond ONLY with a valid JSON object, no explanation."
)

# Model configs
MODEL_CONFIGS = {
    "llama": {
        "base_model": "NousResearch/Meta-Llama-3-8B-Instruct",
        "adapter_path": os.path.join(SCRIPT_DIR, "checkpoints", "llama", "final"),
        "attn_impl": "sdpa",
    },
    "gemma": {
        "base_model": "Efficient-Large-Model/gemma-2-2b-it",  # ungated mirror
        "adapter_path": os.path.join(SCRIPT_DIR, "checkpoints", "gemma", "final"),
        "attn_impl": "eager",
    },
}


def load_test_data(path: str, max_samples: int = None) -> list[dict]:
    """Load test data from JSONL."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
            if max_samples and len(data) >= max_samples:
                break
    return data


def load_model_and_tokenizer(model_name: str):
    """Load base model + LoRA adapter."""
    config = MODEL_CONFIGS[model_name]

    tokenizer = AutoTokenizer.from_pretrained(config["adapter_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=config["attn_impl"],
    )

    model = PeftModel.from_pretrained(base_model, config["adapter_path"])
    model.eval()

    return model, tokenizer


def generate_output(model, tokenizer, user_input: str, model_name: str = "llama", max_new_tokens: int = 256) -> str:
    """Generate model output for a given input."""
    if model_name == "gemma":
        # Gemma does not support system role — merge into user message
        messages = [
            {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_input},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


def normalize_json(json_str: str) -> dict | None:
    """Try to parse and normalize JSON output."""
    try:
        # Try direct parse
        obj = json.loads(json_str)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from the response
    try:
        start = json_str.index("{")
        end = json_str.rindex("}") + 1
        obj = json.loads(json_str[start:end])
        if isinstance(obj, dict):
            return obj
    except (ValueError, json.JSONDecodeError):
        pass

    return None


def compute_metrics(predictions: list[dict], references: list[dict]) -> dict:
    """Compute all evaluation metrics."""
    total = len(predictions)
    exact_match = 0
    valid_json = 0
    room_tp, room_fp, room_fn = 0, 0, 0
    device_correct = 0
    device_total = 0

    for pred, ref in zip(predictions, references):
        pred_json = normalize_json(pred["generated"])
        ref_json = normalize_json(ref["expected_output"])

        if ref_json is None:
            continue

        # Valid JSON rate
        if pred_json is not None:
            valid_json += 1

            # Exact match (after normalization)
            if json.dumps(pred_json, sort_keys=True) == json.dumps(ref_json, sort_keys=True):
                exact_match += 1

            # Room-level metrics
            pred_rooms = set(pred_json.keys())
            ref_rooms = set(ref_json.keys())
            room_tp += len(pred_rooms & ref_rooms)
            room_fp += len(pred_rooms - ref_rooms)
            room_fn += len(ref_rooms - pred_rooms)

            # Device-value accuracy (for overlapping rooms)
            for room in pred_rooms & ref_rooms:
                pred_devices = pred_json.get(room, {})
                ref_devices = ref_json.get(room, {})
                # Handle cases where room value is a string/list instead of dict
                if not isinstance(pred_devices, dict):
                    pred_devices = {}
                if not isinstance(ref_devices, dict):
                    ref_devices = {}
                all_devices = set(pred_devices.keys()) | set(ref_devices.keys())
                for device in all_devices:
                    device_total += 1
                    if str(pred_devices.get(device, "")).lower() == str(ref_devices.get(device, "")).lower():
                        device_correct += 1
        else:
            # Count ref rooms as false negatives
            ref_rooms = set(ref_json.keys())
            room_fn += len(ref_rooms)

    # Compute derived metrics
    room_precision = room_tp / (room_tp + room_fp) if (room_tp + room_fp) > 0 else 0
    room_recall = room_tp / (room_tp + room_fn) if (room_tp + room_fn) > 0 else 0
    room_f1 = 2 * room_precision * room_recall / (room_precision + room_recall) if (room_precision + room_recall) > 0 else 0

    return {
        "total_examples": total,
        "exact_match": exact_match,
        "exact_match_rate": exact_match / total if total > 0 else 0,
        "valid_json": valid_json,
        "valid_json_rate": valid_json / total if total > 0 else 0,
        "room_precision": room_precision,
        "room_recall": room_recall,
        "room_f1": room_f1,
        "device_value_accuracy": device_correct / device_total if device_total > 0 else 0,
        "device_correct": device_correct,
        "device_total": device_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned smart home models")
    parser.add_argument("--model", type=str, required=True, choices=["llama", "gemma", "both"])
    parser.add_argument("--test_data", type=str, default=os.path.join(SCRIPT_DIR, "data", "test_raw.jsonl"))
    parser.add_argument("--max_samples", type=int, default=500, help="Max test samples (for speed)")
    parser.add_argument("--output_dir", type=str, default=os.path.join(SCRIPT_DIR, "results"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    print("Loading test data...")
    test_data = load_test_data(args.test_data, args.max_samples)
    print(f"  Loaded {len(test_data)} test examples")

    models_to_eval = ["llama", "gemma"] if args.model == "both" else [args.model]
    all_results = {}

    for model_name in models_to_eval:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {model_name.upper()}")
        print(f"{'=' * 60}")

        # Load model
        print("Loading model...")
        model, tokenizer = load_model_and_tokenizer(model_name)

        # Generate predictions
        print("Generating predictions...")
        predictions = []
        start_time = time.time()

        for i, example in enumerate(tqdm(test_data, desc=f"Evaluating {model_name}")):
            generated = generate_output(model, tokenizer, example["input"], model_name=model_name)
            predictions.append({
                "input": example["input"],
                "expected": example["expected_output"],
                "generated": generated,
            })

        elapsed = time.time() - start_time

        # Compute metrics
        print("Computing metrics...")
        metrics = compute_metrics(predictions, test_data)
        metrics["inference_time_seconds"] = elapsed
        metrics["avg_time_per_example"] = elapsed / len(test_data) if len(test_data) > 0 else 0

        all_results[model_name] = metrics

        # Print results
        print(f"\n--- {model_name.upper()} Results ---")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # Save predictions
        pred_path = os.path.join(args.output_dir, f"{model_name}_predictions.jsonl")
        with open(pred_path, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")
        print(f"  Predictions saved to {pred_path}")

        # Free memory
        del model, tokenizer
        torch.cuda.empty_cache()

    # Save all results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_path}")

    # Generate results.txt
    results_txt_path = os.path.join(args.output_dir, "results.txt")
    with open(results_txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Smart Home Command Finetuning - Evaluation Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test samples evaluated: {len(test_data)}\n\n")

        for model_name, metrics in all_results.items():
            f.write(f"\n{'─' * 50}\n")
            f.write(f"Model: {model_name.upper()}\n")
            f.write(f"Base: {MODEL_CONFIGS[model_name]['base_model']}\n")
            f.write(f"{'─' * 50}\n\n")
            f.write(f"  Exact Match Accuracy:     {metrics['exact_match_rate']*100:.2f}%  ({metrics['exact_match']}/{metrics['total_examples']})\n")
            f.write(f"  Valid JSON Rate:           {metrics['valid_json_rate']*100:.2f}%  ({metrics['valid_json']}/{metrics['total_examples']})\n")
            f.write(f"  Room-level Precision:      {metrics['room_precision']*100:.2f}%\n")
            f.write(f"  Room-level Recall:         {metrics['room_recall']*100:.2f}%\n")
            f.write(f"  Room-level F1:             {metrics['room_f1']*100:.2f}%\n")
            f.write(f"  Device-Value Accuracy:     {metrics['device_value_accuracy']*100:.2f}%  ({metrics['device_correct']}/{metrics['device_total']})\n")
            f.write(f"  Inference Time:            {metrics['inference_time_seconds']:.1f}s ({metrics['avg_time_per_example']:.3f}s/example)\n")
            f.write("\n")

    print(f"Results summary saved to {results_txt_path}")

    # Generate sample_predictions.txt with human-readable examples
    sample_txt_path = os.path.join(args.output_dir, "sample_predictions.txt")
    with open(sample_txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Smart Home Finetuning - Sample Predictions\n")
        f.write("=" * 70 + "\n")

        for model_name in models_to_eval:
            pred_file = os.path.join(args.output_dir, f"{model_name}_predictions.jsonl")
            if not os.path.exists(pred_file):
                continue

            preds = []
            with open(pred_file, "r") as pf:
                for line in pf:
                    preds.append(json.loads(line.strip()))

            f.write(f"\n{'━' * 70}\n")
            f.write(f"  Model: {model_name.upper()} ({MODEL_CONFIGS[model_name]['base_model']})\n")
            f.write(f"{'━' * 70}\n")

            # Pick 10 diverse examples: first 5 + 5 spaced out
            indices = list(range(min(5, len(preds))))
            step = max(1, len(preds) // 5)
            for i in range(5):
                idx = min(5 + i * step, len(preds) - 1)
                if idx not in indices:
                    indices.append(idx)

            for idx in indices:
                p = preds[idx]
                pred_json = normalize_json(p["generated"])
                expected_json = normalize_json(p["expected"])
                match = "✅ MATCH" if (pred_json and expected_json and
                    json.dumps(pred_json, sort_keys=True) == json.dumps(expected_json, sort_keys=True)) else "❌ MISMATCH"

                f.write(f"\n{'─' * 50}\n")
                f.write(f"Example #{idx + 1}  [{match}]\n")
                f.write(f"{'─' * 50}\n")
                f.write(f"INPUT:\n  {p['input']}\n\n")
                f.write(f"EXPECTED:\n  {json.dumps(expected_json, indent=2) if expected_json else p['expected']}\n\n")
                f.write(f"GENERATED:\n  {json.dumps(pred_json, indent=2) if pred_json else p['generated']}\n")

        f.write(f"\n{'=' * 70}\n")

    print(f"Sample predictions saved to {sample_txt_path}")


if __name__ == "__main__":
    main()
