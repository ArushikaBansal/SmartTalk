"""
Baseline Zero-Shot / Inference Evaluation
==========================================
Loads any HuggingFace causal-LM model, runs it on the smart-home test
set (data/test.jsonl), and reports Exact-Match accuracy plus a few
helper metrics (Valid JSON Rate, Room F1, Device-Value accuracy, mean
latency per sample).

Usage:
    python evaluate_baselines.py --model meta-llama/Meta-Llama-3-8B-Instruct
    python evaluate_baselines.py --model google/gemma-2-2b-it --max_samples 500
    python evaluate_baselines.py --model HuggingFaceTB/SmolLM2-360M-Instruct --int4

The list of baseline models we evaluated for the thesis is kept below
as commented-out lines. Uncomment one or pass via --model.
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

SYSTEM_PROMPT = (
    "You are a smart home assistant for a 2BHK house. "
    "Given a natural language command, output the corresponding device states as a JSON object. "
    "The house has these rooms: bathroom, bedroom, balcony, dining_room, hall, kitchen, living_room, study_room. "
    "Available devices include: ac, blinds, computer, exhaust, fan, geyser, lights, music_system, tv. "
    "Respond ONLY with a valid JSON object, no explanation."
)

# ============================================================
# Baseline models surveyed for the thesis (uncomment one to run)
# ============================================================
BASELINE_MODELS = [
    # ---- Sub-1B edge SLMs ----
    # "HuggingFaceTB/SmolLM2-135M-Instruct",
    # "HuggingFaceTB/SmolLM2-360M-Instruct",
    # "Qwen/Qwen2.5-0.5B-Instruct",
    # "allenai/OLMo-1B-hf",
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # "microsoft/phi-1_5",

    # ---- 1B - 3B small models ----
    # "meta-llama/Llama-3.2-1B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "google/gemma-2-2b-it",
    # "Qwen/Qwen2.5-1.5B-Instruct",
    # "Qwen/Qwen2.5-3B-Instruct",
    # "microsoft/phi-2",
    # "stabilityai/stablelm-2-1_6b-chat",
    # "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # "h2oai/h2o-danube3-4b-chat",

    # ---- 3B - 10B mid-tier baselines ----
    # "meta-llama/Meta-Llama-3-8B-Instruct",
    # "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "microsoft/Phi-3-mini-4k-instruct",
    # "microsoft/Phi-3.5-mini-instruct",
    # "microsoft/Phi-4-mini-instruct",
    # "01-ai/Yi-1.5-6B-Chat",
    # "01-ai/Yi-1.5-9B-Chat",
    # "internlm/internlm2_5-7b-chat",
    # "google/gemma-2-9b-it",
]

# Models that don't accept a "system" role and need it merged into user.
NO_SYSTEM_ROLE = ("gemma", "phi-1", "phi-2", "olmo", "tinyllama")


# ============================================================
# I/O
# ============================================================
def load_test_data(path: str, max_samples=None):
    data = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line.strip())
            data.append(row)
            if max_samples and len(data) >= max_samples:
                break
    return data


def get_input_and_expected(row):
    """test.jsonl rows can be either chat-formatted or raw input/output."""
    if "input" in row and "expected_output" in row:
        return row["input"], row["expected_output"]
    if "messages" in row:
        user_msg = next((m["content"] for m in row["messages"] if m["role"] == "user"), "")
        ref = next((m["content"] for m in row["messages"] if m["role"] == "assistant"), "")
        return user_msg, ref
    raise ValueError(f"Unrecognised row format: {list(row.keys())}")


# ============================================================
# Model loading
# ============================================================
def load_model(model_id: str, int4: bool = False):
    print(f"Loading model: {model_id}  (int4={int4})")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if int4:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tok


# ============================================================
# Generation
# ============================================================
def build_prompt(tok, model_id: str, user_input: str) -> str:
    mid = model_id.lower()
    if any(s in mid for s in NO_SYSTEM_ROLE):
        messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_input}]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Models without a chat template (e.g. raw OLMo base): build a simple prompt.
        return f"{SYSTEM_PROMPT}\n\nUser: {user_input}\nAssistant:"


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int = 256):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000.0
    gen = out[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    return text, dt


# ============================================================
# Metrics
# ============================================================
def normalize_json(s: str):
    if s is None:
        return None
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`").lstrip("json").strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    a, b = s.find("{"), s.rfind("}")
    if 0 <= a < b:
        try:
            obj = json.loads(s[a:b+1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def canonical(d: dict) -> str:
    """Lower-case all string values, sort keys, drop whitespace."""
    def rec(x):
        if isinstance(x, dict):
            return {k: rec(x[k]) for k in sorted(x)}
        if isinstance(x, list):
            return [rec(v) for v in x]
        if isinstance(x, str):
            return x.strip().lower()
        return x
    return json.dumps(rec(d), sort_keys=True)


def evaluate(records):
    n = len(records)
    em = valid = 0
    rt = rfp = rfn = 0
    dvc = dvt = 0
    for r in records:
        pred = normalize_json(r["generated"])
        ref = normalize_json(r["expected"])
        if ref is None:
            continue
        if pred is None:
            rfn += len(set(ref.keys()))
            continue
        valid += 1
        if canonical(pred) == canonical(ref):
            em += 1
        ps, rs = set(pred.keys()), set(ref.keys())
        rt += len(ps & rs); rfp += len(ps - rs); rfn += len(rs - ps)
        for room in ps & rs:
            pd = pred.get(room) if isinstance(pred.get(room), dict) else {}
            rd = ref.get(room) if isinstance(ref.get(room), dict) else {}
            for dev in set(pd) | set(rd):
                dvt += 1
                if str(pd.get(dev, "")).strip().lower() == str(rd.get(dev, "")).strip().lower():
                    dvc += 1
    rp = rt / (rt + rfp) if rt + rfp else 0
    rr = rt / (rt + rfn) if rt + rfn else 0
    rf = 2 * rp * rr / (rp + rr) if rp + rr else 0
    return {
        "total": n,
        "exact_match": em,
        "exact_match_rate": em / n if n else 0,
        "valid_json_rate": valid / n if n else 0,
        "room_precision": rp,
        "room_recall": rr,
        "room_f1": rf,
        "device_value_accuracy": dvc / dvt if dvt else 0,
    }


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id or local path.")
    p.add_argument("--test_file", default=os.path.join(DATA_DIR, "test.jsonl"))
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--int4", action="store_true")
    p.add_argument("--output_dir", default=RESULTS_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Baseline evaluation: {args.model}")
    print(f"Test file: {args.test_file}  (max {args.max_samples} samples)")
    print("=" * 60)

    test_data = load_test_data(args.test_file, args.max_samples)
    model, tok = load_model(args.model, int4=args.int4)

    records, latencies = [], []
    for i, row in enumerate(test_data):
        user_input, expected = get_input_and_expected(row)
        prompt = build_prompt(tok, args.model, user_input)
        gen, ms = generate(model, tok, prompt, args.max_new_tokens)
        latencies.append(ms)
        records.append({"input": user_input, "expected": expected, "generated": gen})
        if (i + 1) % 25 == 0 or i == len(test_data) - 1:
            running_em = sum(
                1 for r in records
                if normalize_json(r["generated"]) is not None
                and normalize_json(r["expected"]) is not None
                and canonical(normalize_json(r["generated"]))
                == canonical(normalize_json(r["expected"]))
            )
            print(f"  [{i+1}/{len(test_data)}] EM so far: "
                  f"{running_em}/{i+1} = {running_em/(i+1)*100:.1f}%  "
                  f"avg latency {sum(latencies)/len(latencies):.0f} ms")

    metrics = evaluate(records)
    metrics["mean_latency_ms"] = sum(latencies) / max(len(latencies), 1)
    metrics["model"] = args.model

    print("\n" + "=" * 60)
    print(f"Results for {args.model}")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s} : {v:.4f}")
        else:
            print(f"  {k:25s} : {v}")

    safe = args.model.replace("/", "__")
    out_json = os.path.join(args.output_dir, f"baseline_{safe}.json")
    out_pred = os.path.join(args.output_dir, f"baseline_{safe}_predictions.jsonl")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_pred, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved metrics  -> {out_json}")
    print(f"Saved predictions -> {out_pred}")


if __name__ == "__main__":
    main()
