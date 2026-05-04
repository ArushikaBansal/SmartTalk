"""
Informal/Casual Command Evaluation for Smart Home Fine-tuned Models
====================================================================
Tests how fine-tuned LLaMA-3-8B and Gemma-2-2B handle commands that are
outside the training distribution: slang, implicit intent, typos,
abbreviations, ambiguous commands, code-switching, etc.

Usage:
    python evaluate_casual.py --model both --max_new_tokens 256
    python evaluate_casual.py --model llama
    python evaluate_casual.py --model gemma
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEM_PROMPT = (
    "You are a smart home assistant for a 2BHK house. "
    "Given a natural language command, output the corresponding device states as a JSON object. "
    "The house has these rooms: bathroom, bedroom, balcony, dining_room, hall, kitchen, living_room, study_room. "
    "Available devices include: ac, blinds, computer, exhaust, fan, geyser, lights, music_system, tv. "
    "Respond ONLY with a valid JSON object, no explanation."
)

MODEL_CONFIGS = {
    "llama": {
        "base_model": "NousResearch/Meta-Llama-3-8B-Instruct",
        "adapter_path": os.path.join(SCRIPT_DIR, "checkpoints", "llama", "final"),
        "attn_impl": "sdpa",
    },
    "gemma": {
        "base_model": "Efficient-Large-Model/gemma-2-2b-it",
        "adapter_path": os.path.join(SCRIPT_DIR, "checkpoints", "gemma", "final"),
        "attn_impl": "eager",
    },
}

# ============================================================
# Informal / Casual Test Commands
# ============================================================
# Each entry: (command, category, expected_output_or_None, notes)
# expected_output is the "reasonable" JSON a human would expect.
# None means the command is intentionally ambiguous — we just check if output is valid JSON.

CASUAL_COMMANDS = [
    # --- Category 1: Slang / Colloquial ---
    {
        "input": "yo turn off everything im heading out",
        "category": "slang",
        "expected": {
            "bedroom": {"ac": "off", "lights": "off", "fan": "off"},
            "living_room": {"tv": "off", "lights": "off", "fan": "off", "music_system": "off"},
            "kitchen": {"lights": "off", "exhaust": "off", "fan": "off"},
            "hall": {"lights": "off", "tv": "off"},
            "bathroom": {"lights": "off", "geyser": "off", "exhaust": "off"},
            "study_room": {"computer": "off", "lights": "off", "fan": "off"},
            "dining_room": {"lights": "off", "fan": "off", "ac": "off"},
            "balcony": {"lights": "off"},
        },
        "notes": "Should map to leaving/away mode",
    },
    {
        "input": "kill the telly in the hall",
        "category": "slang",
        "expected": {"hall": {"tv": "off"}},
        "notes": "telly = tv, kill = turn off",
    },
    {
        "input": "crank up the AC bro its boiling in the bedroom",
        "category": "slang",
        "expected": {"bedroom": {"ac": "18°C"}},
        "notes": "Should lower AC temp; 'boiling' = very hot",
        "accept_any_ac_on": True,
    },
    {
        "input": "lights out in the living room",
        "category": "slang",
        "expected": {"living_room": {"lights": "off"}},
        "notes": "'lights out' = turn off lights",
    },
    {
        "input": "nah dont need the fan anymore in kitchen",
        "category": "slang",
        "expected": {"kitchen": {"fan": "off"}},
        "notes": "Negation-based off command",
    },

    # --- Category 2: Implicit Intent (no explicit device mentioned) ---
    {
        "input": "its too dark in the bedroom",
        "category": "implicit",
        "expected": {"bedroom": {"lights": "on"}},
        "notes": "Implicit: dark → need lights",
        "accept_any_lights_on": True,
    },
    {
        "input": "i cant see anything in the kitchen",
        "category": "implicit",
        "expected": {"kitchen": {"lights": "bright"}},
        "notes": "Implicit: can't see → lights bright/on",
        "accept_any_lights_on": True,
    },
    {
        "input": "its freezing in here",
        "category": "implicit",
        "expected": None,
        "notes": "Ambiguous: which room? Should it turn off AC or turn on geyser?",
    },
    {
        "input": "im sweating in the dining room",
        "category": "implicit",
        "expected": {"dining_room": {"fan": "on"}},
        "notes": "Sweating → turn on fan or AC",
        "accept_alternatives": [
            {"dining_room": {"ac": "on"}},
            {"dining_room": {"fan": "on", "ac": "on"}},
            {"dining_room": {"ac": "22°C"}},
        ],
    },
    {
        "input": "i wanna watch something",
        "category": "implicit",
        "expected": {"living_room": {"tv": "on"}},
        "notes": "Implicit: watch something → TV on in living room",
    },

    # --- Category 3: Typos / Misspellings ---
    {
        "input": "trun on teh lights in teh bedrrom",
        "category": "typos",
        "expected": {"bedroom": {"lights": "on"}},
        "notes": "Multiple typos: trun→turn, teh→the, bedrrom→bedroom",
    },
    {
        "input": "swich off fan in kichen",
        "category": "typos",
        "expected": {"kitchen": {"fan": "off"}},
        "notes": "swich→switch, kichen→kitchen",
    },
    {
        "input": "actuvate the exhoust in bathrom",
        "category": "typos",
        "expected": {"bathroom": {"exhaust": "on"}},
        "notes": "actuvate→activate, exhoust→exhaust, bathrom→bathroom",
    },
    {
        "input": "plz set ac too 22 in bedrum",
        "category": "typos",
        "expected": {"bedroom": {"ac": "22°C"}},
        "notes": "plz→please, too→to, bedrum→bedroom",
    },

    # --- Category 4: Abbreviations / Short-hand ---
    {
        "input": "AC 22 bedroom",
        "category": "abbreviation",
        "expected": {"bedroom": {"ac": "22°C"}},
        "notes": "Telegram-style terse command",
    },
    {
        "input": "lights off LR",
        "category": "abbreviation",
        "expected": {"living_room": {"lights": "off"}},
        "notes": "LR = living_room",
    },
    {
        "input": "fan on kit + bed",
        "category": "abbreviation",
        "expected": {"kitchen": {"fan": "on"}, "bedroom": {"fan": "on"}},
        "notes": "kit=kitchen, bed=bedroom, + joins rooms",
    },
    {
        "input": "tv off",
        "category": "abbreviation",
        "expected": None,
        "notes": "No room specified — should it default to living_room/hall or refuse?",
    },

    # --- Category 5: Ambiguous / Vague Commands ---
    {
        "input": "make it cozy",
        "category": "ambiguous",
        "expected": None,
        "notes": "Vague mood command — no clear device mapping",
    },
    {
        "input": "set up for guests",
        "category": "ambiguous",
        "expected": None,
        "notes": "What does 'guests' setup mean? Model must decide.",
    },
    {
        "input": "can you do the usual",
        "category": "ambiguous",
        "expected": None,
        "notes": "Requires context/history that model doesn't have",
    },
    {
        "input": "everything on in the house",
        "category": "ambiguous",
        "expected": None,
        "notes": "Turn on everything? In all rooms? Which devices?",
    },

    # --- Category 6: Complex / Multi-step Informal ---
    {
        "input": "gonna study for a bit then probably watch a movie later but for now set up study mode",
        "category": "complex_informal",
        "expected": {
            "study_room": {"computer": "on", "lights": "bright", "fan": "on"},
            "living_room": {"tv": "off", "music_system": "off"},
            "bedroom": {"lights": "off"},
        },
        "notes": "Should ignore future intent, focus on 'set up study mode'",
    },
    {
        "input": "bedroom lights dim and also can you close those blinds and maybe turn the ac to like 23 or something",
        "category": "complex_informal",
        "expected": {"bedroom": {"lights": "dim", "blinds": "close", "ac": "23°C"}},
        "notes": "Hedging language: 'maybe', 'like', 'or something'",
    },
    {
        "input": "first kitchen lights on then go to bathroom and start the geyser and the exhaust too",
        "category": "complex_informal",
        "expected": {
            "kitchen": {"lights": "on"},
            "bathroom": {"geyser": "on", "exhaust": "on"},
        },
        "notes": "Sequential phrasing with 'first...then...and...too'",
    },

    # --- Category 7: Negation / Correction ---
    {
        "input": "dont turn off the fan in the bedroom",
        "category": "negation",
        "expected": {"bedroom": {"fan": "on"}},
        "notes": "Double negation: don't turn off = keep on",
    },
    {
        "input": "turn on lights everywhere except the bedroom",
        "category": "negation",
        "expected": None,
        "notes": "Exclusion — requires reasoning about all rooms minus one",
    },
    {
        "input": "actually no keep the ac off in dining room",
        "category": "negation",
        "expected": {"dining_room": {"ac": "off"}},
        "notes": "Correction/negation pattern",
    },

    # --- Category 8: Code-Switching (Hindi-English) ---
    {
        "input": "bedroom mein light on karo",
        "category": "code_switch",
        "expected": {"bedroom": {"lights": "on"}},
        "notes": "Hindi-English: 'mein'=in, 'karo'=do",
    },
    {
        "input": "kitchen ka fan band karo",
        "category": "code_switch",
        "expected": {"kitchen": {"fan": "off"}},
        "notes": "Hindi: 'ka'=of/in, 'band karo'=turn off",
    },
    {
        "input": "sab kuch off kardo ghar mein",
        "category": "code_switch",
        "expected": None,
        "notes": "Hindi: 'sab kuch off kardo'=turn everything off, 'ghar mein'=in house",
    },

    # --- Category 9: Polite but Verbose ---
    {
        "input": "hey so I was wondering if you could maybe turn on the lights in the living room if thats not too much trouble thanks so much",
        "category": "verbose",
        "expected": {"living_room": {"lights": "on"}},
        "notes": "Overly polite, lots of filler",
    },
    {
        "input": "I would really appreciate it if the air conditioning in the bedroom could be set to approximately twenty two degrees celsius",
        "category": "verbose",
        "expected": {"bedroom": {"ac": "22°C"}},
        "notes": "Written-out number, verbose phrasing",
    },

    # --- Category 10: Edge Cases ---
    {
        "input": "",
        "category": "edge_case",
        "expected": {},
        "notes": "Empty command — should return empty JSON or refuse",
    },
    {
        "input": "play some jazz",
        "category": "edge_case",
        "expected": None,
        "notes": "'jazz' is not a valid music_system mode (soft/medium/loud/party/relax)",
    },
    {
        "input": "set ac to 30 in bedroom",
        "category": "edge_case",
        "expected": None,
        "notes": "30°C is outside training range (18-25). What does the model do?",
    },
    {
        "input": "turn on the microwave",
        "category": "edge_case",
        "expected": None,
        "notes": "Microwave is not a known device. Should model refuse or hallucinate?",
    },
    {
        "input": "turn on the lights in the garage",
        "category": "edge_case",
        "expected": None,
        "notes": "Garage is not a known room. Should model refuse or hallucinate?",
    },
]


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
        messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_input}]
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

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def normalize_json(json_str: str):
    """Try to parse JSON from model output."""
    try:
        obj = json.loads(json_str)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        start = json_str.index("{")
        end = json_str.rindex("}") + 1
        obj = json.loads(json_str[start:end])
        if isinstance(obj, dict):
            return obj
    except (ValueError, json.JSONDecodeError, TypeError):
        pass
    return None


def evaluate_casual(model, tokenizer, model_name: str, max_new_tokens: int = 256):
    """Run all casual commands through the model and analyze results."""
    results = []

    for i, cmd in enumerate(CASUAL_COMMANDS):
        user_input = cmd["input"]
        print(f"  [{i+1}/{len(CASUAL_COMMANDS)}] {cmd['category']:20s} | {user_input[:60]}...")

        raw_output = generate_output(model, tokenizer, user_input, model_name, max_new_tokens)
        parsed = normalize_json(raw_output)

        result = {
            "index": i + 1,
            "input": user_input,
            "category": cmd["category"],
            "raw_output": raw_output,
            "parsed_json": parsed,
            "is_valid_json": parsed is not None,
            "expected": cmd.get("expected"),
            "notes": cmd.get("notes", ""),
        }

        # Check match if expected is provided
        if cmd.get("expected") is not None and parsed is not None:
            expected_sorted = json.dumps(cmd["expected"], sort_keys=True)
            predicted_sorted = json.dumps(parsed, sort_keys=True)
            result["exact_match"] = expected_sorted == predicted_sorted
        else:
            result["exact_match"] = None

        results.append(result)

    return results


def print_report(model_name: str, results: list[dict]):
    """Print a formatted report."""
    print(f"\n{'=' * 80}")
    print(f"  CASUAL COMMAND EVALUATION — {model_name.upper()}")
    print(f"{'=' * 80}")

    total = len(results)
    valid_json = sum(1 for r in results if r["is_valid_json"])
    has_expected = [r for r in results if r["expected"] is not None]
    exact_matches = sum(1 for r in has_expected if r["exact_match"])

    print(f"\n  Total commands:     {total}")
    print(f"  Valid JSON output:  {valid_json}/{total} ({100*valid_json/total:.1f}%)")
    print(f"  Exact match:        {exact_matches}/{len(has_expected)} ({100*exact_matches/len(has_expected):.1f}%) [only commands with defined expected output]")

    # Per-category breakdown
    categories = defaultdict(lambda: {"total": 0, "valid": 0, "match": 0, "has_expected": 0})
    for r in results:
        cat = r["category"]
        categories[cat]["total"] += 1
        if r["is_valid_json"]:
            categories[cat]["valid"] += 1
        if r["expected"] is not None:
            categories[cat]["has_expected"] += 1
            if r["exact_match"]:
                categories[cat]["match"] += 1

    print(f"\n  {'Category':<22s} {'Valid JSON':>12s} {'Exact Match':>14s}")
    print(f"  {'─' * 50}")
    for cat in sorted(categories.keys()):
        c = categories[cat]
        valid_str = f"{c['valid']}/{c['total']}"
        if c["has_expected"] > 0:
            match_str = f"{c['match']}/{c['has_expected']}"
        else:
            match_str = "N/A"
        print(f"  {cat:<22s} {valid_str:>12s} {match_str:>14s}")

    # Show failures
    print(f"\n  {'─' * 70}")
    print(f"  NOTABLE OUTPUTS:")
    print(f"  {'─' * 70}")
    for r in results:
        status = "✅" if r["is_valid_json"] else "❌"
        match_status = ""
        if r["exact_match"] is True:
            match_status = " [MATCH]"
        elif r["exact_match"] is False:
            match_status = " [MISMATCH]"

        print(f"\n  {status} [{r['category']}] {r['input'][:70]}{match_status}")
        if r["is_valid_json"]:
            print(f"     → {json.dumps(r['parsed_json'], sort_keys=True)[:120]}")
        else:
            print(f"     → RAW: {r['raw_output'][:120]}")
        if r["notes"]:
            print(f"     Note: {r['notes']}")

    return categories


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models on informal/casual commands")
    parser.add_argument("--model", type=str, required=True, choices=["llama", "gemma", "both"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default=os.path.join(SCRIPT_DIR, "results"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    models_to_eval = ["llama", "gemma"] if args.model == "both" else [args.model]
    all_results = {}

    for model_name in models_to_eval:
        print(f"\n{'=' * 60}")
        print(f"Loading {model_name.upper()} for casual command evaluation...")
        print(f"{'=' * 60}")

        model, tokenizer = load_model_and_tokenizer(model_name)
        results = evaluate_casual(model, tokenizer, model_name, args.max_new_tokens)
        categories = print_report(model_name, results)
        all_results[model_name] = results

        # Save detailed results
        out_path = os.path.join(args.output_dir, f"{model_name}_casual_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Detailed results saved to {out_path}")

        del model, tokenizer
        torch.cuda.empty_cache()

    # Save comparison summary
    summary_path = os.path.join(args.output_dir, "casual_eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Informal/Casual Command Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total test commands: {len(CASUAL_COMMANDS)}\n")
        f.write(f"Categories: {len(set(c['category'] for c in CASUAL_COMMANDS))}\n\n")

        for model_name, results in all_results.items():
            total = len(results)
            valid = sum(1 for r in results if r["is_valid_json"])
            has_exp = [r for r in results if r["expected"] is not None]
            exact = sum(1 for r in has_exp if r["exact_match"])

            f.write(f"\n{'─' * 50}\n")
            f.write(f"Model: {model_name.upper()}\n")
            f.write(f"{'─' * 50}\n")
            f.write(f"  Valid JSON Rate:    {valid}/{total} ({100*valid/total:.1f}%)\n")
            f.write(f"  Exact Match Rate:   {exact}/{len(has_exp)} ({100*exact/len(has_exp):.1f}%)\n\n")

            # Per-category
            cats = defaultdict(lambda: {"total": 0, "valid": 0, "match": 0, "has_exp": 0})
            for r in results:
                cat = r["category"]
                cats[cat]["total"] += 1
                if r["is_valid_json"]:
                    cats[cat]["valid"] += 1
                if r["expected"] is not None:
                    cats[cat]["has_exp"] += 1
                    if r["exact_match"]:
                        cats[cat]["match"] += 1

            f.write(f"  {'Category':<22s} {'Valid JSON':>12s} {'Match':>10s}\n")
            for cat in sorted(cats.keys()):
                c = cats[cat]
                f.write(f"  {cat:<22s} {c['valid']}/{c['total']:>8s} ")
                if c["has_exp"] > 0:
                    f.write(f"{c['match']}/{c['has_exp']:>6s}\n")
                else:
                    f.write(f"{'N/A':>7s}\n")

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
