"""
RAG Inference for Smart Home Commands
======================================
Implements the personalisation pipeline shown in
`generate_rag_visualization.py`:

  user command  --> sentence embedding  --> nearest entry in the
  per-user knowledge base  --> if similarity > threshold, inject the
  matched alias into the model prompt as extra context  --> let the
  fine-tuned SLM produce the final JSON.

The KB is just a small JSON file on the edge device. Each entry maps
a user-defined alias (e.g. "yoyo", "chill mode", "mom's here") to
either an explicit JSON device-state target, or a natural-language
expansion that the model can consume directly.

Launch:
    # one-off command
    python rag_inference.py \\
        --model_path  checkpoints/gemma/final \\
        --kb_path     data/user_kb.json \\
        --command     "yoyo"

    # interactive mode
    python rag_inference.py --model_path checkpoints/gemma/final \\
                            --kb_path data/user_kb.json --interactive
"""

import argparse
import json
import os
import time
from typing import Optional

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KB = os.path.join(SCRIPT_DIR, "data", "user_kb.json")

SYSTEM_PROMPT = (
    "You are a smart home assistant for a 2BHK house. "
    "Given a natural language command, output the corresponding device "
    "states as a JSON object. The house has these rooms: bathroom, "
    "bedroom, balcony, dining_room, hall, kitchen, living_room, "
    "study_room. Available devices include: ac, blinds, computer, "
    "exhaust, fan, geyser, lights, music_system, tv. Respond ONLY with "
    "a valid JSON object, no explanation."
)

# ============================================================
# Default knowledge base (used if --kb_path doesn't exist yet)
# ============================================================
DEFAULT_KB_CONTENT = {
    "entries": [
        {
            "alias": "yoyo",
            "expansion": "Set up movie night in the living room: dim the lights to 20%, turn on the TV, set AC to 22C and switch off the kitchen lights.",
            "target": {
                "living_room": {"lights": "dim", "tv": "on", "ac": "22C"},
                "kitchen": {"lights": "off"},
            },
        },
        {
            "alias": "chill mode",
            "expansion": "Relaxing setup in the hall: dim the lights, switch on the music system at low volume.",
            "target": {
                "hall": {"lights": "dim", "music_system": "on"},
            },
        },
        {
            "alias": "mom's here",
            "expansion": "Switch off the TV in the living room and play soft music in the hall.",
            "target": {
                "living_room": {"tv": "off"},
                "hall": {"music_system": "on"},
            },
        },
        {
            "alias": "study time",
            "expansion": "Turn on the study room lights and computer, switch off the music in the hall.",
            "target": {
                "study_room": {"lights": "on", "computer": "on"},
                "hall": {"music_system": "off"},
            },
        },
        {
            "alias": "good night",
            "expansion": "Turn off all the lights and the TV; set the bedroom AC to 24C.",
            "target": {
                "bedroom": {"lights": "off", "ac": "24C"},
                "hall": {"lights": "off", "tv": "off"},
                "kitchen": {"lights": "off"},
                "living_room": {"lights": "off"},
            },
        },
    ]
}


def load_or_init_kb(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(DEFAULT_KB_CONTENT, f, indent=2)
    return DEFAULT_KB_CONTENT


# ============================================================
# Retriever
# ============================================================
class KBRetriever:
    """
    Sentence-transformer based retriever. We keep this dependency
    optional: if `sentence_transformers` is not installed (e.g. on a
    very small edge device) we fall back to character n-gram cosine
    similarity which is good enough for a few dozen aliases.
    """
    def __init__(self, kb, encoder_name="sentence-transformers/all-MiniLM-L6-v2",
                 threshold=0.55):
        self.kb = kb
        self.threshold = threshold
        self.aliases = [e["alias"] for e in kb["entries"]]
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(encoder_name)
            self.alias_emb = self.encoder.encode(
                self.aliases, normalize_embeddings=True, convert_to_tensor=True)
            self.mode = "st"
        except Exception as e:
            print(f"[RAG] sentence_transformers unavailable ({e}); "
                  "falling back to char n-gram similarity.")
            self.mode = "ngram"
            self._build_ngram_index()

    # ---- char n-gram fallback ----
    def _ngrams(self, s, n=3):
        s = "  " + s.lower().strip() + "  "
        return {s[i:i+n] for i in range(len(s) - n + 1)}

    def _build_ngram_index(self):
        self.alias_ngrams = [self._ngrams(a) for a in self.aliases]

    def _ngram_sim(self, q):
        q_ng = self._ngrams(q)
        sims = []
        for a_ng in self.alias_ngrams:
            inter = len(q_ng & a_ng)
            union = len(q_ng | a_ng) or 1
            sims.append(inter / union)
        return sims

    # ---- public ----
    def retrieve(self, command: str):
        """Return (best_entry, score) or (None, score) if below threshold."""
        if self.mode == "st":
            import torch.nn.functional as F
            q = self.encoder.encode([command], normalize_embeddings=True,
                                    convert_to_tensor=True)
            scores = (q @ self.alias_emb.T).squeeze(0).cpu().tolist()
        else:
            scores = self._ngram_sim(command)

        best_i = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_i]
        if best_score < self.threshold:
            return None, best_score
        return self.kb["entries"][best_i], best_score


# ============================================================
# Prompt construction
# ============================================================
def build_prompt(command: str, retrieved: Optional[dict]) -> list:
    """Build a chat-template messages list, with retrieved context if any."""
    user_msg = command
    if retrieved is not None:
        # Inject the alias hint into the user message itself so any
        # chat template (LLaMA, Gemma, Qwen, Phi) sees it identically.
        user_msg = (
            f"User command: {command}\n\n"
            f"[Personal shortcut detected: '{retrieved['alias']}']\n"
            f"Hint: {retrieved['expansion']}\n"
            f"Use this hint to produce the correct JSON for the user."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


# ============================================================
# Model loading
# ============================================================
def load_model(model_path: str, base_model: Optional[str] = None,
               int4: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    quant_kwargs = {}
    if int4:
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # If model_path is a LoRA adapter folder, load base + adapter.
    has_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    if has_adapter:
        if base_model is None:
            with open(os.path.join(model_path, "adapter_config.json")) as f:
                cfg = json.load(f)
            base_model = cfg.get("base_model_name_or_path")
        tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
            **quant_kwargs,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
            **quant_kwargs,
        )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


# ============================================================
# RAG-augmented generation
# ============================================================
@torch.no_grad()
def generate(model, tokenizer, messages, max_new_tokens=128):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    dt_ms = (time.perf_counter() - t0) * 1000
    text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
    return text, dt_ms


def parse_json(text: str):
    """Best-effort JSON extraction from the model output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        return json.loads(text)
    except Exception:
        # Try to grab the first {...} block.
        s = text.find("{")
        e = text.rfind("}")
        if 0 <= s < e:
            try:
                return json.loads(text[s:e+1])
            except Exception:
                return None
        return None


def run_once(command, retriever, model, tokenizer, max_new_tokens=128):
    entry, score = retriever.retrieve(command)
    messages = build_prompt(command, entry)
    text, dt_ms = generate(model, tokenizer, messages, max_new_tokens)
    parsed = parse_json(text)
    return {
        "command": command,
        "retrieved_alias": entry["alias"] if entry else None,
        "retrieval_score": round(score, 3),
        "raw_output": text,
        "parsed_json": parsed,
        "latency_ms": round(dt_ms, 1),
    }


# ============================================================
# Entry point
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Path to a fine-tuned model OR a LoRA adapter folder.")
    p.add_argument("--base_model", default=None,
                   help="Base model id (only needed if --model_path is a LoRA adapter "
                        "and adapter_config.json doesn't list it).")
    p.add_argument("--kb_path", default=DEFAULT_KB)
    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--threshold", type=float, default=0.55,
                   help="Minimum retrieval similarity to inject the KB hint.")
    p.add_argument("--int4", action="store_true",
                   help="Load the model in 4-bit NF4 (matches a pruned+quant flow).")
    p.add_argument("--command", default=None,
                   help="Run a single command and exit.")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    kb = load_or_init_kb(args.kb_path)
    print(f"[RAG] Loaded KB with {len(kb['entries'])} entries from {args.kb_path}")
    retriever = KBRetriever(kb, encoder_name=args.encoder, threshold=args.threshold)
    print(f"[RAG] Retriever mode = {retriever.mode}")

    print(f"[RAG] Loading model: {args.model_path}")
    model, tokenizer = load_model(args.model_path, args.base_model, int4=args.int4)
    print("[RAG] Ready.")

    if args.command is not None:
        result = run_once(args.command, retriever, model, tokenizer,
                          args.max_new_tokens)
        print(json.dumps(result, indent=2))
        return

    if args.interactive:
        print("Type a command (Ctrl+C to quit).")
        while True:
            try:
                cmd = input("\n>>> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if not cmd:
                continue
            result = run_once(cmd, retriever, model, tokenizer,
                              args.max_new_tokens)
            print(json.dumps(result, indent=2))
        return

    print("Nothing to do; pass --command or --interactive.")


if __name__ == "__main__":
    main()
