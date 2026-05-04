"""
Structured Pruning for Smart Home Models
=========================================
Matches the methodology shown in `generate_pruning_visualization.py` /
`generate_pruning_network_viz.py`: instead of zeroing individual weights
(which gives unstructured sparsity that GPUs cannot accelerate well),
we remove **entire rows / columns** of the projection matrices, i.e.
whole attention heads and MLP neurons. The result is a smaller dense
model that runs faster on any GPU.

Per layer the procedure is:
  (1) Compute importance scores on a small calibration set.
        score_j = ||W[:, j]||_2  *  mean(|activation_j|)
  (2) Drop the lowest-scoring `--sparsity` fraction of rows/columns from
      q/k/v/o/gate/up/down projections (head-aware for q/k/v/o so
      whole heads are removed together).
  (3) Optionally apply INT4 NF4 weight quantisation on top.

Launch:
    python prune_structured.py \
        --model_path checkpoints/llama/final \
        --output_dir checkpoints/llama/pruned_structured \
        --sparsity   0.5 \
        --calib_samples 256 \
        --quantize int4 \
        --benchmark
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Projections we touch. q/k/v/o are head-shaped, gate/up/down are MLP-shaped.
ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJ = ("gate_proj", "up_proj", "down_proj")
ALL_PROJ = ATTN_PROJ + MLP_PROJ


# ============================================================
# Helpers
# ============================================================
def find_decoder_layers(model):
    """Return the list of transformer blocks (works for LLaMA / Gemma / Qwen / Phi)."""
    for path in ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj
    raise RuntimeError("Could not locate decoder layers on this model.")


def linears_in_layer(layer):
    """Return dict {short_name: nn.Linear} for the projections we prune."""
    out = {}
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear):
            short = name.split(".")[-1]
            if short in ALL_PROJ:
                out[short] = mod
    return out


# ============================================================
# Activation collection (calibration)
# ============================================================
class ActivationStats:
    """Hook a Linear layer and accumulate mean(|input|) per input feature."""
    def __init__(self, module: nn.Linear):
        self.module = module
        self.sum_abs = torch.zeros(module.in_features, device=module.weight.device)
        self.count = 0
        self.handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        x = inputs[0]
        if x.dim() == 3:           # (B, T, F)
            x = x.reshape(-1, x.shape[-1])
        self.sum_abs += x.detach().abs().mean(dim=0).to(self.sum_abs.device)
        self.count += 1

    def mean_abs(self):
        return self.sum_abs / max(self.count, 1)

    def close(self):
        self.handle.remove()


@torch.no_grad()
def run_calibration(model, tokenizer, texts, max_len=512):
    """Forward `texts` through the model and return {linear: mean_abs_input}."""
    layers = find_decoder_layers(model)
    stats = {}
    for layer in layers:
        for short, lin in linears_in_layer(layer).items():
            stats[lin] = ActivationStats(lin)

    model.eval()
    for t in texts:
        toks = tokenizer(t, return_tensors="pt", truncation=True,
                         max_length=max_len).to(model.device)
        model(**toks)

    result = {lin: s.mean_abs().clone() for lin, s in stats.items()}
    for s in stats.values():
        s.close()
    return result


# ============================================================
# Structured pruning
# ============================================================
def head_dim_of(config):
    """Best-effort guess at head_dim for grouping q/k/v/o columns into heads."""
    if hasattr(config, "head_dim") and config.head_dim:
        return config.head_dim
    return config.hidden_size // config.num_attention_heads


def importance_for_columns(W: torch.Tensor, act_mean: torch.Tensor) -> torch.Tensor:
    """score_j = ||W[:, j]||_2 * |act|_j  (one score per input column)."""
    col_norm = W.float().pow(2).sum(dim=0).sqrt()    # (in_features,)
    return col_norm * act_mean.float().to(col_norm.device)


def prune_linear_columns(linear: nn.Linear, keep_idx: torch.Tensor):
    """Replace the linear with a smaller one keeping only `keep_idx` input columns."""
    keep_idx = keep_idx.to(linear.weight.device).long().sort().values
    new = nn.Linear(keep_idx.numel(), linear.out_features,
                    bias=(linear.bias is not None),
                    dtype=linear.weight.dtype, device=linear.weight.device)
    with torch.no_grad():
        new.weight.copy_(linear.weight.index_select(1, keep_idx))
        if linear.bias is not None:
            new.bias.copy_(linear.bias)
    return new, keep_idx


def prune_linear_rows(linear: nn.Linear, keep_idx: torch.Tensor):
    """Replace the linear with a smaller one keeping only `keep_idx` output rows."""
    keep_idx = keep_idx.to(linear.weight.device).long().sort().values
    new = nn.Linear(linear.in_features, keep_idx.numel(),
                    bias=(linear.bias is not None),
                    dtype=linear.weight.dtype, device=linear.weight.device)
    with torch.no_grad():
        new.weight.copy_(linear.weight.index_select(0, keep_idx))
        if linear.bias is not None:
            new.bias.copy_(linear.bias.index_select(0, keep_idx))
    return new, keep_idx


def replace_module(parent: nn.Module, child_name: str, new_module: nn.Module):
    setattr(parent, child_name, new_module)


def structured_prune_layer(layer, act_means, sparsity: float, head_dim: int):
    """
    Prune one transformer block.
      - MLP: drop the lowest-scoring `sparsity` fraction of intermediate
             neurons. This means dropping ROWS of gate_proj & up_proj
             AND COLUMNS of down_proj at the same indices.
      - Attention: drop the lowest-scoring fraction of *whole heads*
             (groups of `head_dim` consecutive output rows of q/k/v
              and matching input columns of o_proj).
    """
    linears = linears_in_layer(layer)
    # ---------- MLP intermediate dimension ----------
    if {"gate_proj", "up_proj", "down_proj"}.issubset(linears):
        gate, up, down = linears["gate_proj"], linears["up_proj"], linears["down_proj"]
        inter = gate.out_features
        # importance per intermediate neuron j = ||gate[j,:]||*||up[j,:]|| * mean|act_in_to_down|_j
        gate_row = gate.weight.float().pow(2).sum(dim=1).sqrt()
        up_row   = up.weight.float().pow(2).sum(dim=1).sqrt()
        act_d    = act_means[down]                        # (inter,)
        score    = gate_row * up_row * act_d.to(gate_row.device)
        n_keep   = max(1, int(round(inter * (1 - sparsity))))
        keep     = torch.topk(score, n_keep).indices

        new_gate, _ = prune_linear_rows(gate, keep)
        new_up,   _ = prune_linear_rows(up,   keep)
        new_down, _ = prune_linear_columns(down, keep)

        # find their parent module to replace
        for parent_name, parent in layer.named_modules():
            for cname, cmod in parent.named_children():
                if cmod is gate: replace_module(parent, cname, new_gate)
                elif cmod is up:   replace_module(parent, cname, new_up)
                elif cmod is down: replace_module(parent, cname, new_down)

    # ---------- Attention: prune whole heads ----------
    if {"q_proj", "k_proj", "v_proj", "o_proj"}.issubset(linears):
        q, k, v, o = (linears["q_proj"], linears["k_proj"],
                      linears["v_proj"], linears["o_proj"])
        # head importance = sum over rows in that head (q+k+v) * mean|act| into o
        n_q_heads = q.out_features // head_dim
        n_kv_heads = v.out_features // head_dim   # may differ (GQA)

        def head_score(lin: nn.Linear, n_heads: int):
            row_norm = lin.weight.float().pow(2).sum(dim=1).sqrt()  # (out,)
            return row_norm.view(n_heads, head_dim).sum(dim=1)      # (n_heads,)

        s_q = head_score(q, n_q_heads)
        s_k = head_score(k, n_kv_heads)
        s_v = head_score(v, n_kv_heads)
        # o_proj sees the concatenated heads as input columns
        o_act = act_means[o].view(n_q_heads, head_dim).sum(dim=1).to(s_q.device)

        score_q = s_q * o_act
        # When using grouped-query attention we keep KV heads in proportion
        kv_ratio = n_kv_heads / n_q_heads

        n_keep_q = max(1, int(round(n_q_heads * (1 - sparsity))))
        keep_q = torch.topk(score_q, n_keep_q).indices.sort().values

        # Map kept q-heads back to their kv group
        if n_kv_heads == n_q_heads:
            keep_kv = keep_q
        else:
            group = max(1, n_q_heads // n_kv_heads)
            keep_kv = torch.unique(keep_q // group).sort().values
            # ensure we keep at least one
            if keep_kv.numel() == 0:
                keep_kv = torch.tensor([0], device=keep_q.device)

        # Expand head indices to row indices
        def expand(heads, hd):
            return (heads.unsqueeze(1) * hd
                    + torch.arange(hd, device=heads.device).unsqueeze(0)).reshape(-1)

        rows_q  = expand(keep_q,  head_dim)
        rows_kv = expand(keep_kv, head_dim)
        cols_o  = rows_q                          # o_proj input cols == concat of q heads

        new_q, _ = prune_linear_rows(q, rows_q)
        new_k, _ = prune_linear_rows(k, rows_kv)
        new_v, _ = prune_linear_rows(v, rows_kv)
        new_o, _ = prune_linear_columns(o, cols_o)

        for parent_name, parent in layer.named_modules():
            for cname, cmod in parent.named_children():
                if cmod is q: replace_module(parent, cname, new_q)
                elif cmod is k: replace_module(parent, cname, new_k)
                elif cmod is v: replace_module(parent, cname, new_v)
                elif cmod is o: replace_module(parent, cname, new_o)

        # Update attention config inside the module so forward still works
        attn = None
        for _, m in layer.named_modules():
            if hasattr(m, "num_heads") and hasattr(m, "head_dim"):
                attn = m
                break
        if attn is not None:
            attn.num_heads = keep_q.numel()
            if hasattr(attn, "num_key_value_heads"):
                attn.num_key_value_heads = keep_kv.numel()
            if hasattr(attn, "hidden_size"):
                attn.hidden_size = keep_q.numel() * head_dim


# ============================================================
# Latency benchmark
# ============================================================
@torch.no_grad()
def benchmark_latency(model, tokenizer, n=30, max_new_tokens=64):
    model.eval()
    prompt = "Turn on the living room lights and set the bedroom AC to 22 degrees."
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    for _ in range(3):
        model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================
# Calibration data loading
# ============================================================
def load_calibration_texts(n_samples: int):
    """Take a few hundred training prompts as the calibration set."""
    train_file = os.path.join(DATA_DIR, "train.jsonl")
    if not os.path.exists(train_file):
        # Fallback: a handful of canned prompts
        return [
            "Turn on the lights in the bedroom and set AC to 22C",
            "Switch off the kitchen lights",
            "It's hot in here",
            "Movie night in the hall",
            "Shut everything down, I am leaving",
        ] * n_samples
    texts = []
    with open(train_file, "r") as f:
        for line in f:
            row = json.loads(line)
            # Try common keys
            if "messages" in row:
                txt = " ".join(m.get("content", "") for m in row["messages"])
            elif "text" in row:
                txt = row["text"]
            elif "input" in row:
                txt = row["input"]
            else:
                txt = json.dumps(row)
            texts.append(txt)
            if len(texts) >= n_samples:
                break
    return texts


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="HF id or local path to the (fine-tuned) model.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sparsity", type=float, default=0.5,
                   help="Fraction of heads / MLP neurons to drop per layer.")
    p.add_argument("--calib_samples", type=int, default=256)
    p.add_argument("--quantize", choices=["none", "int4"], default="none",
                   help="Optional INT4 NF4 quantisation of the pruned model.")
    p.add_argument("--benchmark", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Structured pruning  sparsity={args.sparsity}  "
          f"calib_samples={args.calib_samples}")
    print(f"Loading model: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    config = AutoConfig.from_pretrained(args.model_path)
    head_dim = head_dim_of(config)

    n_before = count_params(model)
    if args.benchmark:
        ms_before = benchmark_latency(model, tokenizer)
        print(f"  Latency before : {ms_before:7.2f} ms / sample")
        print(f"  Params  before : {n_before/1e6:7.1f} M")

    # ---- (1) calibration ----
    print("Running calibration...")
    texts = load_calibration_texts(args.calib_samples)
    act_means = run_calibration(model, tokenizer, texts)

    # ---- (2) prune layer by layer ----
    print("Pruning layers...")
    layers = find_decoder_layers(model)
    for i, layer in enumerate(layers):
        structured_prune_layer(layer, act_means, args.sparsity, head_dim)
        if i % 4 == 0:
            print(f"  layer {i+1}/{len(layers)} done")

    n_after = count_params(model)
    print(f"  Params  after  : {n_after/1e6:7.1f} M  "
          f"({100.0 * n_after / n_before:.1f}% of original)")
    if args.benchmark:
        ms_after = benchmark_latency(model, tokenizer)
        print(f"  Latency after  : {ms_after:7.2f} ms / sample  "
              f"({(1 - ms_after / ms_before) * 100:.1f}% faster)")

    # ---- (3) optional quantisation on save ----
    print(f"Saving pruned model to {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    if args.quantize == "int4":
        # We just record the recommended quant config for re-loading.
        bnb = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
        }
        with open(os.path.join(args.output_dir, "quant_config.json"), "w") as f:
            json.dump(bnb, f, indent=2)
        print("  Saved INT4 NF4 quant_config.json (use BitsAndBytesConfig at load).")

    print("Done.")


if __name__ == "__main__":
    main()
