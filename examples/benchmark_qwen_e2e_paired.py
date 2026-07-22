"""Paired Qwen A/B benchmark that alternates baseline and fused MLP runs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_qwen_e2e import make_inputs, timed_generate
from qwen_kernel_lab import native_extension_loaded
from qwen_kernel_lab.integration import (
    replace_qwen_decoder_rmsnorm,
    replace_qwen_mlp,
)


def medians(samples):
    return {
        key: round(statistics.median(sample[key] for sample in samples), 4)
        for key in samples[0]
    }


def measure_pair(baseline, native, input_ids, new_tokens, warmup, runs):
    for index in range(warmup):
        order = (baseline, native) if index % 2 == 0 else (native, baseline)
        for model in order:
            timed_generate(model, input_ids, new_tokens)

    samples = {"baseline": [], "native_fused": []}
    first_outputs = {}
    for index in range(runs):
        order = (
            (("baseline", baseline), ("native_fused", native))
            if index % 2 == 0
            else (("native_fused", native), ("baseline", baseline))
        )
        for label, model in order:
            metrics, tokens, logits = timed_generate(model, input_ids, new_tokens)
            samples[label].append(metrics)
            if label not in first_outputs:
                first_outputs[label] = (tokens, logits)

    base = medians(samples["baseline"])
    candidate = medians(samples["native_fused"])
    base_tokens, base_logits = first_outputs["baseline"]
    native_tokens, native_logits = first_outputs["native_fused"]
    diff = (base_logits - native_logits).abs()
    return {
        "baseline": base,
        "native_fused": candidate,
        "speedup": {
            "ttft": round(base["ttft_ms"] / candidate["ttft_ms"], 4),
            "decode": round(
                candidate["decode_tokens_per_s"] / base["decode_tokens_per_s"], 4
            ),
            "e2e": round(
                candidate["output_tokens_per_s"] / base["output_tokens_per_s"], 4
            ),
        },
        "correctness": {
            "max_abs_first_logits": diff.max().item(),
            "mean_abs_first_logits": diff.mean().item(),
            "generated_tokens_equal": torch.equal(base_tokens, native_tokens),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--fusion",
        choices=("mlp", "rms", "both"),
        default="mlp",
        help="Select the candidate fusion path; 'mlp' is the stable default, "
        "while 'rms' and 'both' are Prefill-oriented experiments.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not native_extension_loaded():
        raise SystemExit("CUDA and the native extension are required")

    from transformers import AutoConfig, AutoModelForCausalLM

    dtype = torch.float16
    config = AutoConfig.from_pretrained(args.model)
    inputs = make_inputs(config.vocab_size, args.prompt_lengths, args.batch_size)
    load_kwargs = {
        "dtype": dtype,
        "attn_implementation": args.attn_implementation,
    }

    baseline = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).cuda().eval()
    native = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).cuda().eval()
    mlp_replacements = (
        replace_qwen_mlp(native) if args.fusion in ("mlp", "both") else []
    )
    decoder_replacements = (
        replace_qwen_decoder_rmsnorm(native) if args.fusion in ("rms", "both") else []
    )
    if not mlp_replacements and not decoder_replacements:
        raise SystemExit("No compatible Qwen modules were found")

    torch.cuda.reset_peak_memory_stats()
    cases = []
    for length in args.prompt_lengths:
        case = measure_pair(
            baseline,
            native,
            inputs[length].cuda(),
            args.new_tokens,
            args.warmup,
            args.runs,
        )
        case["prompt_length"] = length
        cases.append(case)

    result = {
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "model": args.model,
            "dtype": str(dtype),
            "attention": args.attn_implementation,
            "batch_size": args.batch_size,
            "new_tokens": args.new_tokens,
            "warmup": args.warmup,
            "paired_runs": args.runs,
            "fusion": args.fusion,
            "patched_mlp_modules": len(mlp_replacements),
            "patched_decoder_modules": len(decoder_replacements),
            "peak_memory_mb": round(
                torch.cuda.max_memory_allocated() / (1024 * 1024), 2
            ),
        },
        "cases": cases,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
