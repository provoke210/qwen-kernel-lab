"""End-to-end Qwen prefill/decode benchmark with fused MLP A/B comparison."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_kernel_lab import native_extension_loaded
from qwen_kernel_lab.integration import replace_qwen_mlp


def make_inputs(vocab_size, lengths, batch_size):
    generator = torch.Generator(device="cpu").manual_seed(42)
    return {
        length: torch.randint(
            0, vocab_size, (batch_size, length), generator=generator, dtype=torch.long
        )
        for length in lengths
    }


@torch.inference_mode()
def timed_generate(model, input_ids, new_tokens):
    start = torch.cuda.Event(enable_timing=True)
    first = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    outputs = model(input_ids=input_ids, use_cache=True)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    first_logits_gpu = outputs.logits[:, -1, :]
    generated = [next_token]
    cache = outputs.past_key_values
    first.record()

    for _ in range(max(new_tokens - 1, 0)):
        outputs = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        cache = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)

    end.record()
    end.synchronize()
    first_logits = first_logits_gpu.float().cpu()

    ttft_ms = start.elapsed_time(first)
    decode_ms = first.elapsed_time(end)
    total_ms = start.elapsed_time(end)
    batch_size, prompt_length = input_ids.shape
    decode_count = batch_size * max(new_tokens - 1, 0)
    output_count = batch_size * new_tokens
    metrics = {
        "ttft_ms": ttft_ms,
        "prefill_tokens_per_s": batch_size * prompt_length * 1000.0 / ttft_ms,
        "decode_ms": decode_ms,
        "decode_tokens_per_s": decode_count * 1000.0 / decode_ms if decode_count else 0.0,
        "e2e_ms": total_ms,
        "output_tokens_per_s": output_count * 1000.0 / total_ms,
    }
    return metrics, torch.cat(generated, dim=1).cpu(), first_logits


def median_metrics(samples):
    return {
        key: round(statistics.median(sample[key] for sample in samples), 4)
        for key in samples[0]
    }


def benchmark_variant(model, inputs, device, new_tokens, warmup, runs):
    metrics_by_length = {}
    tokens_by_length = {}
    logits_by_length = {}

    for length, cpu_input_ids in inputs.items():
        input_ids = cpu_input_ids.to(device)
        for _ in range(warmup):
            timed_generate(model, input_ids, new_tokens)

        samples = []
        for run in range(runs):
            metrics, tokens, logits = timed_generate(model, input_ids, new_tokens)
            samples.append(metrics)
            if run == 0:
                tokens_by_length[length] = tokens
                logits_by_length[length] = logits
        metrics_by_length[length] = median_metrics(samples)

    return metrics_by_length, tokens_by_length, logits_by_length


def unload(model):
    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    if not native_extension_loaded():
        raise SystemExit("CUDA extension is not loaded; rebuild with QKL_BUILD_CUDA=1")

    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("Install dependencies with: pip install -e '.[qwen]'") from exc

    device = "cuda"
    dtype = torch.float16
    config = AutoConfig.from_pretrained(args.model)
    inputs = make_inputs(config.vocab_size, args.prompt_lengths, args.batch_size)
    load_kwargs = {
        "dtype": dtype,
        "attn_implementation": args.attn_implementation,
    }

    baseline_model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    baseline_model = baseline_model.to(device).eval()
    baseline, baseline_tokens, baseline_logits = benchmark_variant(
        baseline_model, inputs, device, args.new_tokens, args.warmup, args.runs
    )
    unload(baseline_model)

    native_model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    native_model = native_model.to(device).eval()
    replacements = replace_qwen_mlp(native_model)
    if not replacements:
        raise SystemExit("No compatible Qwen MLP modules were found")
    torch.cuda.reset_peak_memory_stats()
    native, native_tokens, native_logits = benchmark_variant(
        native_model, inputs, device, args.new_tokens, args.warmup, args.runs
    )
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    unload(native_model)

    cases = []
    for length in args.prompt_lengths:
        base = baseline[length]
        candidate = native[length]
        logits_diff = (baseline_logits[length] - native_logits[length]).abs()
        cases.append(
            {
                "prompt_length": length,
                "baseline": base,
                "native_mlp": candidate,
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
                    "max_abs_first_logits": logits_diff.max().item(),
                    "mean_abs_first_logits": logits_diff.mean().item(),
                    "generated_tokens_equal": torch.equal(
                        baseline_tokens[length], native_tokens[length]
                    ),
                },
            }
        )

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
            "runs": args.runs,
            "patched_mlp_modules": len(replacements),
            "peak_memory_mb": round(peak_memory_mb, 2),
        },
        "cases": cases,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
