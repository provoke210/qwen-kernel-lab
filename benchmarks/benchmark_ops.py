from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import torch

# Allow direct execution from a source checkout before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_kernel_lab import add_rms_norm, native_extension_loaded, silu_mul


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def latency_us(fn: Callable[[], torch.Tensor], device: torch.device, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)

    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        fn()
        synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1_000.0)
    return statistics.median(samples)


def shapes() -> Iterable[Tuple[str, Tuple[int, ...]]]:
    yield "decode_b1", (1, 1, 896)
    yield "decode_b8", (8, 1, 896)
    yield "prefill_128", (1, 128, 896)
    yield "prefill_512", (1, 512, 896)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float16" if torch.cuda.is_available() else "float32")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        raise SystemExit("Use --dtype float32 for a representative CPU benchmark")

    results: Dict[str, object] = {
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "dtype": args.dtype,
            "native_extension": native_extension_loaded(),
        },
        "cases": [],
    }

    torch.manual_seed(42)
    with torch.inference_mode():
        for label, shape in shapes():
            x = torch.randn(shape, device=device, dtype=dtype)
            residual = torch.randn_like(x)
            weight = torch.randn(shape[-1], device=device, dtype=dtype)
            gate = torch.randn(*shape[:-1], 4864, device=device, dtype=dtype)
            up = torch.randn_like(gate)

            rms_ref = latency_us(
                lambda: add_rms_norm(x, residual, weight, use_native=False),
                device,
                args.warmup,
                args.runs,
            )
            rms_auto = latency_us(
                lambda: add_rms_norm(x, residual, weight, use_native=True),
                device,
                args.warmup,
                args.runs,
            )
            silu_ref = latency_us(
                lambda: silu_mul(gate, up, use_native=False),
                device,
                args.warmup,
                args.runs,
            )
            silu_auto = latency_us(
                lambda: silu_mul(gate, up, use_native=True),
                device,
                args.warmup,
                args.runs,
            )
            results["cases"].append(
                {
                    "name": label,
                    "shape": list(shape),
                    "add_rms_norm_reference_us": round(rms_ref, 3),
                    "add_rms_norm_native_us": round(rms_auto, 3),
                    "add_rms_norm_speedup": round(rms_ref / rms_auto, 3),
                    "silu_mul_reference_us": round(silu_ref, 3),
                    "silu_mul_native_us": round(silu_auto, 3),
                    "silu_mul_speedup": round(silu_ref / silu_auto, 3),
                }
            )

    rendered = json.dumps(results, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

