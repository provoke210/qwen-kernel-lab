from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow direct execution from a source checkout before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_kernel_lab.integration import replace_qwen_rmsnorm


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate custom RMSNorm in a HF Qwen model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", default="Explain KV cache in one sentence.")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install integration dependencies: pip install -e '.[qwen]'") from exc

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    baseline = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    candidate = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    replacements = replace_qwen_rmsnorm(candidate)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)

    with torch.inference_mode():
        reference_logits = baseline(**inputs).logits.float()
        candidate_logits = candidate(**inputs).logits.float()

    diff = (reference_logits - candidate_logits).abs()
    print(f"replaced_modules={len(replacements)}")
    print(f"max_abs_error={diff.max().item():.8g}")
    print(f"mean_abs_error={diff.mean().item():.8g}")
    print("top1_equal=", torch.equal(reference_logits.argmax(-1), candidate_logits.argmax(-1)))


if __name__ == "__main__":
    main()

