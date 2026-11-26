#!/usr/bin/env python3
"""
Convert model chunks (`chunks/<N>nodes/*.pth`, from prepare_model.py) to safetensors,
the format the native node (Android app / `mdi_node --chunk`) memory-maps.

    python scripts/export_chunks.py checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0/chunks/3nodes
    python scripts/export_chunks.py <chunks dir> --dtype bfloat16   # halve the size

Keep the .pth files: the starter uses them to tell that the model was already split.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sub import wire  # noqa: E402

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chunks", type=Path, help="directory with model_*.pth (or a single .pth file)")
    ap.add_argument("--dtype", choices=DTYPES, default=None, help="convert float weights (default: keep)")
    args = ap.parse_args()

    files = [args.chunks] if args.chunks.is_file() else sorted(args.chunks.glob("model_*.pth"))
    if not files:
        sys.exit(f"No model_*.pth files in {args.chunks}")
    for f in files:
        sd = wire.load_chunk(f)
        if args.dtype:
            sd = {k: v.to(DTYPES[args.dtype]) if v.is_floating_point() else v for k, v in sd.items()}
        out = f.with_suffix(".safetensors")
        wire.save_safetensors(out, sd)
        size = out.stat().st_size / 2**20
        print(f"{f.name} -> {out.name} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
