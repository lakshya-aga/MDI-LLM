#!/usr/bin/env python3
"""
Analytic per-node memory breakdown for MDI-LLM.

For each number of nodes, report what every node has to hold in memory:
- weights of its model chunk (counted exactly, on the "meta" device - nothing is
  allocated and no checkpoint is needed);
- KV caches - one per sample, since every sample visits every node, and
  recurrent pipelining needs n_samples >= n_nodes to keep all nodes busy;
- causal mask and RoPE caches (duplicated on every node).

The KV cache is reported both with the old layout (k/v expanded to all query heads)
and the current one (one k/v head per query group, GQA-aware).

Runtime overhead (Python interpreter, PyTorch/CUDA libraries, HTTP server, sockets)
is NOT included - measure it on the target device, e.g. with `mem_monitor.py`.

Examples (from the `src` directory):
    python scripts/mem_breakdown.py --model tiny-llama-1.1b-chat --dtype float32
    python scripts/mem_breakdown.py --model-config checkpoints/.../model_config.yaml \
        --seq-len 1024 --samples-per-node 2
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(os.path.dirname(__file__)).parent))

from sub.config import DTYPE_TORCH_MAPPING, N_LAYERS_NODES  # noqa: E402
from sub.model import Config  # noqa: E402
from sub.submodels import SecondaryNode, StarterNode  # noqa: E402

GB = 1024**3


def param_bytes(module: torch.nn.Module, bytes_per_el: int) -> int:
    return sum(p.numel() for p in module.parameters()) * bytes_per_el


def chunk_weights(cfg: Config, n_layers: int, starter: bool, bytes_per_el: int) -> int:
    cls = StarterNode if starter else SecondaryNode
    with torch.device("meta"):
        model = cls(cfg, n_layers)
    return param_bytes(model, bytes_per_el)


def kv_bytes_per_layer(cfg: Config, seq_len: int, bytes_per_el: int, grouped: bool) -> int:
    if grouped:
        heads = cfg.n_query_groups
    else:  # Previous layout: k/v expanded to n_head (except for MQA)
        heads = 1 if cfg.n_query_groups == 1 else cfg.n_head
    k_last = cfg.head_size  # rope_cache_length + head_size - rope_n_elem == head_size
    return 2 * heads * seq_len * k_last * bytes_per_el


def layer_split(n_layer: int, n_nodes: int):
    """Number of transformer blocks on the starter and on each secondary node."""
    if n_nodes == 1:
        return n_layer, []
    split = N_LAYERS_NODES[n_nodes][n_layer]
    return split["N_LAYERS_START"], [split["N_LAYERS_SECONDARY"]] * (n_nodes - 1)


def main(args):
    if args.model_config:
        cfg = Config.from_file(args.model_config)
    else:
        cfg = Config.from_name(args.model)
    seq_len = args.seq_len or cfg.block_size
    bpe = torch.tensor([], dtype=DTYPE_TORCH_MAPPING[args.dtype]).element_size()

    available = sorted(n for n, d in N_LAYERS_NODES.items() if cfg.n_layer in d)
    nodes_list = args.nodes or available

    print(
        f"{cfg.name}: {cfg.n_layer} layers, n_embd={cfg.n_embd}, n_head={cfg.n_head}, "
        f"n_query_groups={cfg.n_query_groups} | dtype={args.dtype}, seq_len={seq_len}"
    )
    mask_b = seq_len * seq_len  # bool
    rope_b = 2 * seq_len * cfg.rope_n_elem * 4  # cos + sin, float32
    print(f"Duplicated on every node: mask {mask_b / 2**20:.1f} MiB, RoPE {rope_b / 2**20:.1f} MiB")
    print(
        "Not included: Python/PyTorch/CUDA runtime, HTTP server, sockets "
        "(measure on the device)\n"
    )

    hdr = (
        f"{'nodes':>5} {'samples':>7} {'node':>10} {'layers':>6} {'weights':>8} "
        f"{'KV old':>8} {'KV new':>8} {'total old':>9} {'total new':>9}"
    )
    print(hdr + "   (GB)")
    print("-" * len(hdr))
    for n in nodes_list:
        if n not in available:
            print(f"{n:>5}  (no layer split defined in N_LAYERS_NODES for {cfg.n_layer} layers)")
            continue
        n_samples = args.samples or n * args.samples_per_node
        n_start, n_sec = layer_split(cfg.n_layer, n)
        rows = [("starter", n_start, True)] + [
            (f"secondary{i}", nl, False) for i, nl in enumerate(n_sec)
        ]
        tot_old = tot_new = 0.0
        for name, nl, is_start in rows:
            w = chunk_weights(cfg, nl, is_start, bpe)
            kv_old = nl * n_samples * kv_bytes_per_layer(cfg, seq_len, bpe, grouped=False)
            kv_new = nl * n_samples * kv_bytes_per_layer(cfg, seq_len, bpe, grouped=True)
            t_old = w + kv_old + mask_b + rope_b
            t_new = w + kv_new + mask_b + rope_b
            tot_old += t_old
            tot_new += t_new
            print(
                f"{n:>5} {n_samples:>7} {name:>10} {nl:>6} {w / GB:>8.2f} "
                f"{kv_old / GB:>8.2f} {kv_new / GB:>8.2f} {t_old / GB:>9.2f} {t_new / GB:>9.2f}"
            )
        print(f"{'':>5} {'':>7} {'all nodes':>10} {'':>6} {'':>8} {'':>8} {'':>8} "
              f"{tot_old / GB:>9.2f} {tot_new / GB:>9.2f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--model", default="tiny-llama-1.1b-chat", help="litgpt config name")
    src.add_argument("--model-config", type=Path, help="path to a model_config.yaml")
    parser.add_argument(
        "--dtype", default="float32", choices=list(DTYPE_TORCH_MAPPING),
        help="weights/KV dtype (the Jetson TX2 runs float32)",
    )
    parser.add_argument("--seq-len", type=int, default=None,
                        help="KV cache length (default: model block size; see --sequence-length in starter.py)")
    parser.add_argument("--nodes", type=int, nargs="+", default=None,
                        help="node counts to report (default: all defined in N_LAYERS_NODES)")
    parser.add_argument("--samples", type=int, default=None,
                        help="fixed number of samples (overrides --samples-per-node)")
    parser.add_argument("--samples-per-node", type=int, default=1,
                        help="samples = nodes * this (pipelining needs samples >= nodes)")
    main(parser.parse_args())
