#!/usr/bin/env python3
"""
Best possible decode time per token for a model split across a set of devices.

Decoding one token reads every weight (and the KV cache) once, so a device holding B
bytes with attained read bandwidth W spends at least B / W on it. With one request in
flight the stages run one after another, so

    latency bound  = sum over devices of bytes_i / W_i  +  one-way delay of every hop
    throughput cap = 1 / max over devices of bytes_i / W_i   (pipeline kept full)

No real system reaches these numbers; the point is the gap. Run a system (MDI-LLM,
prima.cpp, exo) on the same devices and divide its measured time per token by the
bound. Near 1x means placement and kernels have little left to give.

Inputs come from a JSON file (see docs/headroom.md and cluster_example.json):
devices in pipeline order, the first one being the starter, each with `read_gbps`
from native/bench/membw.c, `rtt_ms` to the starter from link_bench.py, and
`mem_gb` free for the model. Standard library only.

    python scripts/best_time.py scripts/cluster_example.json
    python scripts/best_time.py cluster.json --model llama-3.1-70b --weight-bits 4.5
"""

import argparse
import itertools
import json
from pathlib import Path

# Llama-style dimensions: layers, hidden, heads, KV heads, MLP width, vocab
PRESETS = {
    "tinyllama-1.1b": (22, 2048, 32, 4, 5632, 32000),
    "llama-3.2-3b": (28, 3072, 24, 8, 8192, 128256),
    "llama-3.1-8b": (32, 4096, 32, 8, 14336, 128256),
    "qwen2.5-32b": (64, 5120, 40, 8, 27648, 152064),
    "llama-3.1-70b": (80, 8192, 64, 8, 28672, 128256),
}
GB = 1e9


def dims_from_hf(path: Path):
    c = json.loads(path.read_text())
    return (c["num_hidden_layers"], c["hidden_size"], c["num_attention_heads"],
            c.get("num_key_value_heads", c["num_attention_heads"]),
            c["intermediate_size"], c["vocab_size"])


def model_bytes(dims, weight_bits, kv_bits, context):
    n_layer, d, n_head, n_kv, d_ff, vocab = dims
    hs = d // n_head
    block_params = d * n_head * hs + 2 * d * n_kv * hs + n_head * hs * d + 3 * d * d_ff
    return {
        "n_layer": n_layer,
        "n_embd": d,
        "layer_w": block_params * weight_bits / 8,                # weights per layer
        "layer_kv": 2 * n_kv * hs * context * kv_bits / 8,        # KV per layer, 1 request
        "head": vocab * d * weight_bits / 8,                      # lm_head, read every token
        "embed": vocab * d * weight_bits / 8,                     # stored, 1 row read per token
    }


def hop_s(a, b):
    """One-way delay a -> b, estimated from each device's RTT to the starter.

    All devices sit on one Wi-Fi network, so traffic between two of them goes through
    the access point; half of each device's RTT approximates its leg to the AP.
    For a measured a -> b link, give the receiving device `hop_ms` instead.
    """
    if "hop_ms" in b:
        return b["hop_ms"] / 1e3
    return (a.get("rtt_ms", 0.0) + b.get("rtt_ms", 0.0)) / 4 / 1e3


def evaluate(devs, counts, m, samples):
    """Per-stage read time, memory, and the two bounds for a given layer split."""
    used = [(d, c) for i, (d, c) in enumerate(zip(devs, counts)) if c > 0 or i == 0]
    stages, mems = [], []
    for i, (d, c) in enumerate(used):
        read = c * (m["layer_w"] + m["layer_kv"]) + (m["head"] if i == 0 else 0)
        stages.append(read / (d["read_gbps"] * GB))
        mems.append(c * (m["layer_w"] + samples * m["layer_kv"])
                    + (m["head"] + m["embed"] if i == 0 else 0))
    ring = [d for d, _ in used]
    hops = sum(hop_s(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))) \
        if len(ring) > 1 else 0.0
    return {
        "used": [d["name"] for d in ring],
        "counts": [c for _, c in used],
        "stages": stages,
        "mems": mems,
        "hops": hops,
        "latency": sum(stages) + hops,
        "tput": 1 / max(stages),
        "fits": all(mm <= d["mem_gb"] * GB for mm, d in zip(mems, ring)),
    }


def caps(devs, m, samples):
    """Max layers each device can hold given its free memory (starter also holds head)."""
    out = []
    for i, d in enumerate(devs):
        free = d["mem_gb"] * GB - (m["head"] + m["embed"] if i == 0 else 0)
        out.append(max(0, int(free // (m["layer_w"] + samples * m["layer_kv"]))))
    return out


def best_latency(devs, m, samples):
    """Try every subset of devices that includes the starter; fill fastest first."""
    cap = caps(devs, m, samples)
    best = None
    for r in range(len(devs)):
        for rest in itertools.combinations(range(1, len(devs)), r):
            idx = [0, *rest]
            counts, left = [0] * len(devs), m["n_layer"]
            for i in sorted(idx, key=lambda i: -devs[i]["read_gbps"]):
                counts[i] = min(cap[i], left)
                left -= counts[i]
            if left or any(counts[i] == 0 for i in rest):
                continue
            ev = evaluate(devs, counts, m, samples)
            if best is None or ev["latency"] < best["latency"]:
                best = ev
    return best


def best_throughput(devs, m, samples):
    """Smallest bottleneck stage time with all devices used (binary search)."""
    cap = caps(devs, m, samples)
    per = m["layer_w"] + m["layer_kv"]

    def counts_for(t):
        out = []
        for i, d in enumerate(devs):
            budget = t * d["read_gbps"] * GB - (m["head"] if i == 0 else 0)
            out.append(max(0, min(cap[i], int(budget // per))))
        return out

    lo, hi = 0.0, 10.0
    if sum(counts_for(hi)) < m["n_layer"]:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if sum(counts_for(mid)) >= m["n_layer"] else (mid, hi)
    counts, extra = counts_for(hi), sum(counts_for(hi)) - m["n_layer"]
    for i in reversed(range(len(counts))):  # trim surplus from the slowest-to-fill end
        take = min(extra, counts[i])
        counts[i] -= take
        extra -= take
    return evaluate(devs, counts, m, samples)


def show(title, ev, devs):
    if ev is None:
        print(f"\n{title}: does not fit in the devices' free memory")
        return
    print(f"\n{title}" + ("" if ev["fits"] else "   ** exceeds mem_gb on some device **"))
    print(f"  {'device':<16}{'layers':>7}{'mem GB':>9}{'read ms':>9}")
    for name, c, mm, s in zip(ev["used"], ev["counts"], ev["mems"], ev["stages"]):
        print(f"  {name:<16}{c:>7}{mm / GB:>9.2f}{s * 1e3:>9.1f}")
    lat = ev["latency"]
    print(f"  hops (one-way, all) {ev['hops'] * 1e3:.1f} ms = {ev['hops'] / lat:.0%} of the bound")
    print(f"  latency bound {lat * 1e3:.1f} ms/token  ({1 / lat:.1f} tok/s, one request)")
    print(f"  throughput cap {ev['tput']:.1f} tok/s  (pipeline full, no batching)")


def main(a):
    spec = json.loads(a.cluster.read_text())
    devs = spec["devices"]
    model = a.model or spec.get("model", "llama-3.1-8b")
    dims = dims_from_hf(Path(model)) if model.endswith(".json") else PRESETS[model]
    wb = a.weight_bits or spec.get("weight_bits", 16)
    kb = a.kv_bits or spec.get("kv_bits", 16)
    ctx = a.context or spec.get("context", 1024)
    samples = spec.get("samples", 1)
    m = model_bytes(dims, wb, kb, ctx)

    total = m["n_layer"] * m["layer_w"] + m["head"] + m["embed"]
    print(f"{model}: {m['n_layer']} layers, {wb} bits/weight, KV {kb} bits x {ctx} tokens, "
          f"{samples} request(s) of KV held")
    print(f"weights {total / GB:.2f} GB ({m['layer_w'] / GB * 1e3:.0f} MB/layer, "
          f"lm_head {m['head'] / GB:.2f} GB); activation per hop {m['n_embd'] * 2} B in "
          f"float16 (pass this as link_bench.py --bytes)")
    for d in devs:
        print(f"  {d['name']:<16} {d['read_gbps']:6.1f} GB/s  {d['mem_gb']:5.1f} GB free  "
              f"RTT {d.get('rtt_ms', 0):.1f} ms")

    solo = evaluate(devs[:1], [m["n_layer"]], m, samples)
    show(f"{devs[0]['name']} alone", solo, devs)
    if "layers" in spec:
        show("Your split (from the JSON `layers`)", evaluate(devs, spec["layers"], m, samples), devs)
    show("Best split for latency (one request)", best_latency(devs, m, samples), devs)
    show("Best split for throughput (all devices, pipeline full)",
         best_throughput(devs, m, samples), devs)
    print("\nCompare a real run's ms/token with these. Gap = measured / bound.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cluster", type=Path, help="JSON file describing the devices")
    p.add_argument("--model", help=f"preset ({', '.join(PRESETS)}) or a HF config.json")
    p.add_argument("--weight-bits", type=float, help="bits per weight (16; ~4.5 for Q4_K)")
    p.add_argument("--kv-bits", type=float, help="bits per KV element (default 16)")
    p.add_argument("--context", type=int, help="tokens in the KV cache (default 1024)")
    main(p.parse_args())
