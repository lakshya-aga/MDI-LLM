#!/usr/bin/env python3
"""
Numerical check of the native node against the PyTorch SecondaryNode.

The script plays the starter: it starts `mdi_node`, sends it random weights over
POST /init, then feeds random activations (prompt + decode steps, several interleaved
samples) through the ring and compares each reply with the PyTorch reference.

    python native/tests/test_numeric.py native/build/mdi_node
"""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sub import wire  # noqa: E402
from sub.model import Config  # noqa: E402
from sub.submodels import SecondaryNode  # noqa: E402

BIN = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "native" / "build" / "mdi_node")
HOST = "127.0.0.1"
NODE = {"addr": HOST, "communication": {"port": 18089, "starter_addr": HOST}, "inference": {"port_in": 15090, "port_out": 15091}}
STARTER = {"addr": HOST, "communication": {"port": 18088}, "inference": {"port_in": 15088, "port_out": 15089}}

CONFIGS = {
    "llama-gqa": dict(name="tiny-llama", block_size=64, vocab_size=100, n_layer=2, n_head=8, n_query_groups=2,
                      n_embd=128, intermediate_size=256, rotary_percentage=1.0, parallel_residual=False, bias=False,
                      norm_class_name="RMSNorm", mlp_class_name="LLaMAMLP", norm_eps=1e-5),
    "llama-mqa": dict(name="tiny-mqa", block_size=64, vocab_size=100, n_layer=2, n_head=4, n_query_groups=1,
                      n_embd=64, intermediate_size=160, rotary_percentage=1.0, parallel_residual=False, bias=False,
                      norm_class_name="RMSNorm", mlp_class_name="LLaMAMLP", rope_base=500000),
    "neox": dict(name="tiny-pythia", block_size=64, vocab_size=100, n_layer=2, n_head=4, n_embd=64,
                 rotary_percentage=0.25, parallel_residual=True, bias=True, norm_class_name="LayerNorm",
                 mlp_class_name="GptNeoxMLP"),
    "gemma-tanh": dict(name="tiny-Gemma", block_size=64, vocab_size=100, n_layer=2, n_head=4, n_query_groups=4,
                       n_embd=64, head_size=32, intermediate_size=128, rotary_percentage=1.0, parallel_residual=False,
                       bias=False, norm_class_name="RMSNorm", mlp_class_name="GemmaMLP", gelu_approximate="tanh"),
}


def recv_exact(s, n):
    b = b""
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise ConnectionError("closed")
        b += c
    return b


def recv_frame(s):
    head = recv_exact(s, wire.FRAME_HEADER.size)
    flags, code, ndim, idx, size = wire.decode_frame_header(head)
    shape = recv_exact(s, 4 * ndim)
    payload = recv_exact(s, size) if size else b""
    return wire.frame_to_msg(flags, code, shape, idx, payload)


def run_case(name, cfg_kwargs, weight_dtype, act_dtype, node_args, round_to=None):
    torch.manual_seed(0)
    cfg = Config(**cfg_kwargs)
    ref = SecondaryNode(cfg, cfg.n_layer).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.08)
        for n_, p in ref.named_parameters():
            if "norm" in n_ and n_.endswith("weight"):
                p.add_(1.0)
    max_seq = 48
    ref.max_seq_length = max_seq
    ref.init_rope_mask()
    # `round_to`: send weights already representable in that dtype (for --weights-dtype)
    sd = {k: v.to(round_to or weight_dtype).to(weight_dtype) for k, v in ref.state_dict().items()}
    # The reference uses the same (rounded) weights, computed in float32
    ref.load_state_dict({k: v.float() for k, v in sd.items()})

    proc = subprocess.Popen([BIN, "--addr", HOST, "--port", "18089", "--port-in", "15090", "--port-out", "15091",
                             "--once", *node_args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(50):
            try:
                st = requests.get(f"http://{HOST}:18089/status", timeout=1).json()
                break
            except requests.RequestException:
                time.sleep(0.1)
        assert st["native"] and not st["initialized"]

        meta = {"role": "secondary:0", "prev_node": STARTER, "next_node": STARTER, "model_config": cfg.asdict(),
                "n_nodes": 2, "n_local_layers": cfg.n_layer, "n_samples": 3, "max_seq_length": max_seq}
        lsock = socket.socket(); lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind((HOST, 15088)); lsock.listen(1)
        r = requests.post(f"http://{HOST}:18089/init", data=wire.encode_safetensors(sd, meta), timeout=60)
        assert r.status_code == 200, r.text
        for _ in range(50):  # the node opens its input socket right after /init
            out = socket.socket()  # the node only checks the peer IP, not the port
            try:
                out.connect((HOST, 15090))
                break
            except ConnectionRefusedError:
                out.close(); time.sleep(0.1)
        inp, _ = lsock.accept()

        caches, pos = {}, {}
        worst = 0.0
        g = torch.Generator().manual_seed(1)
        schedule = [(s, 5 + 2 * s) for s in range(3)] + [(s, 1) for _ in range(6) for s in range(3)]
        for sample, T in schedule:
            x = torch.randn(1, T, cfg.n_embd, generator=g)
            if sample not in caches:
                caches[sample] = [b.attn.build_kv_cache(1, max_seq, ref.cos.size(-1), dtype=torch.float32)
                                  for b in ref.transformer.h]
                pos[sample] = torch.arange(0, T)
            for b, c in zip(ref.transformer.h, caches[sample]):
                b.attn.kv_cache = c
            with torch.inference_mode():
                y_ref = ref(x.to(act_dtype).float(), input_pos=pos[sample])
            pos[sample] = pos[sample][-1:] + 1
            out.sendall(wire.encode_frame(sample, x.to(act_dtype), False))
            m = recv_frame(inp)
            assert m["sample_index"] == sample and m["data"].dtype == act_dtype
            err = ((m["data"].float() - y_ref.to(act_dtype).float()).abs().max() / y_ref.abs().max()).item()
            worst = max(worst, err)
        out.sendall(wire.encode_frame(0, None, True))
        m = recv_frame(inp)
        assert m["stop"] and m["sample_index"] == 0
        requests.put(f"http://{HOST}:18089/stop", timeout=5)
        out.close(); inp.close(); lsock.close()
        tol = 1e-4 if act_dtype == torch.float32 else 1e-2
        ok = worst < tol
        print(f"{'PASS' if ok else 'FAIL'}  {name:11s} weights={str(weight_dtype)[6:]:8s} "
              f"acts={str(act_dtype)[6:]:8s} {' '.join(node_args):28s} max rel err {worst:.2e}")
        return ok
    finally:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        if proc.returncode not in (0, None):
            print(proc.stdout.read())


if __name__ == "__main__":
    results = []
    for name, cfg in CONFIGS.items():
        results.append(run_case(name, cfg, torch.float32, torch.float32, []))
    results.append(run_case("llama-gqa", CONFIGS["llama-gqa"], torch.bfloat16, torch.float32, []))
    results.append(run_case("llama-gqa", CONFIGS["llama-gqa"], torch.float16, torch.float16, []))
    results.append(run_case("llama-gqa", CONFIGS["llama-gqa"], torch.float32, torch.bfloat16, ["--threads", "1"]))
    # fp32 weights converted to bf16/f16 storage by the node (values exactly representable)
    results.append(run_case("llama-gqa", CONFIGS["llama-gqa"], torch.float32, torch.float32,
                            ["--weights-dtype", "bfloat16"], round_to=torch.bfloat16))
    results.append(run_case("neox", CONFIGS["neox"], torch.float32, torch.float32,
                            ["--weights-dtype", "float16"], round_to=torch.float16))
    print("all passed" if all(results) else "FAILURES")
    sys.exit(0 if all(results) else 1)
