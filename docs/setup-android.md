# Running MDI-LLM across a laptop and Android phones

The laptop is the **starter** node: it holds the model files, the tokenizer, the
first/last layers, and it sends every phone its share of the weights over HTTP. The
phones are **secondary** nodes running in [Termux](https://termux.dev). They only need
PyTorch and a few pure-Python packages, and no model files.

```
laptop (starter) ──► phone 0 (secondary:0) ──► phone 1 (secondary:1) ──┐
      ▲                                                                 │
      └──────────────────────────── activations ◄──────────────────────┘
```

## 1. Phones: install Termux and the node dependencies

1. Install Termux from **F-Droid** or the GitHub releases. The Play Store build is
   outdated.
2. In Termux:

   ```bash
   pkg update && pkg upgrade
   pkg install python python-torch python-numpy git
   pip install cherrypy requests pyyaml typing_extensions
   git clone https://github.com/lakshya-aga/MDI-LLM.git
   ```

   If `pkg` doesn't find `python-torch`, enable the user repository first with
   `pkg install tur-repo`. Don't `pip install torch` in Termux: no prebuilt wheel
   exists for it.
3. Check the install with `python -c "import torch; print(torch.__version__)"`.
4. Keep the node alive while it runs:
   - run `termux-wake-lock`, or use the Termux notification's *Acquire wakelock*;
   - in Android settings, turn off battery optimization for Termux;
   - keep the phone plugged in, because long generations drain the battery quickly.
5. Note the phone's Wi-Fi IP address: `ip -4 addr show wlan0` (or check
   Settings → Wi-Fi → network details).

## 2. Laptop: install the full requirements and prepare the model

```bash
pip install -r requirements.txt
./src/prepare_model.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 --n-nodes 3
```

The model ends up in `src/checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0`, with the
chunks for 3 nodes in `chunks/3nodes/`. The weights keep the dtype they were downloaded
in (bfloat16 for TinyLlama), so each phone downloads about 0.65 GB and converts it to
float32 locally. If you plan to copy chunks onto the phones (`--chunk`, see below), add
`--dtype float32` here instead. The phones can then memory-map the file as-is rather
than converting it into RAM.

Check that the memory needed per device fits your phones:

```bash
cd src && python scripts/mem_breakdown.py --model tiny-llama-1.1b-chat --nodes 3 --seq-len 512
```

At float32 with a 512-token context, TinyLlama needs about 1.35 GB per phone for the
model (1.31 GB of weights and 0.02 GB of KV cache), plus about 200–300 MB for Python
and PyTorch. Secondary nodes default to float32 on CPU. `--dtype bfloat16` on
`secondary.py` halves the weights, but only if your phone's PyTorch build has fast
bfloat16 CPU kernels. Many don't, so measure before relying on it.

## 3. Write the nodes configuration

Copy [`src/settings_distr/phones_example.json`](../src/settings_distr/phones_example.json)
and set the IP addresses:

- `starter.addr`: the laptop's IP address on the Wi-Fi network;
- each `secondary[i].addr`: phone *i*'s IP address;
- each `secondary[i].communication.starter_addr`: the laptop's IP address.

Use the same file on every device. The order of the `secondary` list is the order of
the ring.

## 4. Run

Start the phones first, one secondary index each:

```bash
# phone 0
cd MDI-LLM/src && python secondary.py --nodes-config settings_distr/phones.json 0 -v
# phone 1
cd MDI-LLM/src && python secondary.py --nodes-config settings_distr/phones.json 1 -v
```

Then start the laptop:

```bash
cd src && python starter.py \
  --ckpt checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --nodes-config settings_distr/phones.json \
  --n-samples 3 --n-tokens 200 --sequence-length 512 \
  --prompt "Who are you?" -v
```

- The phones receive their weights during initialization, which takes a while over
  Wi-Fi. The starter asks each node first and only sends weights to nodes that have
  no local chunk. To skip the transfer, copy `chunks/3nodes/model_secondary<i>.pth` to
  phone *i* (e.g. `adb push`) and start it with `--chunk <path>`.
- `--n-samples` should be at least the number of nodes, so every device stays busy
  (recurrent pipelining).
- `--sequence-length` caps the KV cache. Use prompt length + `--n-tokens`.
- On a Mac, `--device mps` runs the starter on the GPU in float16. The phones stay on
  float32, and activations are converted at each hop.
- Add `--top-k 1` for greedy (deterministic) decoding, which is useful to check that
  a multi-device run matches a single-device one.

## Troubleshooting

- **The starter retries "Unable to reach node" forever:** all devices must be on the
  same network, and phones must be able to accept incoming connections. Guest Wi-Fi
  and many public networks block device-to-device traffic. A phone hotspot, with the
  laptop and the other phone joined to it, works well. On macOS, allow incoming
  connections for Python if the firewall asks.
- **The node is killed during initialization:** the phone ran out of memory. Use a
  shorter `--sequence-length`, more nodes, or `--chunk` (memory-mapped weights).
- **Generation stalls after the screen turns off:** the wakelock isn't held, or
  battery optimization is still on for Termux.
