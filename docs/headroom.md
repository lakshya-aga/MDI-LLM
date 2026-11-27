# How much headroom is left? Measuring the bound for a device cluster

Decoding reads every weight once per token, so each device's attained memory bandwidth
sets a floor on time per token. These tools measure the inputs and compute that floor
for a given cluster. Then run a real system (MDI-LLM, prima.cpp, exo) on the same
devices and compare:

- **gap under ~1.3×:** placement and kernels are nearly done; only ideas that lower the
  floor itself (quantization, sparsity, speculative decoding, batching) remain.
- **gap over ~2×:** find where the time goes (compute, network, disk, idle stages).

| Tool | Runs on | Measures |
|---|---|---|
| `native/bench/membw.c` | every device | attained read and copy bandwidth (GB/s) |
| `src/scripts/link_bench.py` | starter + each other device | RTT and link bandwidth |
| `src/scripts/best_time.py` | the laptop | latency bound, throughput cap, best splits |

All three need no PyTorch.

## Example cluster: MacBook Pro M5 (starter), MacBook Pro M1, Pixel 11 Pro, Nothing Phone (2a)

### 0. Before measuring

- Put every device on the **same Wi-Fi network**, ideally 5 GHz, near the router.
- Plug everything in. Phones throttle on battery saver and when hot; start them cool.
- Close other apps. On the Macs, quit browsers and IDEs while measuring.
- Turn off "battery optimization" for Termux on both phones and run `termux-wake-lock`.

### 1. Memory bandwidth on each device

**Macs** (from the repo root):

```bash
cc -O3 -pthread -o membw native/bench/membw.c
./membw
```

**Phones** (in Termux, [setup-android.md](setup-android.md) step 1 covers installing it):

```bash
pkg install clang
cc -O3 -pthread -o membw MDI-LLM/native/bench/membw.c
./membw -m 512
./membw -m 512 -t 4
```

Use `-m 512` on the phones to stay within their free memory. Try `-t` equal to the
number of big cores as well as the default; keep whichever is higher. Note the
**median read** figure. If `last` is much lower than `best`, the phone is throttling:
re-run with `-r 60` to see the sustained number, which is the honest one for long
generations.

On the Macs this measures the CPU. MDI-LLM's Mac nodes run on the GPU (MPS), which
shares the same unified memory but may reach a somewhat different share of it, so
treat the Mac figures as approximate.

### 2. Free memory on each device

`mem_gb` is what the model may use, not the total RAM:

- **Macs:** RAM minus what macOS and your apps use. Check *Activity Monitor → Memory*;
  on a 16 GB Mac, about 10–11 GB is realistic.
- **Phones:** run `free -g` in Termux and use the *available* column minus 1 GB.
  Android kills background apps that use too much, so stay conservative.

### 3. Link to each device

On each of the M1, the Pixel and the Nothing phone:

```bash
python link_bench.py serve
```

(`src/scripts/link_bench.py` in the repo; on the phones it runs in Termux with the
stock Python.) Then on the M5, once per device:

```bash
python src/scripts/link_bench.py ping <device-ip>
```

Use the device's IP (`ipconfig getifaddr en0` on a Mac, `ip -4 addr show wlan0` on a
phone). Note the **median RTT**. `best_time.py` prints the right `--bytes` for your
model (8192 for Llama 3.1 8B); the default is already 8192.

### 4. Fill in the cluster file and compute the bound

Copy `src/scripts/cluster_example.json`, replace the placeholder numbers with yours,
and list the devices in the order the pipeline visits them, starter first:

```bash
cd src
python scripts/best_time.py my_cluster.json
python scripts/best_time.py my_cluster.json --model llama-3.1-8b --weight-bits 4.5
python scripts/best_time.py my_cluster.json --model qwen2.5-32b --weight-bits 4.5
```

It prints four views:

1. **Starter alone:** if the model fits, this is usually the fastest option, and the
   cluster only adds hops.
2. **Your split:** add `"layers": [16, 8, 5, 3]` to the JSON to check a split you
   actually ran.
3. **Best split for latency:** one request; fills the fastest devices first.
4. **Best split for throughput:** all devices busy with several requests in flight.

Pick a model and precision that **does not fit on the M5 alone**. That is the case
clustering exists for, and the one prima.cpp targets.

### 5. Measure a real system against it

Run the same model, precision and context on the same four devices with MDI-LLM and
with prima.cpp. Record decode ms/token for one request (ignore the first token), and
divide by the latency bound. Also log: per-device time per token, any device that
swaps or throttles, and the hop share. If the gap is large, those logs show where
it goes.

## What the bound ignores

- **Compute.** Prefill and large batches are compute-bound, not bandwidth-bound; this
  bound is for decode only.
- **Disk.** If a device memory-maps more weights than it has RAM (prima.cpp does this
  on purpose), reads come from flash at a few GB/s; lower that device's `read_gbps`
  to model it.
- **Hop estimate.** Device-to-device delay is estimated from each device's RTT to the
  starter. For a measured link, run `serve` on the receiver and `ping` from the sender,
  then set `hop_ms` on the receiving device to half that RTT.
