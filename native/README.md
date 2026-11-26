# Native secondary node (C++)

A dependency-free C++17 implementation of an MDI-LLM secondary node. The Android app
([`../android`](../android), [docs](../docs/android-app.md)) embeds it. It also builds as
a command-line replacement for `src/secondary.py`.

| File | Contents |
|---|---|
| `src/model.*` | Transformer blocks (Llama/Gemma/GPT-NeoX layouts), grouped KV cache, RoPE |
| `src/tensor.*` | safetensors parsing, f32/f16/bf16 storage, threaded NEON matmul |
| `src/net.*` | Sockets and activation frames ([wire format](../docs/wire-format.md)) |
| `src/node.*` | HTTP control API and the ring loop |
| `src/main.cpp`, `src/jni.cpp` | CLI and Android JNI entry points |

Build and test it:

```bash
cmake -B build && cmake --build build -j        # or: clang++ -std=c++17 -O3 -pthread src/{json,tensor,model,net,node,main}.cpp -o build/mdi_node
python tests/test_numeric.py build/mdi_node     # compares hidden states with PyTorch
```

`tests/test_numeric.py` needs the repo's Python requirements. It plays the starter
against one node and checks several architectures and dtypes against
`sub.submodels.SecondaryNode`.

The native node doesn't support mixture-of-experts MLPs (`LLaMAMoE`).
