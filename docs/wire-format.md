# MDI-LLM wire format

Nodes used to exchange pickled Python objects, which only Python nodes can read. Every
message now uses a simple binary format, so nodes in other languages can join the ring
(for example the C++ node in [`native/`](../native), which the Android app uses). The
Python implementation is [`src/sub/wire.py`](../src/sub/wire.py). The C++
implementation is [`native/src/net.cpp`](../native/src/net.cpp) together with
`parse_safetensors` in [`native/src/tensor.cpp`](../native/src/tensor.cpp).

All integers are little-endian.

## HTTP control API (per node)

| Request | Body | Response |
|---|---|---|
| `GET /` | | The node's own entry from the nodes config (JSON) |
| `GET /status` | | `{"role", "has_chunk", "initialized", ...}`; native nodes add `"native": true`, `"state"` and timing stats |
| `POST /init` | safetensors (see below) | 200 once the model is loaded; the node then opens its ring sockets |
| `PUT /stop` | empty | 200; the node leaves the ring. Native nodes go back to waiting for the next `/init` |

The starter calls `GET /status` before `POST /init`. If `has_chunk` is true, the
starter sends no weights and the node loads its local chunk.

## `POST /init` body: safetensors

The body is a valid [safetensors](https://github.com/huggingface/safetensors) file:

```
uint64       N, the header length
N bytes      JSON header, padded with spaces to a multiple of 8
...          tensor data; offsets are relative to the end of the header
```

The header maps tensor names to `{"dtype": "F32" | "F16" | "BF16" | "I64" | "I32",
"shape": [...], "data_offsets": [begin, end]}`. Its `__metadata__` entry holds one
string, `"mdi_init"`, which is a JSON object with these fields:

| Field | Meaning |
|---|---|
| `role` | `"secondary:<i>"` |
| `prev_node`, `next_node` | Entries from the nodes config (`addr`, `inference.port_in`, ...) |
| `model_config` | `Config.asdict()` (LitGPT config) |
| `n_nodes`, `n_local_layers`, `n_samples` | Ring size, layers held by this node, samples to generate |
| `max_seq_length` | KV-cache length, or `null` for the model's block size |

The tensors are the node's chunk (`transformer.h.<j>.…`, with `j` counted from 0). If
the node has its own chunk, the body contains no tensors. Local chunks for native nodes use the
same format: `scripts/export_chunks.py` converts the `.pth` chunks.

## Activation frames (TCP ring)

Node *k* connects from its `inference.port_out` to node *k+1*'s `inference.port_in`.
Receivers accept a connection only if it comes from the previous node's IP address.
Each message is one frame:

```
offset  size   field
0       4      magic "MDIA"
4       1      version = 1
5       1      flags: bit 0 = stop (end of this sample)
6       1      dtype: 0 f32, 1 f16, 2 bf16, 3 i64, 4 i32
7       1      ndim
8       4      sample_index (uint32)
12      8      payload size in bytes (uint64)
20      4*ndim shape (uint32 each)
...            payload: C-contiguous tensor data
```

Activations have shape `(1, T, n_embd)`. T is the prompt length on a sample's first
pass and 1 on every later pass. Each node replies in the dtype it received, so a float16
starter gets float16 back. A stop frame has `ndim = 0` and no payload. Nodes forward it
unchanged and may free that sample's KV cache.
