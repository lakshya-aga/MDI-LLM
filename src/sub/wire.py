"""
Language-neutral wire format used between MDI-LLM nodes.

Replaces the pickled Python objects that were sent before, so that nodes written in
other languages (e.g., the C++ node in `native/`, used by the Android app) can join the
ring. Everything is little-endian. The full spec is in `docs/wire-format.md`.

Activation frames (TCP ring, one per message):

    offset  size  field
    0       4     magic b"MDIA"
    4       1     version (1)
    5       1     flags (bit 0: stop)
    6       1     dtype code (see DTYPES)
    7       1     ndim
    8       4     sample_index (uint32)
    12      8     payload size in bytes (uint64)
    20      4*n   shape (uint32 each)
    ...           payload (raw, C-contiguous tensor data)

Initialization (HTTP POST /init body) and model chunks on disk are safetensors files:

    8 bytes   N = JSON header length (uint64)
    N bytes   JSON: {"__metadata__": {"mdi_init": "<JSON string>"},
                     "<tensor name>": {"dtype": "F32", "shape": [...],
                                       "data_offsets": [begin, end]}, ...}
    ...       tensor data (offsets relative to the end of the header)
"""

import json
import struct
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union

import torch

MAGIC = b"MDIA"
VERSION = 1
FLAG_STOP = 1
FRAME_HEADER = struct.Struct("<4sBBBBIQ")  # 20 bytes

# code -> (torch dtype, safetensors name, element size)
DTYPES = {
    0: (torch.float32, "F32", 4),
    1: (torch.float16, "F16", 2),
    2: (torch.bfloat16, "BF16", 2),
    3: (torch.int64, "I64", 8),
    4: (torch.int32, "I32", 4),
}
_CODE = {v[0]: k for k, v in DTYPES.items()}
_ST_NAME = {v[0]: v[1] for v in DTYPES.values()}
_ST_DTYPE = {v[1]: v[0] for v in DTYPES.values()}


# ---- Activation frames ---------------------------------------------------------------


def encode_frame(sample_index: int, data: Optional[torch.Tensor], stop: bool) -> bytes:
    """Serialize one ring message; `data` is ignored for stop messages."""
    if stop or data is None or not isinstance(data, torch.Tensor):
        return FRAME_HEADER.pack(MAGIC, VERSION, FLAG_STOP if stop else 0, 0, 0, sample_index, 0)
    t = data.detach().to("cpu").contiguous()
    if t.dtype not in _CODE:
        raise TypeError(f"Unsupported dtype on the wire: {t.dtype}")
    payload = _tensor_bytes(t)
    head = FRAME_HEADER.pack(
        MAGIC, VERSION, 0, _CODE[t.dtype], t.dim(), sample_index, len(payload)
    )
    return head + struct.pack(f"<{t.dim()}I", *t.shape) + payload


def decode_frame_header(buf: bytes) -> Tuple[int, int, int, int, int]:
    """Returns (flags, dtype code, ndim, sample_index, payload size)."""
    magic, version, flags, code, ndim, sample_index, size = FRAME_HEADER.unpack(buf)
    if magic != MAGIC:
        raise ValueError(f"Bad frame magic {magic!r} (is the peer running an old version?)")
    if version != VERSION:
        raise ValueError(f"Unsupported frame version {version}")
    return flags, code, ndim, sample_index, size


def frame_to_msg(flags: int, code: int, shape_buf: bytes, sample_index: int, payload: bytes) -> Dict[str, Any]:
    """Build the message dict used by GPTServer from a decoded frame."""
    if flags & FLAG_STOP:
        return {"sample_index": sample_index, "data": "", "stop": True}
    ndim = len(shape_buf) // 4
    shape = struct.unpack(f"<{ndim}I", shape_buf)
    data = _tensor_from_bytes(payload, DTYPES[code][0], shape)
    return {"sample_index": sample_index, "data": data, "stop": False}


# ---- safetensors (init message + chunks) --------------------------------------------


def encode_safetensors(
    tensors: Optional[Dict[str, torch.Tensor]] = None, metadata: Optional[Dict[str, Any]] = None
) -> bytes:
    """Serialize tensors (+ a JSON-able metadata dict) to safetensors bytes."""
    parts = []
    _write_safetensors(_BytesSink(parts), tensors or {}, metadata)
    return b"".join(parts)


def save_safetensors(
    path: Union[str, Path], tensors: Dict[str, torch.Tensor], metadata: Optional[Dict[str, Any]] = None
):
    with open(path, "wb") as f:
        _write_safetensors(f, tensors, metadata)


def decode_safetensors(buf: Union[bytes, memoryview]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Parse safetensors bytes. Returns (tensors, metadata), metadata from "mdi_init"."""
    buf = memoryview(buf)
    (n,) = struct.unpack("<Q", buf[:8])
    header = json.loads(bytes(buf[8 : 8 + n]))
    meta = header.pop("__metadata__", {}) or {}
    metadata = json.loads(meta["mdi_init"]) if "mdi_init" in meta else {}
    base = 8 + n
    tensors = {}
    for name, info in header.items():
        b, e = info["data_offsets"]
        tensors[name] = _tensor_from_bytes(
            buf[base + b : base + e], _ST_DTYPE[info["dtype"]], info["shape"]
        )
    return tensors, metadata


def load_safetensors(path: Union[str, Path]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    with open(path, "rb") as f:
        return decode_safetensors(f.read())


# ---- helpers -------------------------------------------------------------------------


class _BytesSink:
    def __init__(self, parts):
        self.parts = parts

    def write(self, b):
        self.parts.append(bytes(b))


def _write_safetensors(f: BinaryIO, tensors: Dict[str, torch.Tensor], metadata):
    header: Dict[str, Any] = {}
    if metadata is not None:
        header["__metadata__"] = {"mdi_init": json.dumps(metadata)}
    blobs = []
    off = 0
    for name, t in tensors.items():
        t = t.detach().to("cpu").contiguous()
        if t.dtype not in _ST_NAME:
            raise TypeError(f"Unsupported dtype for {name}: {t.dtype}")
        b = _tensor_bytes(t)
        header[name] = {"dtype": _ST_NAME[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)  # keep tensor data 8-byte aligned
    f.write(struct.pack("<Q", len(h)))
    f.write(h)
    for b in blobs:
        f.write(b)


def _tensor_bytes(t: torch.Tensor) -> bytes:
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()


def _tensor_from_bytes(buf, dtype: torch.dtype, shape) -> torch.Tensor:
    shape = tuple(int(s) for s in shape)
    if len(buf) == 0:
        return torch.empty(shape, dtype=dtype)
    raw = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
    return raw.view(dtype).reshape(shape)


def load_chunk(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Load a model chunk (state dict) on CPU from a `.safetensors` or `.pth` file. If
    the given file doesn't exist, the same name with the other extension is tried.
    Both formats are memory-mapped when possible.
    """
    path = Path(path)
    if not path.is_file():
        alt = path.with_suffix(".pth" if path.suffix == ".safetensors" else ".safetensors")
        if alt.is_file():
            path = alt
    if path.suffix == ".safetensors":
        return _mmap_safetensors(path)
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except TypeError:  # older torch: no `mmap`/`weights_only` arguments
        return torch.load(path, map_location="cpu")


def _mmap_safetensors(path: Path) -> Dict[str, torch.Tensor]:
    import mmap

    with open(path, "rb") as f:
        # Copy-on-write mapping: pages are read lazily, the file is never modified
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    (n,) = struct.unpack("<Q", mm[:8])
    header = json.loads(mm[8 : 8 + n])
    header.pop("__metadata__", None)
    base = 8 + n
    out = {}
    for name, info in header.items():
        b, e = info["data_offsets"]
        dtype = _ST_DTYPE[info["dtype"]]
        shape = tuple(info["shape"])
        if e == b:
            out[name] = torch.empty(shape, dtype=dtype)
            continue
        raw = torch.frombuffer(mm, dtype=torch.uint8, count=e - b, offset=base + b)
        out[name] = raw.view(dtype).reshape(shape)
    return out
