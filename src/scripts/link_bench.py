#!/usr/bin/env python3
"""
Round-trip time and bandwidth of the link between two nodes. Standard library only,
so it runs in Termux without PyTorch.

On the far device (for example a phone):
    python link_bench.py serve --port 5999
On the starter (the laptop):
    python link_bench.py ping 192.168.1.21 --port 5999

`--bytes` should match one decode-step activation: n_embd x bytes per element
(TinyLlama float32: 2048 x 4 = 8192; Llama 3.1 8B float16: 4096 x 2 = 8192).
The script also sends a few large messages to estimate link bandwidth, which matters
for prefill and for shipping weights, not for decode.
"""

import argparse
import socket
import statistics
import struct
import time


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        buf += chunk
    return bytes(buf)


def serve(args):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print(f"echo server listening on port {args.port} (Ctrl-C to stop)")
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"connection from {addr[0]}")
        try:
            while True:
                (n,) = struct.unpack("!I", recv_exact(conn, 4))
                payload = recv_exact(conn, n)
                # Echo small messages; acknowledge large ones with 1 byte.
                reply = payload if n <= 1 << 16 else b"\x00"
                conn.sendall(struct.pack("!I", len(reply)) + reply)
        except ConnectionError:
            conn.close()


def round_trip(sock: socket.socket, payload: bytes) -> float:
    t0 = time.perf_counter()
    sock.sendall(struct.pack("!I", len(payload)) + payload)
    (n,) = struct.unpack("!I", recv_exact(sock, 4))
    recv_exact(sock, n)
    return time.perf_counter() - t0


def ping(args):
    sock = socket.create_connection((args.host, args.port), timeout=10)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    payload = bytes(args.bytes)
    for _ in range(10):  # warm-up: wake the radio, open the TCP window
        round_trip(sock, payload)
    rtts = []
    for _ in range(args.n):
        rtts.append(round_trip(sock, payload) * 1e3)
        time.sleep(args.gap_ms / 1e3)
    rtts.sort()
    med = statistics.median(rtts)
    p90 = rtts[int(0.9 * (len(rtts) - 1))]
    print(f"{args.host}: {args.n} round trips of {args.bytes} B")
    print(f"RTT ms: min {rtts[0]:.2f}  median {med:.2f}  p90 {p90:.2f}  max {rtts[-1]:.2f}")

    big = bytes(args.bulk_mb << 20)
    times = [round_trip(sock, big) for _ in range(3)]
    mbps = args.bulk_mb * 8 * 1.048576 / min(times)
    print(f"bulk: {args.bulk_mb} MiB in {min(times) * 1e3:.0f} ms -> {mbps:.0f} Mbit/s")
    print(f'{{"rtt_ms": {med:.2f}, "rtt_p90_ms": {p90:.2f}, "link_mbps": {mbps:.0f}}}')
    sock.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the echo server on the far device")
    s.add_argument("--port", type=int, default=5999)
    c = sub.add_parser("ping", help="measure the link to a device running `serve`")
    c.add_argument("host")
    c.add_argument("--port", type=int, default=5999)
    c.add_argument("--bytes", type=int, default=8192, help="message size (default 8192)")
    c.add_argument("-n", type=int, default=200, help="round trips (default 200)")
    c.add_argument("--gap-ms", type=float, default=20.0,
                   help="pause between round trips, like the gap between tokens (default 20)")
    c.add_argument("--bulk-mb", type=int, default=16, help="large-message size in MiB")
    a = p.parse_args()
    serve(a) if a.cmd == "serve" else ping(a)
