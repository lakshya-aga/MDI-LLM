// Small POSIX socket helpers and the activation frame codec (see src/sub/wire.py).
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "tensor.hpp"

namespace mdi {

// Resolves a host name or dotted address to a dotted IPv4 string ("" on failure)
std::string resolve_ipv4(const std::string& host);

// Listening TCP socket bound to host:port (SO_REUSEADDR). Returns fd or -1.
int listen_tcp(const std::string& host, int port, int backlog);

// Waits up to `timeout_ms` for a connection; returns client fd, or -1 on timeout.
// `peer_ip` receives the client's dotted IPv4 address.
int accept_timeout(int listen_fd, int timeout_ms, std::string* peer_ip);

// Connects to host:port, first trying to bind the local end to bind_host:bind_port
// (as the Python nodes do; skipped silently if binding fails). Returns fd or -1.
int connect_tcp(const std::string& host, int port, const std::string& bind_host, int bind_port);

void set_nodelay(int fd);
void set_recv_timeout(int fd, int seconds);
void close_fd(int& fd);  // closes and sets to -1

// Blocking full read/write; false on EOF/error
bool read_all(int fd, void* buf, size_t n);
bool write_all(int fd, const void* buf, size_t n);

// ---- Activation frames ------------------------------------------------------------------

constexpr uint8_t kFrameVersion = 1;
constexpr uint8_t kFlagStop = 1;
constexpr size_t kFrameHeaderSize = 20;

struct Frame {
  uint8_t flags = 0;
  DType dtype = DType::F32;
  uint32_t sample_index = 0;
  std::vector<uint32_t> shape;
  std::vector<uint8_t> payload;
  bool stop() const { return flags & kFlagStop; }
};

// Reads one frame; false on EOF/error. Throws on a malformed header.
bool read_frame(int fd, Frame* f);
bool write_frame(int fd, const Frame& f);

}  // namespace mdi
