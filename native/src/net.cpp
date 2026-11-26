#include "net.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace mdi {

std::string resolve_ipv4(const std::string& host) {
  struct addrinfo hints;
  std::memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo* res = nullptr;
  if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) return "";
  char buf[INET_ADDRSTRLEN] = {0};
  inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(res->ai_addr)->sin_addr, buf, sizeof buf);
  freeaddrinfo(res);
  return buf;
}

static bool make_addr(const std::string& host, int port, sockaddr_in* sa) {
  std::memset(sa, 0, sizeof *sa);
  sa->sin_family = AF_INET;
  sa->sin_port = htons(static_cast<uint16_t>(port));
  if (host.empty() || host == "0.0.0.0") {
    sa->sin_addr.s_addr = htonl(INADDR_ANY);
    return true;
  }
  std::string ip = resolve_ipv4(host);
  return !ip.empty() && inet_pton(AF_INET, ip.c_str(), &sa->sin_addr) == 1;
}

static int new_socket() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
#ifdef SO_NOSIGPIPE
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
  return fd;
}

int listen_tcp(const std::string& host, int port, int backlog) {
  sockaddr_in sa;
  if (!make_addr(host, port, &sa)) return -1;
  int fd = new_socket();
  if (fd < 0) return -1;
  if (bind(fd, reinterpret_cast<sockaddr*>(&sa), sizeof sa) != 0 || listen(fd, backlog) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

int accept_timeout(int listen_fd, int timeout_ms, std::string* peer_ip) {
  struct pollfd p = {listen_fd, POLLIN, 0};
  int r = poll(&p, 1, timeout_ms);
  if (r <= 0 || !(p.revents & POLLIN)) return -1;
  sockaddr_in sa;
  socklen_t len = sizeof sa;
  int fd = accept(listen_fd, reinterpret_cast<sockaddr*>(&sa), &len);
  if (fd < 0) return -1;
#ifdef SO_NOSIGPIPE
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
  if (peer_ip) {
    char buf[INET_ADDRSTRLEN] = {0};
    inet_ntop(AF_INET, &sa.sin_addr, buf, sizeof buf);
    *peer_ip = buf;
  }
  return fd;
}

int connect_tcp(const std::string& host, int port, const std::string& bind_host, int bind_port) {
  sockaddr_in dst;
  if (!make_addr(host, port, &dst)) return -1;
  // First with the configured local port, then with an ephemeral one: right after a
  // previous run the old connection on that port may still be in TIME_WAIT, which
  // makes connect() fail with the same 4-tuple. Receivers only check the peer IP.
  for (int attempt = 0; attempt < 2; ++attempt) {
    int fd = new_socket();
    if (fd < 0) return -1;
    if (attempt == 0) {
      if (bind_port <= 0) { close(fd); continue; }
      sockaddr_in src;
      if (!make_addr(bind_host, bind_port, &src) ||
          bind(fd, reinterpret_cast<sockaddr*>(&src), sizeof src) != 0) {
        close(fd);
        continue;
      }
    }
    if (connect(fd, reinterpret_cast<sockaddr*>(&dst), sizeof dst) == 0) return fd;
    int err = errno;
    close(fd);
    if (err == ECONNREFUSED || err == ETIMEDOUT || err == EHOSTUNREACH || err == ENETUNREACH) return -1;
  }
  return -1;
}

void set_nodelay(int fd) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
}

void set_recv_timeout(int fd, int seconds) {
  struct timeval tv;
  tv.tv_sec = seconds;
  tv.tv_usec = 0;
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
}

void close_fd(int& fd) {
  if (fd >= 0) {
    shutdown(fd, SHUT_RDWR);
    close(fd);
    fd = -1;
  }
}

bool read_all(int fd, void* buf, size_t n) {
  uint8_t* p = static_cast<uint8_t*>(buf);
  while (n > 0) {
    ssize_t r = recv(fd, p, n, 0);
    if (r == 0) return false;
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

bool write_all(int fd, const void* buf, size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(buf);
  while (n > 0) {
    ssize_t r = send(fd, p, n, MSG_NOSIGNAL);
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

// Little-endian helpers (ARM/x86 hosts are little-endian; memcpy keeps it portable
// with respect to alignment)
template <typename T>
static T get_le(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof v);
  return v;
}
template <typename T>
static void put_le(uint8_t* p, T v) {
  std::memcpy(p, &v, sizeof v);
}

bool read_frame(int fd, Frame* f) {
  uint8_t h[kFrameHeaderSize];
  if (!read_all(fd, h, sizeof h)) return false;
  if (std::memcmp(h, "MDIA", 4) != 0)
    throw std::runtime_error("Bad frame magic (is the peer running an old, pickle-based version?)");
  if (h[4] != kFrameVersion) throw std::runtime_error("Unsupported frame version");
  f->flags = h[5];
  if (h[6] > 4) throw std::runtime_error("Unknown dtype code in frame");
  f->dtype = static_cast<DType>(h[6]);
  uint8_t ndim = h[7];
  f->sample_index = get_le<uint32_t>(h + 8);
  uint64_t size = get_le<uint64_t>(h + 12);
  f->shape.resize(ndim);
  if (ndim) {
    std::vector<uint8_t> s(4u * ndim);
    if (!read_all(fd, s.data(), s.size())) return false;
    for (uint8_t i = 0; i < ndim; ++i) f->shape[i] = get_le<uint32_t>(s.data() + 4 * i);
  }
  if (size > (1ull << 32)) throw std::runtime_error("Frame payload too large");
  f->payload.resize(static_cast<size_t>(size));
  return size == 0 || read_all(fd, f->payload.data(), f->payload.size());
}

bool write_frame(int fd, const Frame& f) {
  std::vector<uint8_t> h(kFrameHeaderSize + 4 * f.shape.size());
  std::memcpy(h.data(), "MDIA", 4);
  h[4] = kFrameVersion;
  h[5] = f.flags;
  h[6] = static_cast<uint8_t>(f.dtype);
  h[7] = static_cast<uint8_t>(f.shape.size());
  put_le<uint32_t>(h.data() + 8, f.sample_index);
  put_le<uint64_t>(h.data() + 12, static_cast<uint64_t>(f.payload.size()));
  for (size_t i = 0; i < f.shape.size(); ++i) put_le<uint32_t>(h.data() + 20 + 4 * i, f.shape[i]);
  return write_all(fd, h.data(), h.size()) &&
         (f.payload.empty() || write_all(fd, f.payload.data(), f.payload.size()));
}

}  // namespace mdi
