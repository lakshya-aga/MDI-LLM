#include "node.hpp"

#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <sstream>
#include <stdexcept>

#include "net.hpp"

namespace mdi {

NodeOptions NodeOptions::from_nodes_config(const Json& cfg, int secondary_index) {
  const Json& nodes = cfg.has("nodes") ? cfg["nodes"] : cfg;
  const Json& sec = nodes["secondary"];
  if (secondary_index < 0 || static_cast<size_t>(secondary_index) >= sec.size())
    throw std::runtime_error("Secondary index out of range in the nodes config");
  const Json& me = sec[static_cast<size_t>(secondary_index)];
  NodeOptions o;
  o.addr = me["addr"].as_string();
  o.http_port = static_cast<int>(me["communication"]["port"].as_int());
  o.starter_addr = me["communication"].str_or("starter_addr", "");
  o.port_in = static_cast<int>(me["inference"]["port_in"].as_int());
  o.port_out = static_cast<int>(me["inference"]["port_out"].as_int());
  o.index = secondary_index;
  return o;
}

Node::Node(NodeOptions opts) : opts_(std::move(opts)) {}

Node::~Node() { stop(); }

const char* Node::state_name(State s) {
  switch (s) {
    case State::Idle: return "idle";
    case State::Loading: return "loading";
    case State::Connecting: return "connecting";
    case State::Running: return "running";
    case State::Stopping: return "stopping";
  }
  return "?";
}

void Node::log(const std::string& msg) {
  {
    std::lock_guard<std::mutex> lk(log_mu_);
    logs_.emplace_back(++log_seq_, msg);
    while (logs_.size() > 500) logs_.pop_front();
  }
  if (sink_) sink_(msg);
  else std::fprintf(stderr, "%s\n", msg.c_str());
}

uint64_t Node::logs_since(uint64_t after, std::string* out) const {
  std::lock_guard<std::mutex> lk(log_mu_);
  for (auto& l : logs_)
    if (l.first > after) { *out += l.second; *out += '\n'; }
  return log_seq_;
}

bool Node::has_chunk() const {
  if (opts_.chunk_path.empty()) return false;
  struct stat st;
  return stat(opts_.chunk_path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string Node::status_json() const {
  Json j = Json::object();
  j.set("state", state_name(state_.load()));
  j.set("role", role_.empty() ? (opts_.index >= 0 ? "secondary:" + std::to_string(opts_.index) : "secondary")
                              : role_);
  j.set("addr", opts_.addr);
  j.set("http_port", opts_.http_port);
  j.set("has_chunk", has_chunk());
  j.set("initialized", initialized_.load());
  uint64_t n = n_forward_.load();
  j.set("tokens", static_cast<int64_t>(n_tokens_.load()));
  j.set("forwards", static_cast<int64_t>(n));
  j.set("last_ms", last_ms_.load());
  j.set("avg_ms", n ? total_ms_.load() / static_cast<double>(n) : 0.0);
  if (initialized_) {
    j.set("n_layers", st_layers_.load());
    j.set("weights_mb", st_weights_mb_.load());
    j.set("kv_cache_mb_per_sample", st_kv_mb_.load());
  }
  return j.dump();
}

void Node::start() {
  listen_fd_ = listen_tcp(opts_.bind_addr, opts_.http_port, 8);
  if (listen_fd_ < 0)
    throw std::runtime_error("Cannot listen on " + opts_.bind_addr + ":" + std::to_string(opts_.http_port));
  server_running_ = true;
  server_thread_ = std::thread([this] { server_loop(); });
  log("[INFO] Node listening on port " + std::to_string(opts_.http_port) + " (addr in config: " +
      opts_.addr + ")" + (has_chunk() ? ", local chunk: " + opts_.chunk_path : ""));
}

void Node::stop() {
  run_flag_ = false;
  for (auto* a : {&in_fd_, &out_fd_, &ring_listen_fd_}) {
    int fd = a->load();
    if (fd >= 0) shutdown(fd, SHUT_RDWR);
  }
  if (ring_thread_.joinable()) ring_thread_.join();
  if (server_running_.exchange(false) && server_thread_.joinable()) server_thread_.join();
  close_fd(listen_fd_);
}

// ---- HTTP ------------------------------------------------------------------------------

void Node::server_loop() {
  while (server_running_) {
    std::string peer;
    int fd = accept_timeout(listen_fd_, 500, &peer);
    if (fd < 0) continue;
    try {
      handle_connection(fd);
    } catch (const std::exception& e) {
      log(std::string("[ERROR] HTTP: ") + e.what());
    }
    close_fd(fd);
  }
}

static void send_response(int fd, int code, const std::string& body) {
  const char* reason = code == 200 ? "OK" : code == 403 ? "Forbidden" : code == 404 ? "Not Found"
                     : code == 411 ? "Length Required" : code == 400 ? "Bad Request" : "Internal Server Error";
  std::string h = "HTTP/1.1 " + std::to_string(code) + " " + reason +
                  "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) +
                  "\r\nConnection: close\r\n\r\n";
  write_all(fd, h.data(), h.size());
  write_all(fd, body.data(), body.size());
}

static std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

void Node::handle_connection(int fd) {
  set_recv_timeout(fd, 300);
  std::string head;
  std::vector<uint8_t> body;
  char buf[65536];
  size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    ssize_t r = recv(fd, buf, sizeof buf, 0);
    if (r <= 0) return;
    head.append(buf, static_cast<size_t>(r));
    header_end = head.find("\r\n\r\n");
    if (header_end == std::string::npos && head.size() > (1u << 20)) return send_response(fd, 400, "{}");
  }
  std::string rest = head.substr(header_end + 4);
  head.resize(header_end);

  std::istringstream hs(head);
  std::string method, target, line;
  hs >> method >> target;
  std::getline(hs, line);
  long long content_length = 0;
  bool chunked = false;
  while (std::getline(hs, line)) {
    auto c = line.find(':');
    if (c == std::string::npos) continue;
    std::string k = lower(line.substr(0, c)), v = line.substr(c + 1);
    while (!v.empty() && (v.front() == ' ')) v.erase(0, 1);
    while (!v.empty() && (v.back() == '\r' || v.back() == ' ')) v.pop_back();
    if (k == "content-length") content_length = std::stoll(v);
    if (k == "transfer-encoding" && lower(v).find("chunked") != std::string::npos) chunked = true;
  }
  if (chunked) return send_response(fd, 411, "{\"error\": \"chunked bodies are not supported\"}");
  if (content_length > 0) {
    body.resize(static_cast<size_t>(content_length));
    size_t have = std::min(rest.size(), body.size());
    std::memcpy(body.data(), rest.data(), have);
    if (have < body.size() && !read_all(fd, body.data() + have, body.size() - have)) {
      log("[ERROR] HTTP: connection closed while receiving the body");
      return;
    }
  }
  std::string path = target.substr(0, target.find('?'));
  std::string resp;
  int code = handle_request(method, path, body, &resp);
  send_response(fd, code, resp);
}

int Node::handle_request(const std::string& method, const std::string& path, std::vector<uint8_t>& body,
                         std::string* resp) {
  if (method == "GET" && (path == "/" || path.empty())) {
    Json j = Json::object();
    j.set("addr", opts_.addr);
    Json comm = Json::object();
    comm.set("port", opts_.http_port);
    comm.set("starter_addr", opts_.starter_addr);
    j.set("communication", comm);
    Json inf = Json::object();
    inf.set("port_in", opts_.port_in);
    inf.set("port_out", opts_.port_out);
    j.set("inference", inf);
    *resp = j.dump();
    return 200;
  }
  if (method == "GET" && path == "/status") {
    Json j = Json::parse(status_json());
    j.set("native", true);
    *resp = j.dump();
    return 200;
  }
  if (method == "POST" && path == "/init") {
    std::lock_guard<std::mutex> lk(run_mu_);
    if (state_ != State::Idle) {
      *resp = "{\"error\": \"node is busy (" + std::string(state_name(state_)) + ")\"}";
      return 403;
    }
    if (ring_thread_.joinable()) ring_thread_.join();  // previous run (already finished)
    state_ = State::Loading;
    try {
      init_from_body(body);
    } catch (const std::exception& e) {
      initialized_ = false;
      model_.reset();
      state_ = State::Idle;
      log(std::string("[ERROR] Initialization failed: ") + e.what());
      Json j = Json::object();
      j.set("error", e.what());
      *resp = j.dump();
      return 500;
    }
    run_flag_ = true;
    state_ = State::Connecting;
    ring_thread_ = std::thread([this] { ring_loop(); });
    *resp = "{}";
    return 200;
  }
  if (method == "PUT" && path == "/stop") {
    log("[INFO] Stop requested by the starter");
    stop_requests_++;
    run_flag_ = false;
    if (state_ != State::Idle) state_ = State::Stopping;
    for (auto* a : {&in_fd_, &out_fd_, &ring_listen_fd_}) {
      int fd = a->load();
      if (fd >= 0) shutdown(fd, SHUT_RDWR);
    }
    *resp = "{}";
    return 200;
  }
  *resp = "{\"error\": \"not found\"}";
  return 404;
}

void Node::init_from_body(std::vector<uint8_t>& body) {
  auto t0 = std::chrono::steady_clock::now();
  if (body.empty()) throw std::runtime_error("empty init body");
  auto buf = heap_buffer(std::move(body));
  Json meta;
  auto tensors = parse_safetensors(buf, &meta);
  if (meta.is_null()) throw std::runtime_error("init body has no mdi_init metadata");
  std::shared_ptr<Buffer> chunk;
  if (tensors.empty()) {
    if (!has_chunk()) throw std::runtime_error("the starter sent no weights and there is no local chunk");
    log("[INFO] Loading weights from " + opts_.chunk_path + " (memory-mapped)");
    chunk = mmap_file(opts_.chunk_path);
    tensors = parse_safetensors(chunk, nullptr);
  } else {
    log("[INFO] Received weights from the starter (" + std::to_string(buf->size() >> 20) + " MB)");
  }
  buf.reset();  // tensors keep the body alive only as long as they reference it

  role_ = meta.str_or("role", role_);
  prev_node_ = meta["prev_node"];
  next_node_ = meta["next_node"];
  ModelConfig cfg = ModelConfig::from_json(meta["model_config"]);
  int n_layers = static_cast<int>(meta["n_local_layers"].as_int());
  int64_t max_seq = static_cast<int64_t>(meta.num_or("max_seq_length", 0));
  int threads = opts_.threads > 0 ? opts_.threads : static_cast<int>(std::thread::hardware_concurrency());
  model_ = std::make_unique<SecondaryModel>(cfg, n_layers, max_seq, tensors,
                                            opts_.has_weight_dtype ? &opts_.weight_dtype : nullptr, threads);
  tensors.clear();
  st_layers_ = model_->n_layers();
  st_weights_mb_ = static_cast<double>(model_->weight_bytes()) / 1048576.0;
  st_kv_mb_ = static_cast<double>(model_->cache_bytes_per_sample()) / 1048576.0;
  initialized_ = true;
  n_embd_ = cfg.n_embd;
  n_forward_ = 0;
  n_tokens_ = 0;
  total_ms_ = 0;
  last_ms_ = 0;
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  char msg[256];
  std::snprintf(msg, sizeof msg,
                "[INFO] %s: %d layers, weights %.0f MB, KV cache %.1f MB/sample, context %lld, %d threads "
                "(%.1fs)",
                role_.c_str(), n_layers, model_->weight_bytes() / 1048576.0,
                model_->cache_bytes_per_sample() / 1048576.0, static_cast<long long>(model_->max_seq_length()),
                threads, secs);
  log(msg);
}

// ---- Activation ring ------------------------------------------------------------------

void Node::ring_loop() {
  int lfd = -1, in = -1, out = -1;
  try {
    const std::string prev_addr = prev_node_["addr"].as_string();
    const std::string next_addr = next_node_["addr"].as_string();
    const int next_port = static_cast<int>(next_node_["inference"]["port_in"].as_int());
    const std::string prev_ip = resolve_ipv4(prev_addr);

    // 1. Input socket: wait for the previous node to connect
    for (int tries = 0; run_flag_ && lfd < 0 && tries < 30; ++tries) {
      lfd = listen_tcp(opts_.bind_addr, opts_.port_in, 1);
      if (lfd < 0) std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (lfd < 0) throw std::runtime_error("cannot listen on port " + std::to_string(opts_.port_in));
    ring_listen_fd_ = lfd;
    log("[INFO] Waiting for the previous node (" + prev_addr + ")");
    while (run_flag_ && in < 0) {
      std::string peer;
      int fd = accept_timeout(lfd, 500, &peer);
      if (fd < 0) continue;
      if (!prev_ip.empty() && peer != prev_ip) {
        log("[WARN] Rejected connection from " + peer + " (expected " + prev_ip + ")");
        close_fd(fd);
        continue;
      }
      in = fd;
    }
    ring_listen_fd_ = -1;
    close_fd(lfd);
    if (in < 0) throw std::runtime_error("stopped before the previous node connected");
    in_fd_ = in;
    set_nodelay(in);

    // 2. Output socket: connect to the next node
    log("[INFO] Connecting to the next node (" + next_addr + ":" + std::to_string(next_port) + ")");
    for (int tries = 0; run_flag_ && out < 0 && tries < 60; ++tries) {
      out = connect_tcp(next_addr, next_port, opts_.addr, opts_.port_out);
      if (out < 0) std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (out < 0) throw std::runtime_error("cannot connect to the next node");
    out_fd_ = out;
    set_nodelay(out);
    state_ = State::Running;
    log("[INFO] Ring connected, processing samples");

    // 3. Process frames
    Frame f;
    std::vector<float> x;
    while (run_flag_) {
      if (!read_frame(in, &f)) {
        if (run_flag_) log("[INFO] Previous node closed the connection");
        break;
      }
      if (f.stop()) {
        model_->drop_sample(static_cast<int>(f.sample_index));
        if (opts_.verbose) log("[DEBUG] Finished sample " + std::to_string(f.sample_index));
        if (!write_frame(out, f)) break;
        continue;
      }
      if (f.shape.size() != 3 || f.shape[0] != 1 || f.shape[2] != n_embd_)
        throw std::runtime_error("unexpected activation shape");
      if (f.dtype != DType::F32 && f.dtype != DType::F16 && f.dtype != DType::BF16)
        throw std::runtime_error("activations must be a float dtype");
      const int64_t T = f.shape[1];
      const size_t n = static_cast<size_t>(T * n_embd_);
      if (f.payload.size() != n * dtype_size(f.dtype)) throw std::runtime_error("bad payload size");
      x.resize(n);
      convert(f.payload.data(), f.dtype, x.data(), DType::F32, n);

      auto t0 = std::chrono::steady_clock::now();
      model_->forward(static_cast<int>(f.sample_index), x.data(), T);
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

      convert(x.data(), DType::F32, f.payload.data(), f.dtype, n);  // reply in the received dtype
      if (!write_frame(out, f)) break;
      last_ms_ = ms;
      total_ms_ = total_ms_.load() + ms;
      n_forward_++;
      n_tokens_ += static_cast<uint64_t>(T);
    }
  } catch (const std::exception& e) {
    log(std::string("[ERROR] ") + e.what());
  }
  ring_listen_fd_ = -1;
  in_fd_ = -1;
  out_fd_ = -1;
  close_fd(lfd);
  close_fd(in);
  close_fd(out);
  end_run();
}

void Node::end_run() {
  run_flag_ = false;
  initialized_ = false;
  model_.reset();
  state_ = State::Idle;
  log("[INFO] Run finished; waiting for the starter");
}

}  // namespace mdi
