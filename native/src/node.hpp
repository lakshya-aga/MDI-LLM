// A secondary MDI-LLM node: HTTP control endpoints (same API as the Python
// GPTServer) + the activation ring. Used by the CLI (main.cpp) and the Android app
// (jni.cpp).
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "json.hpp"
#include "model.hpp"
#include "tensor.hpp"

namespace mdi {

struct NodeOptions {
  std::string addr;          // this node's IP, as in the nodes config ("addr")
  int http_port = 8089;      // communication.port
  int port_in = 5090;        // inference.port_in
  int port_out = 5091;       // inference.port_out
  std::string starter_addr;  // communication.starter_addr (informational)
  std::string bind_addr = "0.0.0.0";  // where the HTTP/input sockets listen
  std::string chunk_path;    // optional local .safetensors chunk (memory-mapped)
  int threads = 0;           // 0: all cores
  bool has_weight_dtype = false;
  DType weight_dtype = DType::F32;  // storage dtype when has_weight_dtype
  int index = -1;            // secondary index, if known
  bool verbose = false;

  // Fills addr/ports from a nodes-config JSON (`settings_distr/*.json`)
  static NodeOptions from_nodes_config(const Json& cfg, int secondary_index);
};

class Node {
 public:
  enum class State { Idle, Loading, Connecting, Running, Stopping };

  explicit Node(NodeOptions opts);
  ~Node();

  // Starts the HTTP server thread; throws if the port can't be bound
  void start();
  // Stops everything (server + ring); returns when threads have exited
  void stop();
  bool alive() const { return server_running_.load(); }
  // Number of PUT /stop requests received (end of a run, from the starter)
  uint64_t stop_requests() const { return stop_requests_.load(); }

  State state() const { return state_.load(); }
  static const char* state_name(State s);
  // JSON with state, role, stats (for UIs)
  std::string status_json() const;
  // Log lines with sequence number > `after`; returns the last sequence number
  uint64_t logs_since(uint64_t after, std::string* out) const;

  using LogSink = std::function<void(const std::string&)>;
  void set_log_sink(LogSink sink) { sink_ = std::move(sink); }

 private:
  void log(const std::string& msg);
  void server_loop();
  void handle_connection(int fd);
  // Returns HTTP status; fills body
  int handle_request(const std::string& method, const std::string& path,
                     std::vector<uint8_t>& body, std::string* resp);
  void init_from_body(std::vector<uint8_t>& body);
  void ring_loop();
  void end_run();
  bool has_chunk() const;

  NodeOptions opts_;
  std::atomic<State> state_{State::Idle};
  std::atomic<bool> server_running_{false}, run_flag_{false};
  int listen_fd_ = -1;
  std::thread server_thread_, ring_thread_;

  // Run configuration (from POST /init)
  std::string role_;
  Json prev_node_, next_node_;
  std::unique_ptr<SecondaryModel> model_;
  int64_t n_embd_ = 0;
  std::atomic<int> in_fd_{-1}, out_fd_{-1}, ring_listen_fd_{-1};

  // Stats
  std::atomic<uint64_t> n_forward_{0}, n_tokens_{0}, stop_requests_{0};
  std::atomic<double> last_ms_{0}, total_ms_{0};
  // Snapshot of the loaded model (read by status_json from other threads)
  std::atomic<bool> initialized_{false};
  std::atomic<int> st_layers_{0};
  std::atomic<double> st_weights_mb_{0}, st_kv_mb_{0};

  mutable std::mutex log_mu_;
  std::deque<std::pair<uint64_t, std::string>> logs_;
  uint64_t log_seq_ = 0;
  LogSink sink_;
  std::mutex run_mu_;  // serializes init/stop
};

}  // namespace mdi
