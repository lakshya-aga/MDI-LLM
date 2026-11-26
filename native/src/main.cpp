// Command-line secondary node, a drop-in replacement for `src/secondary.py`:
//
//   mdi_node --nodes-config settings_distr/phones.json 0 [--chunk model_secondary0.safetensors]
//   mdi_node --addr 192.168.1.21 --port 8089 --port-in 5090 --port-out 5091
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#include "node.hpp"

static volatile std::sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

static void usage() {
  std::fprintf(stderr,
               "usage: mdi_node (--nodes-config FILE INDEX | --addr IP [--port P --port-in P --port-out P])\n"
               "                [--chunk FILE.safetensors] [--threads N] [--weights-dtype float32|float16|bfloat16]\n"
               "                [--bind ADDR] [--once] [-v]\n"
               "\n"
               "  --nodes-config  nodes JSON (same file as the Python nodes) and this node's secondary index\n"
               "  --chunk         local weights (from scripts/export_chunks.py); otherwise the starter sends them\n"
               "  --weights-dtype storage dtype for the weights (default: as received)\n"
               "  --bind          address the sockets listen on (default 0.0.0.0)\n"
               "  --once          exit after the first run instead of waiting for the next one\n");
}

int main(int argc, char** argv) {
  mdi::NodeOptions opts;
  bool have_cfg = false, once = false;
  std::string chunk, bind = "0.0.0.0", wdt;
  int threads = 0;
  bool verbose = false;
  std::string addr;
  int port = 8089, port_in = 5090, port_out = 5091;

  try {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
        return argv[++i];
      };
      if (a == "--nodes-config") {
        std::string path = next();
        int idx = std::stoi(next());
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot read " + path);
        std::stringstream ss;
        ss << f.rdbuf();
        opts = mdi::NodeOptions::from_nodes_config(mdi::Json::parse(ss.str()), idx);
        have_cfg = true;
      } else if (a == "--addr") addr = next();
      else if (a == "--port") port = std::stoi(next());
      else if (a == "--port-in") port_in = std::stoi(next());
      else if (a == "--port-out") port_out = std::stoi(next());
      else if (a == "--chunk") chunk = next();
      else if (a == "--threads") threads = std::stoi(next());
      else if (a == "--weights-dtype") wdt = next();
      else if (a == "--bind") bind = next();
      else if (a == "--once") once = true;
      else if (a == "-v" || a == "--verb") verbose = true;
      else if (a == "-h" || a == "--help") { usage(); return 0; }
      else throw std::runtime_error("unknown argument " + a);
    }
    if (!have_cfg) {
      if (addr.empty()) { usage(); return 2; }
      opts.addr = addr;
      opts.http_port = port;
      opts.port_in = port_in;
      opts.port_out = port_out;
    }
    opts.chunk_path = chunk;
    opts.bind_addr = bind;
    opts.threads = threads;
    opts.verbose = verbose;
    if (!wdt.empty()) {
      if (!mdi::dtype_from_cli(wdt, &opts.weight_dtype)) throw std::runtime_error("bad --weights-dtype " + wdt);
      opts.has_weight_dtype = true;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    usage();
    return 2;
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::signal(SIGPIPE, SIG_IGN);

  mdi::Node node(opts);
  try {
    node.start();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  // --once: exit after a run, once the starter has sent its stop request (it retries
  // until a node answers) or after 15 s of idling
  bool was_busy = false;
  int idle_ticks = 0;
  while (!g_stop) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    bool busy = node.state() != mdi::Node::State::Idle;
    was_busy = was_busy || busy;
    idle_ticks = (was_busy && !busy) ? idle_ticks + 1 : 0;
    if (once && was_busy && !busy && (node.stop_requests() > 0 || idle_ticks > 75)) break;
  }
  node.stop();
  return 0;
}
