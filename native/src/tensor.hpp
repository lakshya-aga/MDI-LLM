// Weight storage, safetensors parsing, dtype conversion and compute kernels.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "json.hpp"

namespace mdi {

// Codes match `src/sub/wire.py` (DTYPES)
enum class DType : uint8_t { F32 = 0, F16 = 1, BF16 = 2, I64 = 3, I32 = 4 };

size_t dtype_size(DType dt);
const char* dtype_name(DType dt);         // safetensors name ("F32", ...)
DType dtype_from_name(const std::string&);  // throws for unsupported names
bool dtype_from_cli(const std::string& s, DType* out);  // "float32", "f16", "bf16", ...

float f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);
inline float bf16_to_f32(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t f32_to_bf16(float f);  // round-to-nearest-even, as torch

// Converts n elements between float dtypes (F32/F16/BF16)
void convert(const void* src, DType src_dt, void* dst, DType dst_dt, size_t n);

// A block of bytes that owns its memory (heap buffer or read-only mmap)
class Buffer {
 public:
  virtual ~Buffer() = default;
  virtual const uint8_t* data() const = 0;
  virtual size_t size() const = 0;
};
std::shared_ptr<Buffer> heap_buffer(std::vector<uint8_t>&& bytes);
std::shared_ptr<Buffer> mmap_file(const std::string& path);  // throws on failure

// Read-only view of a weight tensor; keeps its backing buffer alive
struct Tensor {
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
  const void* data = nullptr;
  std::shared_ptr<Buffer> owner;

  int64_t numel() const;
  size_t nbytes() const { return static_cast<size_t>(numel()) * dtype_size(dtype); }
  bool empty() const { return data == nullptr; }
};

// Parses a safetensors buffer. Tensors point into `buf` (no copy).
// `metadata` receives the parsed "mdi_init" JSON (null if absent).
std::map<std::string, Tensor> parse_safetensors(const std::shared_ptr<Buffer>& buf, Json* metadata);

// Returns a copy of `t` converted to `dt` (or `t` itself if already `dt`)
Tensor to_dtype(const Tensor& t, DType dt);
// Always returns a copy in its own buffer (so `t`'s backing buffer can be freed)
Tensor own_copy(const Tensor& t, DType dt);

// ---- Thread pool ---------------------------------------------------------------------

// Persistent workers; parallel_for hands out chunks dynamically (good for big.LITTLE
// CPUs, where static splits make the slow cores the bottleneck).
class ThreadPool {
 public:
  explicit ThreadPool(int n_threads);
  ~ThreadPool();
  int size() const { return n_threads_; }
  // Calls fn(begin, end) over [0, n) in chunks of `grain`; blocks until done
  void parallel_for(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& fn);

 private:
  void worker_loop();
  void run_chunks();

  int n_threads_;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_start_, cv_done_;
  uint64_t generation_ = 0;
  int active_ = 0;
  bool stop_ = false;
  const std::function<void(int64_t, int64_t)>* fn_ = nullptr;
  int64_t n_ = 0, grain_ = 1;
  std::atomic<int64_t> next_{0};
};

// ---- Kernels ---------------------------------------------------------------------------

// y[t, o] = sum_i x[t, i] * W[o, i] (+ b[o]);  x: [T, in] f32, W: [out, in] any float dtype
void linear(ThreadPool& pool, const float* x, int64_t T, const Tensor& W, const Tensor* b, float* y);

}  // namespace mdi
