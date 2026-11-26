#include "tensor.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <stdexcept>

// NEON kernels use AArch64-only intrinsics (vaddvq, vshll_high, f16 converts); other
// targets (armeabi-v7a, x86_64 emulators) use the portable loops below.
#if defined(__aarch64__)
#include <arm_neon.h>
#define MDI_NEON 1
#endif

namespace mdi {

size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::F32: return 4;
    case DType::F16: return 2;
    case DType::BF16: return 2;
    case DType::I64: return 8;
    case DType::I32: return 4;
  }
  return 0;
}

const char* dtype_name(DType dt) {
  switch (dt) {
    case DType::F32: return "F32";
    case DType::F16: return "F16";
    case DType::BF16: return "BF16";
    case DType::I64: return "I64";
    case DType::I32: return "I32";
  }
  return "?";
}

DType dtype_from_name(const std::string& s) {
  if (s == "F32") return DType::F32;
  if (s == "F16") return DType::F16;
  if (s == "BF16") return DType::BF16;
  if (s == "I64") return DType::I64;
  if (s == "I32") return DType::I32;
  throw std::runtime_error("Unsupported safetensors dtype " + s);
}

bool dtype_from_cli(const std::string& s, DType* out) {
  if (s == "float32" || s == "f32" || s == "fp32") { *out = DType::F32; return true; }
  if (s == "float16" || s == "f16" || s == "fp16") { *out = DType::F16; return true; }
  if (s == "bfloat16" || s == "bf16") { *out = DType::BF16; return true; }
  return false;
}

// ---- float conversions -------------------------------------------------------------

float f16_to_f32(uint16_t h) {
#if defined(__aarch64__)
  __fp16 v;
  std::memcpy(&v, &h, 2);
  return static_cast<float>(v);
#else
  uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FF, u;
  if (exp == 0) {
    if (mant == 0) {
      u = sign;
    } else {  // subnormal
      exp = 127 - 15 + 1;
      while (!(mant & 0x400)) { mant <<= 1; --exp; }
      mant &= 0x3FF;
      u = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    u = sign | 0x7F800000u | (mant << 13);
  } else {
    u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &u, 4);
  return f;
#endif
}

uint16_t f32_to_f16(float f) {
#if defined(__aarch64__)
  __fp16 v = static_cast<__fp16>(f);
  uint16_t h;
  std::memcpy(&h, &v, 2);
  return h;
#else
  uint32_t u;
  std::memcpy(&u, &f, 4);
  uint32_t sign = (u >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = u & 0x7FFFFF;
  if (((u >> 23) & 0xFF) == 0xFF) return static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x200 : 0));
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000;
    uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t half = mant >> shift, rem = mant & ((1u << shift) - 1), mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1))) ++half;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half;
  return static_cast<uint16_t>(half);
#endif
}

uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return static_cast<uint16_t>((u >> 16) | 0x40);  // NaN
  u += 0x7FFF + ((u >> 16) & 1);
  return static_cast<uint16_t>(u >> 16);
}

static inline float load_f(const void* p, DType dt, size_t i) {
  switch (dt) {
    case DType::F32: return static_cast<const float*>(p)[i];
    case DType::F16: return f16_to_f32(static_cast<const uint16_t*>(p)[i]);
    case DType::BF16: return bf16_to_f32(static_cast<const uint16_t*>(p)[i]);
    default: throw std::runtime_error("not a float dtype");
  }
}

void convert(const void* src, DType sdt, void* dst, DType ddt, size_t n) {
  if (sdt == ddt) { std::memcpy(dst, src, n * dtype_size(sdt)); return; }
  for (size_t i = 0; i < n; ++i) {
    float v = load_f(src, sdt, i);
    switch (ddt) {
      case DType::F32: static_cast<float*>(dst)[i] = v; break;
      case DType::F16: static_cast<uint16_t*>(dst)[i] = f32_to_f16(v); break;
      case DType::BF16: static_cast<uint16_t*>(dst)[i] = f32_to_bf16(v); break;
      default: throw std::runtime_error("not a float dtype");
    }
  }
}

// ---- buffers --------------------------------------------------------------------------

namespace {
class HeapBuffer : public Buffer {
 public:
  explicit HeapBuffer(std::vector<uint8_t>&& b) : b_(std::move(b)) {}
  const uint8_t* data() const override { return b_.data(); }
  size_t size() const override { return b_.size(); }

 private:
  std::vector<uint8_t> b_;
};

class MmapBuffer : public Buffer {
 public:
  MmapBuffer(void* p, size_t n) : p_(p), n_(n) {}
  ~MmapBuffer() override { if (p_) munmap(p_, n_); }
  const uint8_t* data() const override { return static_cast<const uint8_t*>(p_); }
  size_t size() const override { return n_; }

 private:
  void* p_;
  size_t n_;
};
}  // namespace

std::shared_ptr<Buffer> heap_buffer(std::vector<uint8_t>&& bytes) {
  return std::make_shared<HeapBuffer>(std::move(bytes));
}

std::shared_ptr<Buffer> mmap_file(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("Cannot open " + path);
  struct stat st;
  if (fstat(fd, &st) != 0) { close(fd); throw std::runtime_error("Cannot stat " + path); }
  size_t n = static_cast<size_t>(st.st_size);
  void* p = mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (p == MAP_FAILED) throw std::runtime_error("Cannot mmap " + path);
  return std::make_shared<MmapBuffer>(p, n);
}

int64_t Tensor::numel() const {
  int64_t n = 1;
  for (auto s : shape) n *= s;
  return n;
}

std::map<std::string, Tensor> parse_safetensors(const std::shared_ptr<Buffer>& buf, Json* metadata) {
  const uint8_t* d = buf->data();
  size_t total = buf->size();
  if (total < 8) throw std::runtime_error("safetensors: buffer too small");
  uint64_t n = 0;
  std::memcpy(&n, d, 8);  // little-endian hosts only (ARM/x86)
  if (8 + n > total) throw std::runtime_error("safetensors: bad header length");
  Json header = Json::parse(std::string(reinterpret_cast<const char*>(d + 8), n));
  const uint8_t* base = d + 8 + n;
  size_t data_len = total - 8 - n;

  std::map<std::string, Tensor> out;
  if (metadata) *metadata = Json();
  for (auto& kv : header.members()) {
    if (kv.first == "__metadata__") {
      if (metadata && kv.second.has("mdi_init"))
        *metadata = Json::parse(kv.second["mdi_init"].as_string());
      continue;
    }
    const Json& info = kv.second;
    Tensor t;
    t.dtype = dtype_from_name(info["dtype"].as_string());
    for (auto& s : info["shape"].items()) t.shape.push_back(s.as_int());
    uint64_t b = static_cast<uint64_t>(info["data_offsets"][0].as_int());
    uint64_t e = static_cast<uint64_t>(info["data_offsets"][1].as_int());
    if (e < b || e > data_len || e - b != t.nbytes())
      throw std::runtime_error("safetensors: bad offsets for " + kv.first);
    t.data = base + b;
    t.owner = buf;
    out.emplace(kv.first, std::move(t));
  }
  return out;
}

Tensor to_dtype(const Tensor& t, DType dt) {
  if (t.dtype == dt) return t;
  return own_copy(t, dt);
}

Tensor own_copy(const Tensor& t, DType dt) {
  std::vector<uint8_t> bytes(static_cast<size_t>(t.numel()) * dtype_size(dt));
  convert(t.data, t.dtype, bytes.data(), dt, static_cast<size_t>(t.numel()));
  Tensor o;
  o.dtype = dt;
  o.shape = t.shape;
  o.owner = heap_buffer(std::move(bytes));
  o.data = o.owner->data();
  return o;
}

// ---- thread pool ------------------------------------------------------------------------

ThreadPool::ThreadPool(int n_threads) : n_threads_(n_threads < 1 ? 1 : n_threads) {
  for (int i = 1; i < n_threads_; ++i) workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_start_.notify_all();
  for (auto& w : workers_) w.join();
}

void ThreadPool::run_chunks() {
  while (true) {
    int64_t b = next_.fetch_add(grain_);
    if (b >= n_) break;
    int64_t e = b + grain_ < n_ ? b + grain_ : n_;
    (*fn_)(b, e);
  }
}

void ThreadPool::worker_loop() {
  uint64_t seen = 0;
  while (true) {
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_start_.wait(lk, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
    }
    run_chunks();
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (--active_ == 0) cv_done_.notify_one();
    }
  }
}

void ThreadPool::parallel_for(int64_t n, int64_t grain,
                              const std::function<void(int64_t, int64_t)>& fn) {
  if (n <= 0) return;
  if (grain < 1) grain = 1;
  if (workers_.empty() || n <= grain) { fn(0, n); return; }
  {
    std::lock_guard<std::mutex> lk(mu_);
    fn_ = &fn;
    n_ = n;
    grain_ = grain;
    next_.store(0);
    active_ = static_cast<int>(workers_.size());
    ++generation_;
  }
  cv_start_.notify_all();
  run_chunks();  // the calling thread works too
  std::unique_lock<std::mutex> lk(mu_);
  cv_done_.wait(lk, [&] { return active_ == 0; });
}

// ---- dot products ---------------------------------------------------------------------------

static float dot_f32(const float* x, const float* w, int64_t n) {
  int64_t i = 0;
#ifdef MDI_NEON
  float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
  for (; i + 16 <= n; i += 16) {
    a0 = vfmaq_f32(a0, vld1q_f32(x + i), vld1q_f32(w + i));
    a1 = vfmaq_f32(a1, vld1q_f32(x + i + 4), vld1q_f32(w + i + 4));
    a2 = vfmaq_f32(a2, vld1q_f32(x + i + 8), vld1q_f32(w + i + 8));
    a3 = vfmaq_f32(a3, vld1q_f32(x + i + 12), vld1q_f32(w + i + 12));
  }
  float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
#else
  float acc[8] = {0};
  for (; i + 8 <= n; i += 8)
    for (int k = 0; k < 8; ++k) acc[k] += x[i + k] * w[i + k];
  float s = 0;
  for (float a : acc) s += a;
#endif
  for (; i < n; ++i) s += x[i] * w[i];
  return s;
}

static float dot_bf16(const float* x, const uint16_t* w, int64_t n) {
  int64_t i = 0;
#ifdef MDI_NEON
  float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
  for (; i + 16 <= n; i += 16) {
    uint16x8_t w0 = vld1q_u16(w + i), w1 = vld1q_u16(w + i + 8);
    a0 = vfmaq_f32(a0, vld1q_f32(x + i), vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w0), 16)));
    a1 = vfmaq_f32(a1, vld1q_f32(x + i + 4), vreinterpretq_f32_u32(vshll_high_n_u16(w0, 16)));
    a2 = vfmaq_f32(a2, vld1q_f32(x + i + 8), vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w1), 16)));
    a3 = vfmaq_f32(a3, vld1q_f32(x + i + 12), vreinterpretq_f32_u32(vshll_high_n_u16(w1, 16)));
  }
  float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
#else
  float acc[8] = {0};
  for (; i + 8 <= n; i += 8)
    for (int k = 0; k < 8; ++k) acc[k] += x[i + k] * bf16_to_f32(w[i + k]);
  float s = 0;
  for (float a : acc) s += a;
#endif
  for (; i < n; ++i) s += x[i] * bf16_to_f32(w[i]);
  return s;
}

static float dot_f16(const float* x, const uint16_t* w, int64_t n) {
  int64_t i = 0;
#ifdef MDI_NEON
  float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
  const float16_t* wh = reinterpret_cast<const float16_t*>(w);
  for (; i + 16 <= n; i += 16) {
    float16x8_t w0 = vld1q_f16(wh + i), w1 = vld1q_f16(wh + i + 8);
    a0 = vfmaq_f32(a0, vld1q_f32(x + i), vcvt_f32_f16(vget_low_f16(w0)));
    a1 = vfmaq_f32(a1, vld1q_f32(x + i + 4), vcvt_high_f32_f16(w0));
    a2 = vfmaq_f32(a2, vld1q_f32(x + i + 8), vcvt_f32_f16(vget_low_f16(w1)));
    a3 = vfmaq_f32(a3, vld1q_f32(x + i + 12), vcvt_high_f32_f16(w1));
  }
  float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
#else
  float s = 0;
#endif
  for (; i < n; ++i) s += x[i] * f16_to_f32(w[i]);
  return s;
}

void linear(ThreadPool& pool, const float* x, int64_t T, const Tensor& W, const Tensor* b, float* y) {
  const int64_t out = W.shape[0], in = W.shape[1];
  const DType dt = W.dtype;
  const uint8_t* wbase = static_cast<const uint8_t*>(W.data);
  const size_t row_bytes = static_cast<size_t>(in) * dtype_size(dt);
  const bool has_bias = b && !b->empty();
  pool.parallel_for(out, 16, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const void* w = wbase + static_cast<size_t>(r) * row_bytes;
      float bias = has_bias ? load_f(b->data, b->dtype, static_cast<size_t>(r)) : 0.f;
      for (int64_t t = 0; t < T; ++t) {
        const float* xt = x + t * in;
        float s;
        switch (dt) {
          case DType::F32: s = dot_f32(xt, static_cast<const float*>(w), in); break;
          case DType::BF16: s = dot_bf16(xt, static_cast<const uint16_t*>(w), in); break;
          case DType::F16: s = dot_f16(xt, static_cast<const uint16_t*>(w), in); break;
          default: throw std::runtime_error("linear: unsupported weight dtype");
        }
        y[t * out + r] = s + bias;
      }
    }
  });
}

}  // namespace mdi
