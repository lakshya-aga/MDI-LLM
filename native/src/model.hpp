// Transformer blocks of a secondary node (LitGPT layout: Llama, Gemma, GPT-NeoX/Pythia,
// ...), with a per-sample KV cache. Matches `src/sub/model.py` (Block,
// CausalSelfAttention with the grouped KV cache, RMSNorm/LayerNorm, MLPs).
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "json.hpp"
#include "tensor.hpp"

namespace mdi {

struct ModelConfig {
  std::string name;
  int64_t n_embd = 0, n_head = 0, n_query_groups = 0, head_size = 0;
  int64_t intermediate_size = 0, rope_n_elem = 0, block_size = 0;
  double rope_base = 10000, rope_condense_ratio = 1, norm_eps = 1e-5;
  bool rms_norm = false, norm_unit_offset = false;  // Gemma: x * (1 + w)
  enum class Mlp { LLaMA, Gemma, GptNeox } mlp = Mlp::LLaMA;
  bool gelu_tanh = false;
  bool parallel_residual = false, shared_attention_norm = false;

  // From `Config.asdict()` (the init message's "model_config")
  static ModelConfig from_json(const Json& j);
};

struct Layer {
  Tensor norm_1_w, norm_1_b, norm_2_w, norm_2_b;
  Tensor attn_w, attn_b, proj_w, proj_b;
  Tensor fc_1_w, fc_1_b, fc_2_w, fc_2_b;  // GptNeox: fc_1 holds "mlp.fc"
  Tensor mlp_proj_w, mlp_proj_b;
};

// KV cache of one sample: per layer, [n_query_groups, max_seq, head_size] for k and v
struct SampleCache {
  std::vector<std::vector<float>> k, v;
  int64_t pos = 0;  // next position to write (= tokens processed so far)
};

class SecondaryModel {
 public:
  // `weights`: state dict of the chunk (keys "transformer.h.<i>....").
  // `weight_dtype`: storage dtype; weights are converted if they differ (F32 keeps
  // full precision, F16/BF16 halve memory). Pass nullptr to keep the given dtype.
  SecondaryModel(const ModelConfig& cfg, int n_layers, int64_t max_seq_length,
                 const std::map<std::string, Tensor>& weights, const DType* weight_dtype,
                 int n_threads);

  // x: [T, n_embd] f32, processed in place. Positions start at the sample's cache
  // position (the first call is the prompt). Creates the cache on first use.
  void forward(int sample, float* x, int64_t T);
  void drop_sample(int sample) { caches_.erase(sample); }
  void clear_samples() { caches_.clear(); }

  int n_layers() const { return static_cast<int>(layers_.size()); }
  int64_t max_seq_length() const { return max_seq_; }
  size_t weight_bytes() const;
  size_t cache_bytes_per_sample() const;
  const ModelConfig& config() const { return cfg_; }

 private:
  void norm(const float* x, int64_t T, const Tensor& w, const Tensor& b, float* out) const;
  void attention(const Layer& L, SampleCache& c, int layer, const float* xn, int64_t T, float* out);
  void mlp(const Layer& L, const float* xn, int64_t T, float* out);
  SampleCache& cache_for(int sample);

  ModelConfig cfg_;
  int64_t max_seq_;
  std::vector<Layer> layers_;
  std::vector<float> cos_, sin_;  // [max_seq, rope_n_elem]
  std::map<int, SampleCache> caches_;
  std::unique_ptr<ThreadPool> pool_;
  // scratch buffers (resized as needed)
  std::vector<float> xn_, attn_out_, qkv_, y_, h1_, h2_, tmp_;
};

}  // namespace mdi
