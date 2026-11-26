#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mdi {

ModelConfig ModelConfig::from_json(const Json& j) {
  ModelConfig c;
  c.name = j.str_or("name", "");
  c.n_embd = j["n_embd"].as_int();
  c.n_head = j["n_head"].as_int();
  c.head_size = static_cast<int64_t>(j.num_or("head_size", static_cast<double>(c.n_embd / c.n_head)));
  c.n_query_groups = static_cast<int64_t>(j.num_or("n_query_groups", static_cast<double>(c.n_head)));
  c.intermediate_size = static_cast<int64_t>(j.num_or("intermediate_size", static_cast<double>(4 * c.n_embd)));
  c.block_size = j["block_size"].as_int();
  double rotary_percentage = j.num_or("rotary_percentage", 0.25);
  c.rope_n_elem = static_cast<int64_t>(rotary_percentage * static_cast<double>(c.head_size));
  c.rope_base = j.num_or("rope_base", 10000);
  c.rope_condense_ratio = j.num_or("rope_condense_ratio", 1);
  c.norm_eps = j.num_or("norm_eps", 1e-5);
  c.parallel_residual = j.bool_or("parallel_residual", true);
  c.shared_attention_norm = j.bool_or("shared_attention_norm", false);
  c.gelu_tanh = j.str_or("gelu_approximate", "none") == "tanh";

  std::string norm = j.str_or("norm_class_name", "LayerNorm");
  if (norm == "RMSNorm") {
    c.rms_norm = true;
    c.norm_unit_offset = c.name.find("Gemma") != std::string::npos;
  } else if (norm != "LayerNorm") {
    throw std::runtime_error("Unsupported norm class " + norm);
  }
  std::string mlp = j.str_or("mlp_class_name", "GptNeoxMLP");
  if (mlp == "LLaMAMLP") c.mlp = Mlp::LLaMA;
  else if (mlp == "GemmaMLP") c.mlp = Mlp::Gemma;
  else if (mlp == "GptNeoxMLP") c.mlp = Mlp::GptNeox;
  else throw std::runtime_error("Unsupported MLP class " + mlp + " (the native node has no MoE)");

  if (c.n_head % c.n_query_groups != 0) throw std::runtime_error("n_head % n_query_groups != 0");
  if (c.rope_n_elem % 2 != 0) throw std::runtime_error("rope_n_elem must be even");
  if (!c.parallel_residual && c.shared_attention_norm)
    throw std::runtime_error("non-parallel residual with shared attention norm is unsupported");
  return c;
}

namespace {
Tensor get(const std::map<std::string, Tensor>& w, const std::string& key, bool required) {
  auto it = w.find(key);
  if (it == w.end()) {
    if (required) throw std::runtime_error("Missing weight " + key);
    return Tensor();
  }
  return it->second;
}

void expect_shape(const Tensor& t, std::vector<int64_t> shape, const std::string& what) {
  if (t.empty()) return;
  if (t.shape != shape) {
    std::string s;
    for (auto d : t.shape) s += std::to_string(d) + ",";
    throw std::runtime_error("Unexpected shape for " + what + ": [" + s + "]");
  }
}
}  // namespace

SecondaryModel::SecondaryModel(const ModelConfig& cfg, int n_layers, int64_t max_seq_length,
                               const std::map<std::string, Tensor>& weights,
                               const DType* weight_dtype, int n_threads)
    : cfg_(cfg), max_seq_(max_seq_length > 0 ? max_seq_length : cfg.block_size) {
  if (max_seq_ > cfg_.block_size) throw std::runtime_error("max_seq_length exceeds block size");
  pool_ = std::make_unique<ThreadPool>(n_threads);

  const int64_t C = cfg_.n_embd, hs = cfg_.head_size, I = cfg_.intermediate_size;
  const int64_t qkv_rows = (cfg_.n_head + 2 * cfg_.n_query_groups) * hs;
  auto mat = [&](const std::string& key, bool required, std::vector<int64_t> shape) {
    Tensor t = get(weights, key, required);
    expect_shape(t, shape, key);
    if (!t.empty() && weight_dtype) t = to_dtype(t, *weight_dtype);
    return t;
  };
  auto vec = [&](const std::string& key, bool required, int64_t n) {
    Tensor t = get(weights, key, required);
    expect_shape(t, {n}, key);
    return t.empty() ? t : own_copy(t, DType::F32);  // small: own f32 copy
  };

  for (int i = 0; i < n_layers; ++i) {
    const std::string p = "transformer.h." + std::to_string(i) + ".";
    Layer L;
    L.norm_1_w = vec(p + "norm_1.weight", true, C);
    L.norm_1_b = vec(p + "norm_1.bias", false, C);
    if (!cfg_.shared_attention_norm) {
      L.norm_2_w = vec(p + "norm_2.weight", true, C);
      L.norm_2_b = vec(p + "norm_2.bias", false, C);
    }
    L.attn_w = mat(p + "attn.attn.weight", true, {qkv_rows, C});
    L.attn_b = vec(p + "attn.attn.bias", false, qkv_rows);
    L.proj_w = mat(p + "attn.proj.weight", true, {C, cfg_.n_head * hs});
    L.proj_b = vec(p + "attn.proj.bias", false, C);
    if (cfg_.mlp == ModelConfig::Mlp::GptNeox) {
      L.fc_1_w = mat(p + "mlp.fc.weight", true, {I, C});
      L.fc_1_b = vec(p + "mlp.fc.bias", false, I);
    } else {
      L.fc_1_w = mat(p + "mlp.fc_1.weight", true, {I, C});
      L.fc_1_b = vec(p + "mlp.fc_1.bias", false, I);
      L.fc_2_w = mat(p + "mlp.fc_2.weight", true, {I, C});
      L.fc_2_b = vec(p + "mlp.fc_2.bias", false, I);
    }
    L.mlp_proj_w = mat(p + "mlp.proj.weight", true, {C, I});
    L.mlp_proj_b = vec(p + "mlp.proj.bias", false, C);
    layers_.push_back(std::move(L));
  }

  // RoPE cache, computed in float32 as `build_rope_cache`
  const int64_t n = cfg_.rope_n_elem, half = n / 2;
  cos_.resize(static_cast<size_t>(max_seq_ * n));
  sin_.resize(cos_.size());
  std::vector<float> theta(static_cast<size_t>(half));
  for (int64_t i = 0; i < half; ++i)
    theta[i] = 1.0f / std::pow(static_cast<float>(cfg_.rope_base),
                               static_cast<float>(2 * i) / static_cast<float>(n));
  for (int64_t p = 0; p < max_seq_; ++p) {
    float pos = static_cast<float>(p) / static_cast<float>(cfg_.rope_condense_ratio);
    for (int64_t j = 0; j < n; ++j) {
      float a = pos * theta[j % half];
      cos_[p * n + j] = std::cos(a);
      sin_[p * n + j] = std::sin(a);
    }
  }
}

size_t SecondaryModel::weight_bytes() const {
  size_t s = 0;
  for (auto& L : layers_)
    for (const Tensor* t : {&L.norm_1_w, &L.norm_1_b, &L.norm_2_w, &L.norm_2_b, &L.attn_w, &L.attn_b,
                            &L.proj_w, &L.proj_b, &L.fc_1_w, &L.fc_1_b, &L.fc_2_w, &L.fc_2_b,
                            &L.mlp_proj_w, &L.mlp_proj_b})
      if (!t->empty()) s += t->nbytes();
  return s;
}

size_t SecondaryModel::cache_bytes_per_sample() const {
  return layers_.size() * 2 * static_cast<size_t>(cfg_.n_query_groups * max_seq_ * cfg_.head_size) *
         sizeof(float);
}

SampleCache& SecondaryModel::cache_for(int sample) {
  auto it = caches_.find(sample);
  if (it != caches_.end()) return it->second;
  SampleCache c;
  size_t per = static_cast<size_t>(cfg_.n_query_groups * max_seq_ * cfg_.head_size);
  c.k.assign(layers_.size(), std::vector<float>(per, 0.f));
  c.v.assign(layers_.size(), std::vector<float>(per, 0.f));
  return caches_.emplace(sample, std::move(c)).first->second;
}

void SecondaryModel::norm(const float* x, int64_t T, const Tensor& w, const Tensor& b, float* out) const {
  const int64_t C = cfg_.n_embd;
  const float* wf = static_cast<const float*>(w.data);
  const float* bf = b.empty() ? nullptr : static_cast<const float*>(b.data);
  const float eps = static_cast<float>(cfg_.norm_eps);
  for (int64_t t = 0; t < T; ++t) {
    const float* xt = x + t * C;
    float* ot = out + t * C;
    if (cfg_.rms_norm) {
      float ss = 0;
      for (int64_t i = 0; i < C; ++i) ss += xt[i] * xt[i];
      float r = 1.0f / std::sqrt(ss / static_cast<float>(C) + eps);
      for (int64_t i = 0; i < C; ++i)
        ot[i] = xt[i] * r * (cfg_.norm_unit_offset ? 1.0f + wf[i] : wf[i]);
    } else {
      float mean = 0;
      for (int64_t i = 0; i < C; ++i) mean += xt[i];
      mean /= static_cast<float>(C);
      float var = 0;
      for (int64_t i = 0; i < C; ++i) var += (xt[i] - mean) * (xt[i] - mean);
      var /= static_cast<float>(C);
      float r = 1.0f / std::sqrt(var + eps);
      for (int64_t i = 0; i < C; ++i) ot[i] = (xt[i] - mean) * r * wf[i] + (bf ? bf[i] : 0.f);
    }
  }
}

void SecondaryModel::attention(const Layer& L, SampleCache& c, int layer, const float* xn, int64_t T,
                               float* out) {
  const int64_t hs = cfg_.head_size, G = cfg_.n_query_groups, nh = cfg_.n_head;
  const int64_t qpk = nh / G, per_group = (qpk + 2) * hs, row = G * per_group;
  const int64_t n = cfg_.rope_n_elem, half = n / 2;
  const int64_t pos0 = c.pos;
  if (pos0 + T > max_seq_) throw std::runtime_error("Sequence exceeds the KV cache length");

  qkv_.resize(static_cast<size_t>(T * row));
  linear(*pool_, xn, T, L.attn_w, &L.attn_b, qkv_.data());

  // RoPE on q and k (first rope_n_elem dims), then write k/v to the cache
  auto rope = [&](float* v, int64_t p) {
    const float* cs = cos_.data() + p * n;
    const float* sn = sin_.data() + p * n;
    for (int64_t i = 0; i < half; ++i) {
      float x1 = v[i], x2 = v[i + half];
      v[i] = x1 * cs[i] - x2 * sn[i];
      v[i + half] = x2 * cs[i + half] + x1 * sn[i + half];
    }
  };
  float* K = c.k[layer].data();
  float* V = c.v[layer].data();
  for (int64_t t = 0; t < T; ++t) {
    const int64_t p = pos0 + t;
    float* r = qkv_.data() + t * row;
    for (int64_t g = 0; g < G; ++g) {
      float* grp = r + g * per_group;
      for (int64_t j = 0; j < qpk; ++j) rope(grp + j * hs, p);
      float* k = grp + qpk * hs;
      float* v = k + hs;
      rope(k, p);
      std::memcpy(K + (g * max_seq_ + p) * hs, k, static_cast<size_t>(hs) * sizeof(float));
      std::memcpy(V + (g * max_seq_ + p) * hs, v, static_cast<size_t>(hs) * sizeof(float));
    }
  }

  // Causal attention of every (token, head) over the cached positions [0, p]
  y_.resize(static_cast<size_t>(T * nh * hs));
  const float scale = 1.0f / std::sqrt(static_cast<float>(hs));
  pool_->parallel_for(T * nh, 1, [&](int64_t b, int64_t e) {
    thread_local std::vector<float> scores;
    for (int64_t task = b; task < e; ++task) {
      const int64_t t = task / nh, h = task % nh, g = h / qpk, j = h % qpk;
      const int64_t p = pos0 + t;
      const float* q = qkv_.data() + t * row + g * per_group + j * hs;
      const float* Kg = K + g * max_seq_ * hs;
      const float* Vg = V + g * max_seq_ * hs;
      scores.resize(static_cast<size_t>(p + 1));
      float mx = -INFINITY;
      for (int64_t s = 0; s <= p; ++s) {
        const float* k = Kg + s * hs;
        float d = 0;
        for (int64_t i = 0; i < hs; ++i) d += q[i] * k[i];
        scores[s] = d * scale;
        mx = std::max(mx, scores[s]);
      }
      float sum = 0;
      for (int64_t s = 0; s <= p; ++s) { scores[s] = std::exp(scores[s] - mx); sum += scores[s]; }
      float* yo = y_.data() + (t * nh + h) * hs;
      std::fill(yo, yo + hs, 0.f);
      for (int64_t s = 0; s <= p; ++s) {
        const float w = scores[s] / sum;
        const float* v = Vg + s * hs;
        for (int64_t i = 0; i < hs; ++i) yo[i] += w * v[i];
      }
    }
  });

  linear(*pool_, y_.data(), T, L.proj_w, &L.proj_b, out);
}

static inline float gelu(float x, bool tanh_approx) {
  if (tanh_approx) {
    const float k = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
  }
  return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865476f));
}

void SecondaryModel::mlp(const Layer& L, const float* xn, int64_t T, float* out) {
  const int64_t I = cfg_.intermediate_size;
  h1_.resize(static_cast<size_t>(T * I));
  linear(*pool_, xn, T, L.fc_1_w, &L.fc_1_b, h1_.data());
  if (cfg_.mlp == ModelConfig::Mlp::GptNeox) {
    for (auto& v : h1_) v = gelu(v, cfg_.gelu_tanh);
  } else {
    h2_.resize(h1_.size());
    linear(*pool_, xn, T, L.fc_2_w, &L.fc_2_b, h2_.data());
    if (cfg_.mlp == ModelConfig::Mlp::LLaMA) {
      for (size_t i = 0; i < h1_.size(); ++i) h1_[i] = h1_[i] / (1.0f + std::exp(-h1_[i])) * h2_[i];
    } else {
      for (size_t i = 0; i < h1_.size(); ++i) h1_[i] = gelu(h1_[i], cfg_.gelu_tanh) * h2_[i];
    }
  }
  linear(*pool_, h1_.data(), T, L.mlp_proj_w, &L.mlp_proj_b, out);
}

void SecondaryModel::forward(int sample, float* x, int64_t T) {
  const int64_t C = cfg_.n_embd;
  SampleCache& c = cache_for(sample);
  xn_.resize(static_cast<size_t>(T * C));
  attn_out_.resize(xn_.size());
  tmp_.resize(xn_.size());
  for (size_t li = 0; li < layers_.size(); ++li) {
    const Layer& L = layers_[li];
    norm(x, T, L.norm_1_w, L.norm_1_b, xn_.data());
    attention(L, c, static_cast<int>(li), xn_.data(), T, attn_out_.data());
    if (cfg_.parallel_residual) {
      const float* xn2 = xn_.data();
      if (!cfg_.shared_attention_norm) {
        norm(x, T, L.norm_2_w, L.norm_2_b, tmp_.data());
        xn2 = tmp_.data();
      }
      std::vector<float> m(static_cast<size_t>(T * C));
      mlp(L, xn2, T, m.data());
      for (int64_t i = 0; i < T * C; ++i) x[i] = m[i] + attn_out_[i] + x[i];
    } else {
      for (int64_t i = 0; i < T * C; ++i) x[i] = attn_out_[i] + x[i];
      norm(x, T, L.norm_2_w, L.norm_2_b, tmp_.data());
      mlp(L, tmp_.data(), T, xn_.data());  // xn_ reused as the MLP output
      for (int64_t i = 0; i < T * C; ++i) x[i] = xn_[i] + x[i];
    }
  }
  c.pos += T;
}

}  // namespace mdi
