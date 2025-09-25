// src/embedder.cpp
#include "embedder.hpp"
#include <llama.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

struct Embedder::Impl {
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;
  const llama_vocab* vocab = nullptr;
  int n_ctx = 1024;
  int dim = 0;

  explicit Impl(const std::string& model_path) {
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0; // CPU
    model = llama_load_model_from_file(model_path.c_str(), mp);
    if (!model) throw std::runtime_error("embedder: failed to load model");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.embeddings = true;                // IMPORTANT for embeddings
    ctx = llama_new_context_with_model(model, cp);
    if (!ctx) {
      llama_free_model(model);
      throw std::runtime_error("embedder: failed to create context");
    }

    vocab = llama_model_get_vocab(model);
    dim = llama_n_embd(model);
    if (dim <= 0) throw std::runtime_error("embedder: invalid embedding dim");
  }

  ~Impl() {
    if (ctx) llama_free(ctx);
    if (model) llama_free_model(model);
    llama_backend_free();
  }

  std::vector<llama_token> tokenize(const std::string& text) {
    // first pass for length
    int32_t needed = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                                    nullptr, 0, /*add_bos=*/true, /*special=*/false);
    if (needed <= 0) throw std::runtime_error("embedder: tokenize failed (len)");
    std::vector<llama_token> toks(needed);
    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                               toks.data(), (int32_t)toks.size(),
                               /*add_bos=*/true, /*special=*/false);
    if (n != needed) throw std::runtime_error("embedder: tokenize failed");
    return toks;
  }

  std::vector<float> encode_text(const std::string& text) {
    auto toks = tokenize(text);

    // Build a batch for the whole prompt
    llama_batch batch = llama_batch_init((int)toks.size(), /*embd*/ 0, /*n_seq*/ 1);
    llama_seq_id seq0 = 0;
    for (int i = 0; i < (int)toks.size(); ++i) {
      llama_batch_add(batch, toks[i], /*pos*/ i, &seq0, 1, /*logits*/ false);
    }
    if (llama_decode(ctx, batch) != 0) {
      llama_batch_free(batch);
      throw std::runtime_error("embedder: llama_decode failed");
    }
    llama_batch_free(batch);

    const float* emb = llama_get_embeddings(ctx);
    if (!emb) throw std::runtime_error("embedder: embeddings null");

    std::vector<float> v(emb, emb + dim);
    // L2 normalize
    double s = 0.0; for (float x : v) s += (double)x * (double)x;
    float norm = (float)std::sqrt(std::max(s, 1e-12));
    for (auto& x : v) x /= norm;
    return v;
  }
};

Embedder::Embedder(const std::string& embed_model_path)
  : impl_(new Impl(embed_model_path)) {
  dim_ = impl_->dim;
}

Embedder::~Embedder() { delete impl_; }

std::vector<float> Embedder::encode(const std::string& text) {
  return impl_->encode_text(text);
}
