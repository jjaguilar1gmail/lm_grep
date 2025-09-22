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
  int n_ctx = 1024;
  int dim = 0;

  explicit Impl(const std::string& model_path) {
    // NOTE: In this process, the planner may have already called llama_backend_init().
    // llama.cpp safely handles repeated init/free across same process if paired.
    // If you want stricter control, centralize backend init in your app bootstrap.
    // llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0; // CPU by default
    model = llama_load_model_from_file(model_path.c_str(), mp);
    if (!model) throw std::runtime_error("Failed to load embedding model: " + model_path);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.embeddings = true;  // IMPORTANT: enable embeddings
    ctx = llama_new_context_with_model(model, cp);
    if (!ctx) {
      llama_free_model(model);
      throw std::runtime_error("Failed to create llama context for embeddings");
    }

    dim = llama_n_embd(model);
    if (dim <= 0) throw std::runtime_error("Embedding dimension not available");
  }

  ~Impl() {
    if (ctx) {
      llama_free(ctx);
      ctx = nullptr;
    }
    if (model) {
      llama_free_model(model);
      model = nullptr;
    }
    // Do NOT call llama_backend_free() here if planner also needs it later in same process.
    // Let your app call llama_backend_free() once at shutdown, or rely on OS cleanup.
  }

  std::vector<float> encode_text(const std::string& text) {
    // Tokenize (with beginning-of-sentence)
    std::vector<llama_token> tokens;
    tokens.resize(text.size() + 8);
    int n_toks = llama_tokenize(model, text.c_str(), (int)text.size(), tokens.data(), (int)tokens.size(), true, false);
    if (n_toks < 0) throw std::runtime_error("embed: tokenization failed");
    tokens.resize(n_toks);

    // Feed tokens; embeddings must be enabled in context params
    {
      llama_set_embeddings(ctx, true); // ensure on

      llama_batch batch = llama_batch_init(n_toks, /*embd*/ 1, /*n_seq*/ 1);
      for (int i = 0; i < n_toks; ++i) {
        // For embeddings, set embd=1 on the last token typically suffices,
        // but enabling embeddings on the context gives you pooled representation.
        llama_batch_add(batch, tokens[i], /*pos*/ i, /*seq_id*/ {0}, /*logits*/ false);
      }
      if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("embed: llama_decode failed");
      }
      llama_batch_free(batch);
    }

    const float* emb = llama_get_embeddings(ctx);
    if (!emb) {
      throw std::runtime_error("embed: llama_get_embeddings returned null");
    }

    std::vector<float> v(emb, emb + dim);

    // L2 normalize
    double sum = 0.0;
    for (float x : v) sum += (double)x * (double)x;
    float norm = (float)std::sqrt(std::max(sum, 1e-12));
    for (auto& x : v) x /= norm;
    return v;
  }
};

Embedder::Embedder(const std::string& embed_model_path)
  : ctx_(nullptr), dim_(0), impl_(new Impl(embed_model_path)) {
  ctx_ = impl_->ctx;
  dim_ = impl_->dim;
}

Embedder::~Embedder() {
  delete impl_;
}

std::vector<float> Embedder::encode(const std::string& text) {
  return impl_->encode_text(text);
}
