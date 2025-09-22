#include "planner.hpp"

#include <llama.h>
#include <nlohmann/json.hpp>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

using json = nlohmann::json;

namespace {

// Small helper: RAII for llama_batch
struct Batch {
  llama_batch b;
  Batch(int n_tokens, int embd = 0, int n_seq_max = 1) {
    b = llama_batch_init(n_tokens, embd, n_seq_max);
  }
  ~Batch() { llama_batch_free(b); }
};

std::string trim(const std::string& s) {
  auto a = s.find_first_not_of(" \t\r\n");
  auto b = s.find_last_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  return s.substr(a, b - a + 1);
}

// Extract first top-level JSON object from a string (best-effort)
std::string extract_first_json_object(const std::string& text) {
  size_t start = text.find('{');
  if (start == std::string::npos) return "";
  int depth = 0;
  for (size_t i = start; i < text.size(); ++i) {
    char c = text[i];
    if (c == '{') depth++;
    else if (c == '}') {
      depth--;
      if (depth == 0) {
        return text.substr(start, i - start + 1);
      }
    }
  }
  return "";
}

} // namespace

struct Planner::Impl {
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;
  int n_ctx = 2048;

  explicit Impl(const std::string& model_path) {
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    // If you have GPU, set n_gpu_layers accordingly; CPU-only keeps 0.
    mp.n_gpu_layers = 0;
    model = llama_load_model_from_file(model_path.c_str(), mp);
    if (!model) {
      throw std::runtime_error("Failed to load planner model: " + model_path);
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.embeddings = false;
    ctx = llama_new_context_with_model(model, cp);
    if (!ctx) {
      llama_free_model(model);
      throw std::runtime_error("Failed to create llama context for planner");
    }
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
    llama_backend_free();
  }

  // Deterministic generation with small top-k and temperature ~0
  std::string generate_json_plan(const std::string& prompt) {
    // Tokenize prompt
    std::vector<llama_token> tokens;
    tokens.resize(prompt.size() + 8);
    int n_toks = llama_tokenize(model, prompt.c_str(), (int)prompt.size(), tokens.data(), (int)tokens.size(), true, false);
    if (n_toks < 0) {
      throw std::runtime_error("planner: tokenization failed");
    }
    tokens.resize(n_toks);

    // Feed the prompt
    {
      Batch batch(/*n_tokens*/ std::max(32, n_toks + 16), /*embd*/ 0, /*n_seq*/ 1);
      for (int i = 0; i < n_toks; ++i) {
        llama_batch_add(batch.b, tokens[i], /*pos*/ i, /*seq_id*/ {0}, /*logits*/ false);
      }
      if (llama_decode(ctx, batch.b) != 0) {
        throw std::runtime_error("planner: llama_decode(prompt) failed");
      }
    }

    // Sampler: greedy-ish (temperature 0), very small top_k
    llama_sampler* smpl = llama_sampler_init(llama_sampler_chain_default_params());
    // force near-greedy
    llama_sampler_set_temperature(smpl, 0.0f);
    llama_sampler_set_top_k(smpl, 1);
    llama_sampler_set_top_p(smpl, 0.95f);
    llama_sampler_set_min_p(smpl, 0.05f);
    llama_sampler_set_max_len(smpl, 512);

    std::string out_text;
    const int max_new = 256;
    int cur_pos = n_toks;
    for (int t = 0; t < max_new; ++t) {
      // Get logits for next token
      llama_token new_tok = llama_sampler_sample(smpl, ctx, /*idx*/ -1);
      if (new_tok == llama_token_eos(model)) break;

      // Append
      out_text += llama_token_to_piece(model, new_tok);

      // Stop early if looks like we completed a JSON object
      auto candidate = extract_first_json_object(out_text);
      if (!candidate.empty()) {
        out_text = candidate;
        llama_sampler_free(smpl);
        return trim(out_text);
      }

      // Feed back the sampled token
      {
        Batch batch(1, 0, 1);
        llama_batch_add(batch.b, new_tok, cur_pos, {0}, true);
        if (llama_decode(ctx, batch.b) != 0) {
          break;
        }
        ++cur_pos;
      }
    }

    llama_sampler_free(smpl);
    // Last resort: try to extract JSON if not found
    return trim(extract_first_json_object(out_text));
  }
};

static const char* SYSTEM_INSTRUCTIONS =
"Convert the user's natural-language search into a conservative JSON plan.\n"
"Return ONLY a single JSON object with keys:\n"
"{\"filters\": [\"...\"], \"regex\": [\"...\"], \"time_from\": \"\", \"time_to\": \"\"}\n"
"- Keep regex short and safe (RE2 syntax). No catastrophic patterns.\n"
"- Use filters as plain keywords or globs like \"*.log\" if the user implies file types.\n"
"- If no times are specified, leave time fields empty strings.\n";

Planner::Planner(const std::string& model_path)
  : impl_(new Impl(model_path)) {}

Planner::~Planner() { delete impl_; }

Plan Planner::compile(const std::string& natural_query) {
  // Simple prompt format; you can switch to chat templates if your model expects them.
  std::string prompt;
  prompt.reserve(2048);
  prompt.append(SYSTEM_INSTRUCTIONS);
  prompt.append("\nUser:\n");
  prompt.append(natural_query);
  prompt.append("\nJSON:");

  std::string raw = impl_->generate_json_plan(prompt);

  Plan p;
  if (raw.empty()) {
    // Safe fallback
    return p;
  }

  try {
    auto j = json::parse(raw);
    if (j.contains("filters") && j["filters"].is_array()) {
      for (auto& s : j["filters"]) p.filters.push_back(s.get<std::string>());
    }
    if (j.contains("regex") && j["regex"].is_array()) {
      for (auto& s : j["regex"]) p.regex.push_back(s.get<std::string>());
    }
    if (j.contains("time_from")) p.time_from = j["time_from"].get<std::string>();
    if (j.contains("time_to"))   p.time_to   = j["time_to"].get<std::string>();
  } catch (...) {
    // ignore parse failures and return empty plan
  }
  return p;
}
