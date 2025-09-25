// src/planner.cpp
#include "planner.hpp"
#include <llama.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

using json = nlohmann::json;

namespace {
std::string trim(const std::string& s) {
  auto a = s.find_first_not_of(" \t\r\n");
  auto b = s.find_last_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  return s.substr(a, b - a + 1);
}
std::string extract_first_json_object(const std::string& text) {
  size_t start = text.find('{');
  if (start == std::string::npos) return "";
  int depth = 0;
  for (size_t i = start; i < text.size(); ++i) {
    char c = text[i];
    if (c == '{') depth++;
    else if (c == '}') {
      if (--depth == 0) return text.substr(start, i - start + 1);
    }
  }
  return "";
}
}

static const char* SYSTEM_INSTRUCTIONS =
"Convert the user's natural-language search into a conservative JSON plan.\n"
"Return ONLY a single JSON object with keys:\n"
"{\"filters\": [\"...\"], \"regex\": [\"...\"], \"time_from\": \"\", \"time_to\": \"\"}\n"
"- Keep regex short and safe (RE2 syntax). No catastrophic patterns.\n"
"- Use filters as plain keywords or globs like \"*.log\".\n"
"- Leave time fields empty strings if not specified.\n";

struct Planner::Impl {
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;
  const llama_vocab* vocab = nullptr;
  int n_ctx = 2048;

  explicit Impl(const std::string& model_path) {
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    model = llama_load_model_from_file(model_path.c_str(), mp);
    if (!model) throw std::runtime_error("planner: failed to load model");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.embeddings = false;
    ctx = llama_new_context_with_model(model, cp);
    if (!ctx) {
      llama_free_model(model);
      throw std::runtime_error("planner: failed to create context");
    }

    vocab = llama_model_get_vocab(model);
  }

  ~Impl() {
    if (ctx) llama_free(ctx);
    if (model) llama_free_model(model);
    llama_backend_free();
  }

  std::vector<llama_token> tokenize(const std::string& s, bool add_bos=true) {
    int32_t need = llama_tokenize(vocab, s.c_str(), (int32_t)s.size(), nullptr, 0, add_bos, /*special=*/false);
    if (need <= 0) throw std::runtime_error("planner: tokenize failed (len)");
    std::vector<llama_token> t(need);
    int32_t n = llama_tokenize(vocab, s.c_str(), (int32_t)s.size(), t.data(), (int32_t)t.size(), add_bos, false);
    if (n != need) throw std::runtime_error("planner: tokenize failed");
    return t;
  }

  llama_token argmax_token() {
    const float* logits = llama_get_logits(ctx);
    const int n_vocab = llama_n_vocab(vocab);
    int best = 0;
    float bestLogit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
      if (logits[i] > bestLogit) { bestLogit = logits[i]; best = i; }
    }
    return (llama_token)best;
  }

  std::string token_to_string(llama_token tok) {
    // get length
    size_t need = llama_token_to_piece(vocab, tok, nullptr, 0, /*special*/ false);
    std::string s; s.resize(need);
    llama_token_to_piece(vocab, tok, s.data(), need, false);
    return s;
  }

  std::string generate_json_plan(const std::string& prompt) {
    auto toks = tokenize(prompt);

    // feed prompt
    {
      llama_batch batch = llama_batch_init((int)toks.size(), 0, 1);
      llama_seq_id seq0 = 0;
      for (int i = 0; i < (int)toks.size(); ++i) {
        llama_batch_add(batch, toks[i], /*pos*/ i, &seq0, 1, /*logits*/ false);
      }
      if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("planner: decode(prompt) failed");
      }
      llama_batch_free(batch);
    }

    std::string out;
    const int max_new = 256;
    llama_seq_id seq0 = 0;
    int pos = (int)toks.size();
    for (int t = 0; t < max_new; ++t) {
      // request logits for next token by feeding a dummy (last token is implicit in ctx)
      llama_batch batch = llama_batch_init(1, 0, 1);
      // add a repeat of last token position to obtain next-step logits
      llama_batch_add(batch, llama_token_eos(vocab), /*pos*/ pos, &seq0, 1, /*logits*/ true);
      if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        break;
      }
      llama_batch_free(batch);

      llama_token tok = argmax_token();
      if (tok == llama_token_eos(vocab)) break;

      out += token_to_string(tok);
      ++pos;

      auto cand = extract_first_json_object(out);
      if (!cand.empty()) return trim(cand);
    }
    return trim(extract_first_json_object(out));
  }
};

Plan Planner::compile(const std::string& natural_query) {
  std::string prompt;
  prompt.reserve(2048);
  prompt.append(SYSTEM_INSTRUCTIONS);
  prompt.append("\nUser:\n");
  prompt.append(natural_query);
  prompt.append("\nJSON:");

  std::string raw = impl_->generate_json_plan(prompt);

  Plan p;
  if (raw.empty()) return p;
  try {
    auto j = json::parse(raw);
    if (j.contains("filters")) for (auto& s : j["filters"]) p.filters.push_back(s.get<std::string>());
    if (j.contains("regex"))   for (auto& s : j["regex"])   p.regex.push_back(s.get<std::string>());
    if (j.contains("time_from")) p.time_from = j["time_from"].get<std::string>();
    if (j.contains("time_to"))   p.time_to   = j["time_to"].get<std::string>();
  } catch (...) { /* ignore */ }
  return p;
}

Planner::Planner(const std::string& model_path) : impl_(new Impl(model_path)) {}
Planner::~Planner() { delete impl_; }
