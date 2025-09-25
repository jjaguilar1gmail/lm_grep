// src/filters.cpp
#include "filters.hpp"
#include "planner.hpp"
#include "store.hpp"
#include <re2/re2.h>
#include <algorithm>
#include <fstream>
#include <string>

static std::string read_slice(const std::string& file, size_t b0, size_t b1, size_t max_bytes=2000) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return {};
  if (b1 > b0 && (b1 - b0) > max_bytes) b1 = b0 + max_bytes;
  in.seekg((std::streamoff)b0);
  std::string s; s.resize(b1 - b0);
  in.read(s.data(), (std::streamsize)s.size());
  return s;
}

std::vector<Hit> apply_filters(const std::vector<int>& cands,
                               const Plan& plan,
                               const Store& store,
                               int max_hits) {
  std::vector<RE2> regs;
  regs.reserve(plan.regex.size());
  for (auto& r : plan.regex) if (!r.empty()) regs.emplace_back(r);

  std::vector<Hit> hits;
  hits.reserve(std::min<int>(cands.size(), max_hits));

  for (int id : cands) {
    auto meta = store.get_chunk(id);
    std::string text = read_slice(meta.file, meta.byte_start, meta.byte_end);

    // keyword filter
    bool ok = true;
    if (!plan.filters.empty()) {
      std::string hay = meta.file + " " + text;
      std::transform(hay.begin(), hay.end(), hay.begin(), ::tolower);
      for (auto f : plan.filters) {
        std::transform(f.begin(), f.end(), f.begin(), ::tolower);
        if (hay.find(f) == std::string::npos) { ok = false; break; }
      }
    }
    if (!ok) continue;

    // regex pass
    if (!regs.empty()) {
      bool any = false;
      for (auto& re : regs) {
        if (RE2::PartialMatch(text, re)) { any = true; break; }
      }
      if (!any) continue;
    }

    Hit h{ id, meta.file, meta.ls, meta.le,
           text.size() > 300 ? text.substr(0,300) : text };
    hits.push_back(std::move(h));
    if ((int)hits.size() >= max_hits) break;
  }
  return hits;
}
