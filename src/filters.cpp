#include "filters.hpp"
#include <re2/re2.h>
#include <algorithm>

std::vector<Hit> apply_filters(const std::vector<int>& cands,
                               const Plan& plan,
                               const Store& store,
                               int max_hits) {
  std::vector<RE2> regs;
  regs.reserve(plan.regex.size());
  for (auto& r : plan.regex) regs.emplace_back(r);

  std::vector<Hit> hits; hits.reserve(std::min<int>(cands.size(), max_hits));
  for (int id : cands) {
    auto c = store.get_chunk(id);
    // keyword/glob filter (simplified): require each filter token to appear
    bool ok = true;
    for (auto& f : plan.filters) {
      if (f.empty()) continue;
      auto needle = f;
      std::string hay = c.file + " " + c.text;
      std::transform(hay.begin(), hay.end(), hay.begin(), ::tolower);
      std::transform(needle.begin(), needle.end(), needle.begin(), ::tolower);
      if (hay.find(needle) == std::string::npos) { ok = false; break; }
    }
    if (!ok) continue;

    // regex pass
    if (!regs.empty()) {
      bool any=false;
      for (auto& r : regs) {
        if (RE2::PartialMatch(c.text, r)) { any = true; break; }
      }
      if (!any) continue;
    }

    Hit h{ id, c.file, c.ls, c.le, c.text.substr(0, 300) };
    hits.push_back(std::move(h));
    if ((int)hits.size() >= max_hits) break;
  }
  return hits;
}
