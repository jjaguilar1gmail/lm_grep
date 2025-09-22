#pragma once
#include "planner.hpp"
#include "index.hpp"
#include <string>
#include <vector>

struct Hit {
  int id;
  std::string file;
  int ls;
  int le;
  std::string snippet;
};

std::vector<Hit> apply_filters(const std::vector<int>& candidates,
                               const Plan& plan,
                               const Store& store,
                               int max_hits);
