#pragma once
#include <string>
#include <vector>

struct Plan {
  std::vector<std::string> filters; // file globs/keywords
  std::vector<std::string> regex;   // RE2 syntax
  std::string time_from;            // optional ISO
  std::string time_to;              // optional ISO
  // you can add numeric constraints later
};

class Planner {
public:
  Planner(const std::string& model_path);
  ~Planner();

  Plan compile(const std::string& natural_query);

private:
  void* ctx_; // forward-declare llama_context internally
};
