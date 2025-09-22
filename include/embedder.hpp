#include "embedder.hpp"
#pragma once
#include <string>
#include <vector>

class Embedder {
public:
  explicit Embedder(const std::string& embed_model_path);
  ~Embedder();

  std::vector<float> encode(const std::string& text);
  int dim() const { return dim_; }

private:
  struct Impl;
  Impl* impl_;
  void* ctx_;
  int dim_;
};
