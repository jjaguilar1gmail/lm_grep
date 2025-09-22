#pragma once
#include <string>
#include <vector>
#include <memory>

class Index {
public:
  Index(const std::string& path, int dim, int M=16, int efC=200, int efS=64);
  ~Index();

  void add(const std::vector<float>& vec);              // append-only
  std::vector<int> search(const std::vector<float>& q, int k) const;

  void save() const;   // writes to <path>
  void load();         // loads from <path> (if exists)

  int dim() const { return dim_; }
  size_t size() const;

private:
  std::string path_;
  int dim_, M_, efC_, efS_;
  bool created_;
  // pimpl so headers stay light
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
