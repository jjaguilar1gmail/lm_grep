#pragma once
#include <string>
#include <vector>

struct Chunk {
  int id;                 // implicit vector index id
  std::string file;
  int ls;                 // line start
  int le;                 // line end
  size_t byte_start;      // inclusive
  size_t byte_end;        // exclusive
};

class Store {
public:
  explicit Store(const std::string& sqlite_path);
  ~Store();

  void ensure_schema();
  void upsert_chunk(const Chunk& c);
  Chunk get_chunk(int id) const;

private:
  struct Impl;
  Impl* impl_;
};
