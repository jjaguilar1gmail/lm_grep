#pragma once
#include <string>

struct Args {
  std::string mode;          // "index" or "query"
  std::string root_path;
  std::string sqlite_path = "./index/chunks.sqlite";
  std::string hnsw_path   = "./index/vectors.hnsw";
  std::string instruct_model = "./models/instruct.gguf";
  std::string embed_model    = "./models/embed.gguf";
  std::string query;
  int k = 80;
  int max_hits = 20;
  int chunk_size = 150;
  int chunk_overlap = 20;
};

Args parse_cli(int argc, char** argv);
