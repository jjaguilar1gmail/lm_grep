#include "cli.hpp"
#include <iostream>
#include <cstring>

static const char* USAGE =
"llm_grep index <root> [--sqlite path] [--hnsw path] [--embed-model path] [--chunk-size N] [--chunk-overlap N]\n"
"llm_grep query \"text\" [--sqlite path] [--hnsw path] [--instruct-model path] [--embed-model path] [-k N] [--max-hits N]\n";

Args parse_cli(int argc, char** argv) {
  Args a;
  if (argc < 2) { std::cerr << USAGE; std::exit(1); }
  a.mode = argv[1];
  int i = 2;
  if (a.mode == "index") {
    if (i >= argc) { std::cerr << USAGE; std::exit(1); }
    a.root_path = argv[i++];
  } else if (a.mode == "query") {
    if (i >= argc) { std::cerr << USAGE; std::exit(1); }
    a.query = argv[i++];
  } else {
    std::cerr << USAGE; std::exit(1);
  }

  while (i < argc) {
    std::string f = argv[i++];
    auto next = [&](std::string& dst){
      if (i >= argc) { std::cerr << "Missing value after " << f << "\n"; std::exit(1); }
      dst = argv[i++];
    };
    if (f == "--sqlite") next(a.sqlite_path);
    else if (f == "--hnsw") next(a.hnsw_path);
    else if (f == "--instruct-model") next(a.instruct_model);
    else if (f == "--embed-model") next(a.embed_model);
    else if (f == "-k") { std::string v; next(v); a.k = std::stoi(v); }
    else if (f == "--max-hits") { std::string v; next(v); a.max_hits = std::stoi(v); }
    else if (f == "--chunk-size") { std::string v; next(v); a.chunk_size = std::stoi(v); }
    else if (f == "--chunk-overlap") { std::string v; next(v); a.chunk_overlap = std::stoi(v); }
    else { std::cerr << "Unknown flag: " << f << "\n"; std::exit(1); }
  }
  return a;
}
