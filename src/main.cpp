#include "cli.hpp"
#include "planner.hpp"
#include "embedder.hpp"
#include "index.hpp"
#include "store.hpp"
#include "chunker.hpp"

#include <iostream>
#include <fstream>

static std::string read_context(const std::string& file, size_t b0, size_t b1, int extra_lines=5) {
  // read a little more context by expanding byte window to include +/- extra_lines
  // naive approach: re-read and count newlines around the window
  std::ifstream in(file, std::ios::binary);
  if (!in) return {};
  std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

  // expand backwards by N newlines
  size_t start = b0;
  int back = extra_lines;
  while (start > 0 && back > 0) {
    start--;
    if (data[start] == '\n') back--;
  }
  // expand forwards by N newlines
  size_t end = b1;
  int fwd = extra_lines;
  while (end < data.size() && fwd > 0) {
    if (data[end] == '\n') fwd--;
    end++;
  }
  return data.substr(start, end - start);
}

int main(int argc, char** argv) {
  auto args = parse_cli(argc, argv);

  if (args.mode == "index") {
    Store store(args.sqlite_path);
    Index idx(args.hnsw_path, /*dim*/ 0); // will reset after embedder knows dim

    Embedder emb(args.embed_model);
    Index index(args.hnsw_path, emb.dim());
    index.load();

    auto chunks = chunk_folder(args.root_path, args.chunk_size, args.chunk_overlap);

    int id = (int)index.size();  // continue appending
    for (auto& cwt : chunks) {
      auto v = emb.encode(cwt.text);
      index.add(v);

      Chunk meta = cwt.meta;
      meta.id = id++;
      store.upsert_chunk(meta);

      if ((meta.id % 500) == 0) std::cerr << "Indexed up to id " << meta.id << "\n";
    }
    index.save();
    std::cerr << "Done.\n";
    return 0;
  }

  if (args.mode == "query") {
    Store store(args.sqlite_path);
    Planner planner(args.instruct_model);
    Embedder emb(args.embed_model);
    Index index(args.hnsw_path, emb.dim());
    index.load();

    auto plan = planner.compile(args.query);
    auto qv = emb.encode(args.query);
    auto ids = index.search(qv, args.k);

    // Apply light keyword/regex filtering inline (or use your earlier filters.cpp)
    // Here: load chunk meta, read the chunk text, apply simple contains (for brevity)
    // If you already have filters.cpp with RE2, call that instead.

    // Print plan
    std::cout << "Plan:\n  filters=";
    for (auto& f : plan.filters) std::cout << f << " ";
    std::cout << "\n  regex=";
    for (auto& r : plan.regex) std::cout << r << " ";
    std::cout << "\n\n";

    int shown = 0;
    for (int id : ids) {
      auto c = store.get_chunk(id);
      auto ctx = read_context(c.file, c.byte_start, c.byte_end, /*extra_lines=*/5);
      std::cout << c.file << ":" << c.ls << "-" << c.le << "\n";
      // truncate display
      if (ctx.size() > 1200) ctx.resize(1200);
      // show with line breaks intact
      std::cout << ctx << "\n---\n";
      if (++shown >= args.max_hits) break;
    }
    return 0;
  }

  return 1;
}
