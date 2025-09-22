#include "chunker.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>

using std::string;
namespace fs = std::filesystem;

static bool is_text_ext(const string& ext) {
  static const char* bad[] = {".png",".jpg",".jpeg",".tif",".tiff",".pdf",".zip",
                              ".mp4",".mov",".mp3",".wav",".ogg",".bin",".so",".dll"};
  for (auto* b : bad) if (ext == b) return false;
  return true;
}

std::vector<std::string> list_text_files(const std::string& root) {
  std::vector<string> out;
  for (auto& p : fs::recursive_directory_iterator(root)) {
    if (!p.is_regular_file()) continue;
    auto ext = p.path().extension().string();
    for (auto& c : ext) c = (char)tolower(c);
    if (!is_text_ext(ext)) continue;
    out.push_back(p.path().string());
  }
  return out;
}

std::vector<ChunkWithText> chunk_file(const std::string& path, int size, int overlap) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  // Read file once and keep line start offsets
  std::vector<size_t> line_offsets; line_offsets.push_back(0);
  std::string data;
  {
    std::ostringstream ss; ss << in.rdbuf(); data = ss.str();
  }
  for (size_t i = 0; i < data.size(); ++i) {
    if (data[i] == '\n') line_offsets.push_back(i+1);
  }
  // last sentinel
  line_offsets.push_back(data.size());

  int n_lines = (int)line_offsets.size() - 1;
  std::vector<ChunkWithText> chunks;
  for (int i = 0; i < n_lines; ) {
    int ls = i + 1;
    int le = std::min(n_lines, i + size);
    size_t b0 = line_offsets[ls-1];
    size_t b1 = line_offsets[le];
    std::string text = data.substr(b0, b1 - b0);

    ChunkWithText cwt;
    cwt.meta = Chunk{ /*id*/ -1, path, ls, le, b0, b1 };
    cwt.text = std::move(text);
    chunks.push_back(std::move(cwt));

    if (le == n_lines) break;
    i = le - overlap;
    if (i < 0) i = 0;
  }
  return chunks;
}

std::vector<ChunkWithText> chunk_folder(const std::string& root, int size, int overlap) {
  auto files = list_text_files(root);
  std::vector<ChunkWithText> all;
  for (auto& f : files) {
    auto v = chunk_file(f, size, overlap);
    all.insert(all.end(), v.begin(), v.end());
  }
  return all;
}
