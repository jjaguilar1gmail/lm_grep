#pragma once
#include "store.hpp"
#include <string>
#include <vector>

struct ChunkWithText {
  Chunk meta;       // filled id later by caller
  std::string text; // chunk text
};

std::vector<std::string> list_text_files(const std::string& root);

// Chunk by a sliding window of lines (size, overlap)
// Records LS/LE and byte offsets (start/end) into the file for fast re-read.
std::vector<ChunkWithText> chunk_file(const std::string& path, int size=150, int overlap=20);

// Convenience: chunk an entire folder
std::vector<ChunkWithText> chunk_folder(const std::string& root, int size=150, int overlap=20);
