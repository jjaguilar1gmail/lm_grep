cmake_minimum_required(VERSION 3.19)
project(llm_grep CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LLM_GREP_STATIC "Link statically where possible" OFF)
if(LLM_GREP_STATIC)
  if(UNIX AND NOT APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -static-libstdc++ -static-libgcc")
  endif()
endif()

# llama.cpp
add_subdirectory(third_party/llama.cpp)

# RE2
add_subdirectory(third_party/re2)

# simdjson
add_subdirectory(third_party/simdjson)

# HNSW (header-only)
include_directories(third_party/hnswlib)

# SQLite amalgamation
add_library(sqlite3 STATIC third_party/sqlite/sqlite3.c)
target_include_directories(sqlite3 PUBLIC third_party/sqlite)

include_directories(include)
add_executable(llm_grep
  src/main.cpp
  src/fs_utils.cpp
  src/chunker.cpp
  src/embedder.cpp
  src/planner.cpp
  src/index.cpp
  src/store.cpp
  src/filters.cpp
  src/cli.cpp
)

target_link_libraries(llm_grep
  PRIVATE
    llama         # from llama.cpp
    sqlite3
    re2
    simdjson
)

# Speed flags
if (MSVC)
  target_compile_options(llm_grep PRIVATE /O2 /DNOMINMAX)
else()
  target_compile_options(llm_grep PRIVATE -O3 -DNDEBUG)
endif()
