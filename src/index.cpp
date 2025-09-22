#include "index.hpp"
#include <hnswlib/hnswlib.h>
#include <filesystem>
#include <stdexcept>

struct Index::Impl {
  std::unique_ptr<hnswlib::L2Space> space;   // cosine/IP often work better; L2 is fine if vectors are L2-normalized
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
  size_t next_id = 0;
};

Index::Index(const std::string& path, int dim, int M, int efC, int efS)
  : path_(path), dim_(dim), M_(M), efC_(efC), efS_(efS), created_(false), impl_(new Impl) {}

Index::~Index() = default;

void Index::load() {
  impl_->space.reset(new hnswlib::L2Space(dim_));
  if (std::filesystem::exists(path_)) {
    impl_->hnsw.reset(new hnswlib::HierarchicalNSW<float>(impl_->space.get(), path_));
    impl_->hnsw->setEf(efS_);
    impl_->next_id = impl_->hnsw->cur_element_count;
    created_ = true;
  } else {
    // create empty index
    impl_->hnsw.reset(new hnswlib::HierarchicalNSW<float>(impl_->space.get(), 10000 /*max_elements*/, M_, efC_));
    impl_->hnsw->setEf(efS_);
    impl_->next_id = 0;
    created_ = true;
  }
}

void Index::save() const {
  if (!created_) return;
  impl_->hnsw->saveIndex(path_);
}

void Index::add(const std::vector<float>& vec) {
  if (!created_) load();
  if ((int)vec.size() != dim_) throw std::runtime_error("Index::add dimension mismatch");
  impl_->hnsw->addPoint((void*)vec.data(), impl_->next_id++);
}

std::vector<int> Index::search(const std::vector<float>& q, int k) const {
  if (!created_) throw std::runtime_error("Index not initialized");
  if ((int)q.size() != dim_) throw std::runtime_error("Index::search dimension mismatch");
  auto res = impl_->hnsw->searchKnn((void*)q.data(), k);
  std::vector<int> ids; ids.reserve(k);
  while (!res.empty()) { ids.push_back(res.top().second); res.pop(); }
  // hnsw returns nearest first at top; reversing keeps “closest → farthest”
  std::reverse(ids.begin(), ids.end());
  return ids;
}

size_t Index::size() const {
  return created_ ? impl_->hnsw->cur_element_count : 0;
}
