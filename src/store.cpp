#include "store.hpp"
#include <sqlite3.h>
#include <stdexcept>

struct Store::Impl {
  sqlite3* db = nullptr;
};

Store::Store(const std::string& path) : impl_(new Impl) {
  if (sqlite3_open(path.c_str(), &impl_->db) != SQLITE_OK) {
    throw std::runtime_error("sqlite open failed");
  }
  ensure_schema();
}

Store::~Store() {
  if (impl_) {
    if (impl_->db) sqlite3_close(impl_->db);
    delete impl_;
  }
}

void Store::ensure_schema() {
  const char* sql =
    "CREATE TABLE IF NOT EXISTS chunks ("
    " id INTEGER PRIMARY KEY,"
    " file TEXT NOT NULL,"
    " ls INTEGER NOT NULL,"
    " le INTEGER NOT NULL,"
    " byte_start INTEGER NOT NULL,"
    " byte_end INTEGER NOT NULL"
    ");";
  char* err=nullptr;
  if (sqlite3_exec(impl_->db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
    std::string e = err ? err : "unknown";
    sqlite3_free(err);
    throw std::runtime_error("sqlite schema: " + e);
  }
}

void Store::upsert_chunk(const Chunk& c) {
  const char* sql =
    "INSERT INTO chunks (id, file, ls, le, byte_start, byte_end) "
    "VALUES (?, ?, ?, ?, ?, ?) "
    "ON CONFLICT(id) DO UPDATE SET "
    " file=excluded.file, ls=excluded.ls, le=excluded.le, "
    " byte_start=excluded.byte_start, byte_end=excluded.byte_end;";
  sqlite3_stmt* st=nullptr;
  if (sqlite3_prepare_v2(impl_->db, sql, -1, &st, nullptr) != SQLITE_OK)
    throw std::runtime_error("sqlite prepare failed");
  sqlite3_bind_int(st, 1, c.id);
  sqlite3_bind_text(st, 2, c.file.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int(st, 3, c.ls);
  sqlite3_bind_int(st, 4, c.le);
  sqlite3_bind_int64(st, 5, (sqlite3_int64)c.byte_start);
  sqlite3_bind_int64(st, 6, (sqlite3_int64)c.byte_end);

  if (sqlite3_step(st) != SQLITE_DONE) {
    sqlite3_finalize(st);
    throw std::runtime_error("sqlite insert failed");
  }
  sqlite3_finalize(st);
}

Chunk Store::get_chunk(int id) const {
  const char* sql =
    "SELECT file, ls, le, byte_start, byte_end FROM chunks WHERE id=?";
  sqlite3_stmt* st=nullptr;
  if (sqlite3_prepare_v2(impl_->db, sql, -1, &st, nullptr) != SQLITE_OK)
    throw std::runtime_error("sqlite prepare failed");
  sqlite3_bind_int(st, 1, id);
  Chunk c{ id, "", 0, 0, 0, 0 };
  int rc = sqlite3_step(st);
  if (rc == SQLITE_ROW) {
    c.file = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
    c.ls   = sqlite3_column_int(st, 1);
    c.le   = sqlite3_column_int(st, 2);
    c.byte_start = (size_t)sqlite3_column_int64(st, 3);
    c.byte_end   = (size_t)sqlite3_column_int64(st, 4);
  } else {
    sqlite3_finalize(st);
    throw std::runtime_error("chunk id not found");
  }
  sqlite3_finalize(st);
  return c;
}
