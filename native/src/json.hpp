// Minimal JSON value + parser/serializer (enough for node configs, safetensors
// headers and the init metadata). Not a general-purpose library.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mdi {

class Json {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Json() = default;
  Json(std::nullptr_t) {}
  Json(bool b) : type_(Type::Bool), b_(b) {}
  Json(double d) : type_(Type::Number), n_(d) {}
  Json(int i) : type_(Type::Number), n_(i) {}
  Json(int64_t i) : type_(Type::Number), n_(static_cast<double>(i)) {}
  Json(const char* s) : type_(Type::String), s_(s) {}
  Json(std::string s) : type_(Type::String), s_(std::move(s)) {}

  static Json array() { Json j; j.type_ = Type::Array; return j; }
  static Json object() { Json j; j.type_ = Type::Object; return j; }

  static Json parse(const std::string& text);
  std::string dump() const;

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_object() const { return type_ == Type::Object; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_string() const { return type_ == Type::String; }
  bool is_number() const { return type_ == Type::Number; }

  bool as_bool() const;
  double as_number() const;
  int64_t as_int() const { return static_cast<int64_t>(as_number()); }
  const std::string& as_string() const;
  const std::vector<Json>& items() const;
  const std::vector<std::pair<std::string, Json>>& members() const;

  bool has(const std::string& key) const;
  // Object access; throws if missing
  const Json& operator[](const std::string& key) const;
  const Json& operator[](size_t i) const { return items().at(i); }
  size_t size() const;

  // Value or default when the key is missing or null
  double num_or(const std::string& key, double def) const;
  std::string str_or(const std::string& key, const std::string& def) const;
  bool bool_or(const std::string& key, bool def) const;

  // Builders
  Json& set(const std::string& key, Json v);
  Json& push(Json v);

 private:
  Type type_ = Type::Null;
  bool b_ = false;
  double n_ = 0;
  std::string s_;
  std::vector<Json> a_;
  std::vector<std::pair<std::string, Json>> o_;  // keeps insertion order

  friend class JsonParser;
  void dump_to(std::string& out) const;
};

}  // namespace mdi
