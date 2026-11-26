#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace mdi {

class JsonParser {
 public:
  explicit JsonParser(const std::string& t) : t_(t) {}

  Json parse_root() {
    Json v = value();
    ws();
    if (i_ != t_.size()) fail("trailing characters");
    return v;
  }

 private:
  const std::string& t_;
  size_t i_ = 0;

  [[noreturn]] void fail(const char* what) {
    throw std::runtime_error(std::string("JSON parse error: ") + what + " at offset " +
                             std::to_string(i_));
  }
  void ws() {
    while (i_ < t_.size() && (t_[i_] == ' ' || t_[i_] == '\n' || t_[i_] == '\r' || t_[i_] == '\t'))
      ++i_;
  }
  char peek() {
    ws();
    if (i_ >= t_.size()) fail("unexpected end");
    return t_[i_];
  }
  void expect(char c) {
    if (peek() != c) fail("unexpected character");
    ++i_;
  }
  bool lit(const char* s) {
    size_t n = std::char_traits<char>::length(s);
    if (t_.compare(i_, n, s) == 0) { i_ += n; return true; }
    return false;
  }

  Json value() {
    char c = peek();
    if (c == '{') return object();
    if (c == '[') return array();
    if (c == '"') return Json(string());
    if (lit("true")) return Json(true);
    if (lit("false")) return Json(false);
    if (lit("null")) return Json();
    if (lit("NaN")) return Json(std::nan(""));
    if (lit("Infinity")) return Json(HUGE_VAL);
    if (lit("-Infinity")) return Json(-HUGE_VAL);
    return number();
  }

  Json number() {
    const char* start = t_.c_str() + i_;
    char* end = nullptr;
    double d = std::strtod(start, &end);
    if (end == start) fail("invalid value");
    i_ += static_cast<size_t>(end - start);
    return Json(d);
  }

  static void utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) out += static_cast<char>(cp);
    else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  uint32_t hex4() {
    if (i_ + 4 > t_.size()) fail("bad \\u escape");
    uint32_t v = static_cast<uint32_t>(std::stoul(t_.substr(i_, 4), nullptr, 16));
    i_ += 4;
    return v;
  }

  std::string string() {
    expect('"');
    std::string out;
    while (true) {
      if (i_ >= t_.size()) fail("unterminated string");
      char c = t_[i_++];
      if (c == '"') break;
      if (c != '\\') { out += c; continue; }
      if (i_ >= t_.size()) fail("bad escape");
      char e = t_[i_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && lit("\\u")) {
            uint32_t lo = hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          utf8(out, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
    return out;
  }

  Json array() {
    expect('[');
    Json j = Json::array();
    if (peek() == ']') { ++i_; return j; }
    while (true) {
      j.a_.push_back(value());
      char c = peek();
      ++i_;
      if (c == ']') break;
      if (c != ',') fail("expected , or ]");
    }
    return j;
  }

  Json object() {
    expect('{');
    Json j = Json::object();
    if (peek() == '}') { ++i_; return j; }
    while (true) {
      if (peek() != '"') fail("expected key");
      std::string k = string();
      expect(':');
      j.o_.emplace_back(std::move(k), value());
      char c = peek();
      ++i_;
      if (c == '}') break;
      if (c != ',') fail("expected , or }");
    }
    return j;
  }
};

Json Json::parse(const std::string& text) { return JsonParser(text).parse_root(); }

bool Json::as_bool() const {
  if (type_ != Type::Bool) throw std::runtime_error("JSON: not a bool");
  return b_;
}
double Json::as_number() const {
  if (type_ == Type::Bool) return b_ ? 1 : 0;
  if (type_ != Type::Number) throw std::runtime_error("JSON: not a number");
  return n_;
}
const std::string& Json::as_string() const {
  if (type_ != Type::String) throw std::runtime_error("JSON: not a string");
  return s_;
}
const std::vector<Json>& Json::items() const {
  if (type_ != Type::Array) throw std::runtime_error("JSON: not an array");
  return a_;
}
const std::vector<std::pair<std::string, Json>>& Json::members() const {
  if (type_ != Type::Object) throw std::runtime_error("JSON: not an object");
  return o_;
}
bool Json::has(const std::string& key) const {
  if (type_ != Type::Object) return false;
  for (auto& kv : o_) if (kv.first == key) return true;
  return false;
}
const Json& Json::operator[](const std::string& key) const {
  for (auto& kv : members()) if (kv.first == key) return kv.second;
  throw std::runtime_error("JSON: missing key '" + key + "'");
}
size_t Json::size() const {
  if (type_ == Type::Array) return a_.size();
  if (type_ == Type::Object) return o_.size();
  return 0;
}
double Json::num_or(const std::string& key, double def) const {
  if (!has(key) || (*this)[key].is_null()) return def;
  return (*this)[key].as_number();
}
std::string Json::str_or(const std::string& key, const std::string& def) const {
  if (!has(key) || (*this)[key].is_null()) return def;
  return (*this)[key].as_string();
}
bool Json::bool_or(const std::string& key, bool def) const {
  if (!has(key) || (*this)[key].is_null()) return def;
  return (*this)[key].as_bool();
}
Json& Json::set(const std::string& key, Json v) {
  if (type_ == Type::Null) type_ = Type::Object;
  for (auto& kv : o_) if (kv.first == key) { kv.second = std::move(v); return *this; }
  o_.emplace_back(key, std::move(v));
  return *this;
}
Json& Json::push(Json v) {
  if (type_ == Type::Null) type_ = Type::Array;
  a_.push_back(std::move(v));
  return *this;
}

static void dump_string(std::string& out, const std::string& s) {
  out += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  out += '"';
}

void Json::dump_to(std::string& out) const {
  switch (type_) {
    case Type::Null: out += "null"; break;
    case Type::Bool: out += b_ ? "true" : "false"; break;
    case Type::Number: {
      if (std::isfinite(n_) && n_ == std::floor(n_) && std::fabs(n_) < 1e15) {
        out += std::to_string(static_cast<long long>(n_));
      } else {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.17g", n_);
        out += buf;
      }
      break;
    }
    case Type::String: dump_string(out, s_); break;
    case Type::Array: {
      out += '[';
      for (size_t i = 0; i < a_.size(); ++i) { if (i) out += ", "; a_[i].dump_to(out); }
      out += ']';
      break;
    }
    case Type::Object: {
      out += '{';
      for (size_t i = 0; i < o_.size(); ++i) {
        if (i) out += ", ";
        dump_string(out, o_[i].first);
        out += ": ";
        o_[i].second.dump_to(out);
      }
      out += '}';
      break;
    }
  }
}

std::string Json::dump() const {
  std::string out;
  dump_to(out);
  return out;
}

}  // namespace mdi
