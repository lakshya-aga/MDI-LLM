// JNI bindings for the Android app (android/app/.../NativeNode.kt).
#include <android/log.h>
#include <jni.h>

#include <memory>
#include <string>

#include "node.hpp"

namespace {

std::string str(JNIEnv* env, jstring s) {
  if (!s) return "";
  const char* c = env->GetStringUTFChars(s, nullptr);
  std::string out(c);
  env->ReleaseStringUTFChars(s, c);
  return out;
}

mdi::Node* node_of(jlong h) { return reinterpret_cast<mdi::Node*>(h); }

void throw_java(JNIEnv* env, const char* msg) {
  jclass cls = env->FindClass("java/lang/IllegalStateException");
  if (cls) env->ThrowNew(cls, msg);
}

}  // namespace

extern "C" {

// Returns a handle (>0) or throws IllegalStateException
JNIEXPORT jlong JNICALL Java_com_mdillm_node_NativeNode_nativeStart(
    JNIEnv* env, jclass, jstring addr, jint http_port, jint port_in, jint port_out, jstring starter_addr,
    jstring chunk_path, jint threads, jstring weight_dtype, jint index) {
  mdi::NodeOptions o;
  o.addr = str(env, addr);
  o.http_port = http_port;
  o.port_in = port_in;
  o.port_out = port_out;
  o.starter_addr = str(env, starter_addr);
  o.chunk_path = str(env, chunk_path);
  o.threads = threads;
  o.index = index;
  std::string wdt = str(env, weight_dtype);
  if (!wdt.empty() && wdt != "auto") {
    if (!mdi::dtype_from_cli(wdt, &o.weight_dtype)) {
      throw_java(env, ("unknown weight dtype " + wdt).c_str());
      return 0;
    }
    o.has_weight_dtype = true;
  }
  auto node = std::make_unique<mdi::Node>(o);
  node->set_log_sink([](const std::string& m) { __android_log_print(ANDROID_LOG_INFO, "mdinode", "%s", m.c_str()); });
  try {
    node->start();
  } catch (const std::exception& e) {
    throw_java(env, e.what());
    return 0;
  }
  return reinterpret_cast<jlong>(node.release());
}

JNIEXPORT void JNICALL Java_com_mdillm_node_NativeNode_nativeStop(JNIEnv*, jclass, jlong h) {
  if (h) {
    mdi::Node* n = node_of(h);
    n->stop();
    delete n;
  }
}

JNIEXPORT jstring JNICALL Java_com_mdillm_node_NativeNode_nativeStatus(JNIEnv* env, jclass, jlong h) {
  if (!h) return env->NewStringUTF("{}");
  return env->NewStringUTF(node_of(h)->status_json().c_str());
}

// Returns "<last seq>\n<lines...>" with the log lines newer than `after`
JNIEXPORT jstring JNICALL Java_com_mdillm_node_NativeNode_nativeLogs(JNIEnv* env, jclass, jlong h, jlong after) {
  if (!h) return env->NewStringUTF("0\n");
  std::string lines;
  uint64_t last = node_of(h)->logs_since(static_cast<uint64_t>(after), &lines);
  std::string out = std::to_string(last) + "\n" + lines;
  // NewStringUTF expects modified UTF-8; log lines are ASCII
  return env->NewStringUTF(out.c_str());
}

}  // extern "C"
