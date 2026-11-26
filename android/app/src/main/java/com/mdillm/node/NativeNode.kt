package com.mdillm.node

/** JNI bindings to the C++ node (`native/src/jni.cpp`). */
object NativeNode {
    init {
        System.loadLibrary("mdinode")
    }

    /** Starts the node's HTTP server; returns a handle. Throws IllegalStateException on failure. */
    @JvmStatic
    external fun nativeStart(
        addr: String,
        httpPort: Int,
        portIn: Int,
        portOut: Int,
        starterAddr: String,
        chunkPath: String,
        threads: Int,
        weightDtype: String,
        index: Int,
    ): Long

    @JvmStatic
    external fun nativeStop(handle: Long)

    /** JSON: state, role, initialized, n_layers, weights_mb, tokens, avg_ms, ... */
    @JvmStatic
    external fun nativeStatus(handle: Long): String

    /** "<last seq>\n<log lines newer than [after]>" */
    @JvmStatic
    external fun nativeLogs(handle: Long, after: Long): String
}
