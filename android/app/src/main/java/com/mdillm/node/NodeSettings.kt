package com.mdillm.node

import android.content.Context
import android.content.Intent
import org.json.JSONObject
import java.net.Inet4Address
import java.net.NetworkInterface

/** Node settings, persisted in SharedPreferences and passed to [NodeService]. */
data class NodeSettings(
    val addr: String = "",
    val httpPort: Int = 8089,
    val portIn: Int = 5090,
    val portOut: Int = 5091,
    val starterAddr: String = "",
    val index: Int = 0,
    val threads: Int = 0,
    val weightDtype: String = "auto",
    val useChunk: Boolean = false,
) {
    fun save(ctx: Context) {
        prefs(ctx).edit()
            .putString("addr", addr).putInt("httpPort", httpPort).putInt("portIn", portIn)
            .putInt("portOut", portOut).putString("starterAddr", starterAddr).putInt("index", index)
            .putInt("threads", threads).putString("weightDtype", weightDtype)
            .putBoolean("useChunk", useChunk)
            .apply()
    }

    fun toIntent(intent: Intent): Intent = intent
        .putExtra("addr", addr).putExtra("httpPort", httpPort).putExtra("portIn", portIn)
        .putExtra("portOut", portOut).putExtra("starterAddr", starterAddr).putExtra("index", index)
        .putExtra("threads", threads).putExtra("weightDtype", weightDtype)
        .putExtra("useChunk", useChunk)

    companion object {
        private fun prefs(ctx: Context) = ctx.getSharedPreferences("node", Context.MODE_PRIVATE)

        fun load(ctx: Context): NodeSettings {
            val p = prefs(ctx)
            return NodeSettings(
                addr = p.getString("addr", "") ?: "",
                httpPort = p.getInt("httpPort", 8089),
                portIn = p.getInt("portIn", 5090),
                portOut = p.getInt("portOut", 5091),
                starterAddr = p.getString("starterAddr", "") ?: "",
                index = p.getInt("index", 0),
                threads = p.getInt("threads", 0),
                weightDtype = p.getString("weightDtype", "auto") ?: "auto",
                useChunk = p.getBoolean("useChunk", false),
            )
        }

        fun fromIntent(i: Intent) = NodeSettings(
            addr = i.getStringExtra("addr") ?: "",
            httpPort = i.getIntExtra("httpPort", 8089),
            portIn = i.getIntExtra("portIn", 5090),
            portOut = i.getIntExtra("portOut", 5091),
            starterAddr = i.getStringExtra("starterAddr") ?: "",
            index = i.getIntExtra("index", 0),
            threads = i.getIntExtra("threads", 0),
            weightDtype = i.getStringExtra("weightDtype") ?: "auto",
            useChunk = i.getBooleanExtra("useChunk", false),
        )

        /**
         * Reads secondary node [index] from a nodes-config JSON (the same file the
         * Python nodes use, e.g. `src/settings_distr/phones_example.json`).
         */
        fun fromNodesConfig(json: String, index: Int, base: NodeSettings): NodeSettings {
            val root = JSONObject(json)
            val nodes = if (root.has("nodes")) root.getJSONObject("nodes") else root
            val secondary = nodes.getJSONArray("secondary")
            require(index in 0 until secondary.length()) {
                "The config has ${secondary.length()} secondary nodes; index $index is out of range"
            }
            val me = secondary.getJSONObject(index)
            val comm = me.getJSONObject("communication")
            val inf = me.getJSONObject("inference")
            return base.copy(
                addr = me.getString("addr"),
                httpPort = comm.getInt("port"),
                starterAddr = comm.optString("starter_addr", ""),
                portIn = inf.getInt("port_in"),
                portOut = inf.getInt("port_out"),
                index = index,
            )
        }

        /** Number of secondary nodes in a nodes-config JSON. */
        fun secondaryCount(json: String): Int {
            val root = JSONObject(json)
            val nodes = if (root.has("nodes")) root.getJSONObject("nodes") else root
            return nodes.getJSONArray("secondary").length()
        }

        /** IPv4 addresses of this device, Wi-Fi / hotspot interfaces first. */
        fun localIpv4(): List<Pair<String, String>> {
            val out = mutableListOf<Pair<String, String>>()
            try {
                for (ni in NetworkInterface.getNetworkInterfaces()) {
                    if (!ni.isUp || ni.isLoopback) continue
                    for (a in ni.inetAddresses) {
                        if (a is Inet4Address && !a.isLoopbackAddress) out += ni.name to (a.hostAddress ?: continue)
                    }
                }
            } catch (_: Exception) {
            }
            fun rank(name: String) = when {
                name.startsWith("wlan") -> 0
                name.startsWith("ap") || name.startsWith("swlan") -> 1
                name.startsWith("eth") -> 2
                else -> 3
            }
            return out.sortedBy { rank(it.first) }
        }
    }
}
