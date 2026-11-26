package com.mdillm.node

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.text.InputType
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import org.json.JSONObject
import kotlin.concurrent.thread

/**
 * Single screen: configure this phone as secondary node *i* of the ring, start/stop the
 * node service, and watch its status and log.
 */
class MainActivity : Activity() {
    private lateinit var addr: EditText
    private lateinit var index: EditText
    private lateinit var httpPort: EditText
    private lateinit var portIn: EditText
    private lateinit var portOut: EditText
    private lateinit var starterAddr: EditText
    private lateinit var threads: EditText
    private lateinit var dtype: Spinner
    private lateinit var useChunk: CheckBox
    private lateinit var chunkInfo: TextView
    private lateinit var ipHint: TextView
    private lateinit var startStop: Button
    private lateinit var status: TextView
    private lateinit var logView: TextView
    private lateinit var logScroll: ScrollView

    private val ui = Handler(Looper.getMainLooper())
    private var logSeq = 0L
    private var logHandle = 0L
    private var nodesConfig: String? = null

    private val dtypes = listOf("auto", "float32", "float16", "bfloat16")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = "MDI-LLM node"
        buildUi()
        loadSettings(NodeSettings.load(this))
        nodesConfig = getSharedPreferences("node", MODE_PRIVATE).getString("nodesConfig", null)
        refreshChunkInfo()
        if (Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 0)
        }
    }

    override fun onResume() {
        super.onResume()
        ui.post(poll)
    }

    override fun onPause() {
        super.onPause()
        ui.removeCallbacks(poll)
    }

    // ---- UI ------------------------------------------------------------------------

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun label(text: String, bold: Boolean = false) = TextView(this).apply {
        this.text = text
        if (bold) {
            setTypeface(typeface, Typeface.BOLD)
            setPadding(0, dp(16), 0, dp(4))
        }
    }

    private fun field(hint: String, number: Boolean = false) = EditText(this).apply {
        this.hint = hint
        isSingleLine = true
        inputType = if (number) InputType.TYPE_CLASS_NUMBER else InputType.TYPE_CLASS_TEXT
    }

    private fun row(vararg pairs: Pair<String, View>) = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        for ((name, v) in pairs) {
            addView(LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
                addView(label(name))
                addView(v, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
            }, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f).apply { marginEnd = dp(8) })
        }
    }

    private fun button(text: String, onClick: () -> Unit) = Button(this).apply {
        this.text = text
        isAllCaps = false
        setOnClickListener { onClick() }
    }

    private fun buildUi() {
        addr = field("this phone's Wi-Fi IP")
        index = field("0", number = true)
        httpPort = field("8089", number = true)
        portIn = field("5090", number = true)
        portOut = field("5091", number = true)
        starterAddr = field("laptop IP")
        threads = field("0 = all cores", number = true)
        dtype = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, dtypes)
        }
        useChunk = CheckBox(this).apply { text = "Use the local chunk instead of receiving weights" }
        chunkInfo = label("")
        ipHint = label("")
        status = TextView(this).apply { typeface = Typeface.MONOSPACE }
        logView = TextView(this).apply {
            typeface = Typeface.MONOSPACE
            textSize = 11f
            setTextIsSelectable(true)
        }
        startStop = button("Start node") { toggle() }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(8), dp(16), dp(16))

            addView(label("Ring configuration", bold = true))
            addView(label("Import the same nodes JSON the laptop uses, then pick this phone's secondary index."))
            addView(LinearLayout(context).apply {
                addView(button("Import nodes config…") { pickConfig() })
                addView(button("Apply index") { applyIndex() })
            })
            addView(row("Secondary index" to index, "This phone's IP" to addr))
            addView(ipHint)
            addView(button("Use detected IP") {
                NodeSettings.localIpv4().firstOrNull()?.let { addr.setText(it.second) }
            })
            addView(row("HTTP port" to httpPort, "Input port" to portIn, "Output port" to portOut))
            addView(row("Starter (laptop) IP" to starterAddr))

            addView(label("Performance", bold = true))
            addView(row("Threads" to threads, "Weights stored as" to dtype))
            addView(label("auto keeps the dtype the starter sends (bfloat16 for TinyLlama: half the memory of float32)."))

            addView(label("Local weights (optional)", bold = true))
            addView(label("A .safetensors chunk from scripts/export_chunks.py. It is memory-mapped, so it skips the Wi-Fi transfer and needs less RAM."))
            addView(LinearLayout(context).apply {
                addView(button("Import chunk…") { pickChunk() })
                addView(button("Delete chunk") {
                    NodeService.chunkFile(this@MainActivity).delete()
                    refreshChunkInfo()
                })
            })
            addView(chunkInfo)
            addView(useChunk)

            addView(label("Node", bold = true))
            addView(startStop, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
            addView(button("Battery optimization settings") {
                startActivity(Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS))
            })
            addView(status)
            addView(label("Log", bold = true))
            addView(logView)
        }
        logScroll = ScrollView(this).apply { addView(root) }
        setContentView(logScroll)
    }

    private fun loadSettings(s: NodeSettings) {
        addr.setText(s.addr)
        index.setText(s.index.toString())
        httpPort.setText(s.httpPort.toString())
        portIn.setText(s.portIn.toString())
        portOut.setText(s.portOut.toString())
        starterAddr.setText(s.starterAddr)
        threads.setText(s.threads.toString())
        dtype.setSelection(dtypes.indexOf(s.weightDtype).coerceAtLeast(0))
        useChunk.isChecked = s.useChunk
    }

    private fun readSettings() = NodeSettings(
        addr = addr.text.toString().trim(),
        httpPort = httpPort.text.toString().toIntOrNull() ?: 8089,
        portIn = portIn.text.toString().toIntOrNull() ?: 5090,
        portOut = portOut.text.toString().toIntOrNull() ?: 5091,
        starterAddr = starterAddr.text.toString().trim(),
        index = index.text.toString().toIntOrNull() ?: 0,
        threads = threads.text.toString().toIntOrNull() ?: 0,
        weightDtype = dtypes[dtype.selectedItemPosition],
        useChunk = useChunk.isChecked,
    )

    private fun setEditable(enabled: Boolean) {
        for (v in listOf<View>(addr, index, httpPort, portIn, portOut, starterAddr, threads, dtype, useChunk))
            v.isEnabled = enabled
    }

    // ---- Actions ---------------------------------------------------------------------

    private fun toggle() {
        if (NodeService.handle != 0L) {
            startService(Intent(this, NodeService::class.java).setAction(NodeService.ACTION_STOP))
            return
        }
        val s = readSettings()
        if (s.addr.isEmpty()) {
            toast("Set this phone's IP (it must match its entry in the nodes config)")
            return
        }
        s.save(this)
        NodeService.lastError = null
        val i = s.toIntent(Intent(this, NodeService::class.java).setAction(NodeService.ACTION_START))
        startForegroundService(i)
    }

    private fun pickConfig() {
        startActivityForResult(
            Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*"),
            REQ_CONFIG,
        )
    }

    private fun pickChunk() {
        startActivityForResult(
            Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*"),
            REQ_CHUNK,
        )
    }

    private fun applyIndex() {
        val cfg = nodesConfig ?: return toast("Import a nodes config first")
        try {
            loadSettings(NodeSettings.fromNodesConfig(cfg, index.text.toString().toIntOrNull() ?: 0, readSettings()))
        } catch (e: Exception) {
            toast(e.message ?: "Invalid config")
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val uri = data?.data
        if (resultCode != RESULT_OK || uri == null) return
        when (requestCode) {
            REQ_CONFIG -> importConfig(uri)
            REQ_CHUNK -> importChunk(uri)
        }
    }

    private fun importConfig(uri: Uri) {
        try {
            val text = contentResolver.openInputStream(uri)!!.use { it.readBytes().decodeToString() }
            val n = NodeSettings.secondaryCount(text)
            nodesConfig = text
            getSharedPreferences("node", MODE_PRIVATE).edit().putString("nodesConfig", text).apply()
            // Pick the entry whose address is one of this phone's IPs, if any
            val mine = NodeSettings.localIpv4().map { it.second }.toSet()
            val match = (0 until n).firstOrNull {
                NodeSettings.fromNodesConfig(text, it, readSettings()).addr in mine
            }
            if (match != null) index.setText(match.toString())
            applyIndex()
            toast("Config with $n secondary nodes" + (match?.let { "; this phone is secondary:$it" } ?: ""))
        } catch (e: Exception) {
            toast("Cannot read config: ${e.message}")
        }
    }

    private fun importChunk(uri: Uri) {
        chunkInfo.text = "Copying chunk…"
        thread {
            val dst = NodeService.chunkFile(this)
            val tmp = java.io.File(dst.path + ".part")
            val result = try {
                contentResolver.openInputStream(uri)!!.use { input ->
                    tmp.outputStream().use { input.copyTo(it, 1 shl 20) }
                }
                if (!tmp.renameTo(dst)) throw IllegalStateException("rename failed")
                null
            } catch (e: Exception) {
                tmp.delete()
                e.message ?: e.toString()
            }
            ui.post {
                refreshChunkInfo()
                if (result != null) toast("Import failed: $result") else useChunk.isChecked = true
            }
        }
    }

    private fun refreshChunkInfo() {
        val f = NodeService.chunkFile(this)
        chunkInfo.text = if (f.isFile) "Local chunk: %.0f MB".format(f.length() / 1048576.0) else "No local chunk"
        useChunk.isEnabled = f.isFile && NodeService.handle == 0L
        if (!f.isFile) useChunk.isChecked = false
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_LONG).show()

    // ---- Status polling ------------------------------------------------------------------

    private val poll = object : Runnable {
        override fun run() {
            updateStatus()
            ui.postDelayed(this, 1000)
        }
    }

    private fun updateStatus() {
        val h = NodeService.handle
        val running = h != 0L
        startStop.text = if (running) "Stop node" else "Start node"
        setEditable(!running)
        if (!running) useChunk.isEnabled = NodeService.chunkFile(this).isFile

        val ips = NodeSettings.localIpv4()
        val typed = addr.text.toString().trim()
        ipHint.text = when {
            ips.isEmpty() -> "No network: join the same Wi-Fi (or hotspot) as the laptop"
            typed.isNotEmpty() && ips.none { it.second == typed } ->
                "⚠ $typed is not an address of this phone (${ips.joinToString { it.second }})"
            else -> "Detected: ${ips.joinToString { "${it.second} (${it.first})" }}"
        }

        if (!running) {
            status.text = NodeService.lastError?.let { "Not running: $it" } ?: "Not running"
            return
        }
        if (h != logHandle) {  // new node instance: restart the log
            logHandle = h
            logSeq = 0
            logView.text = ""
        }
        try {
            val j = JSONObject(NativeNode.nativeStatus(h))
            val sb = StringBuilder()
            sb.append("state     ${j.optString("state")}  (${j.optString("role")})\n")
            sb.append("listening ${j.optString("addr")}:${j.optInt("http_port")}\n")
            if (j.optBoolean("initialized")) {
                sb.append("model     ${j.optInt("n_layers")} layers, %.0f MB weights, %.1f MB KV cache/sample\n"
                    .format(j.optDouble("weights_mb"), j.optDouble("kv_cache_mb_per_sample")))
            }
            sb.append("tokens    ${j.optLong("tokens")} in ${j.optLong("forwards")} passes\n")
            sb.append("latency   last %.1f ms, avg %.1f ms per pass".format(j.optDouble("last_ms"), j.optDouble("avg_ms")))
            status.text = sb.toString()

            val logs = NativeNode.nativeLogs(h, logSeq)
            val nl = logs.indexOf('\n')
            if (nl > 0) {
                logSeq = logs.substring(0, nl).toLongOrNull() ?: logSeq
                val lines = logs.substring(nl + 1)
                if (lines.isNotEmpty()) {
                    logView.append(lines)
                    val text = logView.text
                    if (text.length > 20000) logView.text = text.subSequence(text.length - 15000, text.length)
                }
            }
        } catch (e: Exception) {
            status.text = "Status error: ${e.message}"
        }
    }

    companion object {
        private const val REQ_CONFIG = 1
        private const val REQ_CHUNK = 2
    }
}
