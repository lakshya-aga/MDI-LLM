package com.mdillm.node

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.wifi.WifiManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import java.io.File

/**
 * Foreground service that owns the native node, so it keeps running with the screen
 * off. Holds a partial wakelock (CPU) and a Wi-Fi lock (no Wi-Fi power saving, which
 * otherwise adds tens of ms of latency per hop).
 */
class NodeService : Service() {
    private var wakeLock: PowerManager.WakeLock? = null
    private var wifiLock: WifiManager.WifiLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                shutdown()
                stopSelf()
            }
            ACTION_START -> start(NodeSettings.fromIntent(intent))
        }
        return START_NOT_STICKY
    }

    private fun start(s: NodeSettings) {
        if (handle != 0L) return
        startInForeground("Starting…")  // must be called promptly after startForegroundService
        val chunk = chunkFile(this)
        val chunkPath = if (s.useChunk && chunk.isFile) chunk.absolutePath else ""
        try {
            handle = NativeNode.nativeStart(
                s.addr, s.httpPort, s.portIn, s.portOut, s.starterAddr, chunkPath,
                s.threads, s.weightDtype, s.index,
            )
            lastError = null
        } catch (e: Exception) {
            Log.e(TAG, "Cannot start node", e)
            lastError = e.message ?: e.toString()
            shutdown()
            stopSelf()
            return
        }
        acquireLocks()
        startInForeground("secondary:${s.index} on ${s.addr}:${s.httpPort}")
    }

    private fun acquireLocks() {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "mdinode:cpu").apply {
            setReferenceCounted(false)
            acquire()
        }
        val wm = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
        @Suppress("DEPRECATION")
        val mode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) WifiManager.WIFI_MODE_FULL_LOW_LATENCY
        else WifiManager.WIFI_MODE_FULL_HIGH_PERF
        wifiLock = wm.createWifiLock(mode, "mdinode:wifi").apply {
            setReferenceCounted(false)
            acquire()
        }
    }

    private fun shutdown() {
        val h = handle
        handle = 0L
        if (h != 0L) NativeNode.nativeStop(h)
        wakeLock?.let { if (it.isHeld) it.release() }
        wifiLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
        wifiLock = null
        stopForeground(STOP_FOREGROUND_REMOVE)
    }

    override fun onDestroy() {
        shutdown()
        super.onDestroy()
    }

    private fun startInForeground(text: String) {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL, "Node", NotificationManager.IMPORTANCE_LOW)
        )
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val stop = PendingIntent.getService(
            this, 1, Intent(this, NodeService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val n: Notification = Notification.Builder(this, CHANNEL)
            .setContentTitle("MDI-LLM node running")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setContentIntent(open)
            .addAction(Notification.Action.Builder(null, "Stop", stop).build())
            .setOngoing(true)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTIFICATION_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(NOTIFICATION_ID, n)
        }
    }

    companion object {
        const val ACTION_START = "com.mdillm.node.START"
        const val ACTION_STOP = "com.mdillm.node.STOP"
        private const val CHANNEL = "node"
        private const val NOTIFICATION_ID = 1
        private const val TAG = "NodeService"

        /** Native handle of the running node (0 when stopped); read by the UI. */
        @Volatile
        var handle: Long = 0L
            private set

        /** Error of the last failed start, shown by the UI. */
        @Volatile
        var lastError: String? = null

        /** Where an imported .safetensors chunk is stored (memory-mapped by the node). */
        fun chunkFile(ctx: Context) = File(ctx.filesDir, "chunk.safetensors")
    }
}
