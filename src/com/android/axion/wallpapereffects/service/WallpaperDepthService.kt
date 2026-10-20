/*
 * Copyright (C) 2025-2026 AxionOS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.android.axion.wallpapereffects.service

import android.app.Service
import android.app.WallpaperManager
import android.content.Intent
import android.content.res.Configuration
import android.database.ContentObserver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.graphics.drawable.BitmapDrawable
import android.net.Uri
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.provider.Settings
import android.util.Log
import com.android.axion.util.DisplayUtils
import com.android.axion.util.DisplayUtils.DisplayLayout
import com.android.axion.util.DisplayUtils.DisplayLayoutTarget
import com.android.axion.wallpapereffects.util.DepthMaskUtils
import com.android.axion.wallpapereffects.util.PortraitSegmenter
import java.io.File
import kotlin.math.roundToInt

private const val TAG = "WallpaperDepthService"
private const val SETTING_DEPTH_MASK = "ax_depth_subject_mask"
private const val SETTING_DEPTH_BOUNDS = "ax_depth_subject_bounds"
private const val SETTING_DEPTH_ENABLED = "ax_depth_clock_enabled"
private const val DE_PHOTO_FILE = "wallpaper.jpg"
/** Per-preset outline tunables written by the Rime lock screen page; see DepthMaskUtils.Options. */
private const val SETTING_DEPTH_OPTIONS = "ax_depth_options"

/** Preview protocol for the Rime page (same uid): request what=1 with "photo", "options", "request". */
const val MSG_PREVIEW_MASK = 1

class WallpaperDepthService : Service() {

    private val handler = Handler(Looper.getMainLooper())
    private var processingThread: Thread? = null
    @Volatile
    private var currentProcessToken: ProcessToken? = null
    private var lastScreenW = 0
    private var lastScreenH = 0

    private val colorsListener =
        WallpaperManager.OnColorsChangedListener { _, which ->
            if (which and (WallpaperManager.FLAG_SYSTEM or WallpaperManager.FLAG_LOCK) != 0) {
                Log.d(TAG, "Wallpaper colors changed (which=$which), scheduling depth processing")
                scheduleProcess()
            }
        }

    private val enabledObserver =
        object : ContentObserver(handler) {
            override fun onChange(selfChange: Boolean, uri: Uri?) {
                val enabled = Settings.Secure.getInt(contentResolver, SETTING_DEPTH_ENABLED, 0)
                Log.d(TAG, "Depth enabled setting changed: $enabled")
                scheduleProcess()
            }
        }

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "onCreate")
        lastScreenW = resources.displayMetrics.widthPixels
        lastScreenH = resources.displayMetrics.heightPixels
        val wm = WallpaperManager.getInstance(this)
        wm.addOnColorsChangedListener(colorsListener, handler)
        contentResolver.registerContentObserver(
            Settings.Secure.getUriFor(SETTING_DEPTH_ENABLED),
            false,
            enabledObserver,
        )
        contentResolver.registerContentObserver(
            Settings.Secure.getUriFor(SETTING_DEPTH_OPTIONS),
            false,
            enabledObserver,
        )

        scheduleProcess()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        val w = resources.displayMetrics.widthPixels
        val h = resources.displayMetrics.heightPixels
        if (w != lastScreenW || h != lastScreenH) {
            Log.d(TAG, "Screen dimensions changed ${lastScreenW}x${lastScreenH} -> ${w}x${h}")
            lastScreenW = w
            lastScreenH = h
            scheduleProcess()
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "onStartCommand")
        scheduleProcess()
        return START_STICKY
    }

    private val previewThread by lazy { android.os.HandlerThread("DepthPreview").apply { start() } }
    @Volatile private var latestPreview = 0
    private var previewPhoto: String? = null
    private var previewForeground: Bitmap? = null

    /**
     * The Rime page asks for a preset photo's subject outline while editing, before anything is
     * applied. Only the latest request is answered; the segmentation of the last photo is kept so
     * that changing an outline option only redoes the cheap extraction.
     */
    private val previewMessenger by lazy {
        android.os.Messenger(object : Handler(previewThread.looper) {
            override fun handleMessage(msg: android.os.Message) {
                if (msg.what != MSG_PREVIEW_MASK) return
                val request = msg.data.getInt("request")
                val replyTo = msg.replyTo ?: return
                latestPreview = request
                // A newer request is already queued behind this one; let it answer.
                if (hasMessages(MSG_PREVIEW_MASK)) return
                // Throwable: an out-of-memory here must not take down this persistent process.
                val mask = try {
                    previewMask(msg.data.getString("photo"), msg.data.getString("options"))
                } catch (t: Throwable) {
                    Log.e(TAG, "Preview mask failed", t)
                    previewForeground?.recycle(); previewForeground = null; previewPhoto = null
                    null
                }
                val reply = android.os.Message.obtain(null, MSG_PREVIEW_MASK).apply {
                    data = android.os.Bundle().apply { putInt("request", request); putString("mask", mask) }
                }
                try { replyTo.send(reply) } catch (_: android.os.RemoteException) {}
            }
        })
    }

    private fun previewMask(photo: String?, options: String?): String? {
        photo ?: return null
        // Keyed by modification time too: a replaced photo keeps its file name.
        val best = options?.contains("quality=best") == true
        val key = "$photo@${File(photo).lastModified()}@$best"
        if (key != previewPhoto) {
            previewForeground?.recycle()
            previewForeground = null
            previewPhoto = null
            // The outline is normalised and extracted 512 wide: a ~1024 px photo loses nothing.
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(photo, bounds)
            var sample = 1
            while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= 1024) sample *= 2
            val bitmap = BitmapFactory.decodeFile(photo, BitmapFactory.Options().apply { inSampleSize = sample }) ?: return null
            val segmenter = PortraitSegmenter(this)
            try {
                segmenter.init()
                previewForeground = segmenter.segment(bitmap, best)
            } finally {
                segmenter.release()
                bitmap.recycle()
            }
            previewPhoto = key
        }
        val foreground = previewForeground ?: return null
        return DepthMaskUtils.extractSubjectPath(foreground, DepthMaskUtils.Options.parse(options))
    }

    override fun onBind(intent: Intent?): IBinder? = previewMessenger.binder

    override fun onUnbind(intent: Intent?): Boolean {
        // The page closed; the cached segmentation is not worth keeping in a persistent process.
        Handler(previewThread.looper).post { previewForeground?.recycle(); previewForeground = null; previewPhoto = null }
        return false
    }

    override fun onDestroy() {
        Log.d(TAG, "onDestroy")
        WallpaperManager.getInstance(this).removeOnColorsChangedListener(colorsListener)
        contentResolver.unregisterContentObserver(enabledObserver)
        currentProcessToken = null
        handler.removeCallbacks(startProcess)
        processingThread?.interrupt()
        previewThread.quitSafely()
        super.onDestroy()
    }

    private val startProcess = Runnable { processNow() }

    /**
     * Coalesced: one wallpaper apply fires the colours listener for home and lock and the settings
     * observers, and each full segmentation needs a lot of memory. Interrupting does not stop a
     * running model, so without this several of them would run at once.
     */
    private fun scheduleProcess() {
        handler.removeCallbacks(startProcess)
        handler.postDelayed(startProcess, 500L)
    }

    private fun processNow() {
        val token = ProcessToken()
        currentProcessToken = token
        processingThread?.interrupt()
        val thread =
            Thread {
                try {
                    processWallpaper(token)
                } catch (e: InterruptedException) {
                    Log.d(TAG, "Processing interrupted")
                } catch (e: Throwable) {
                    // Throwable: an out-of-memory must not take down this persistent process.
                    Log.e(TAG, "Failed to process wallpaper depth", e)
                    if (isCurrentProcess(token)) {
                        clearAllMasks()
                    }
                }
            }
        processingThread = thread
        thread.start()
    }

    private fun processWallpaper(token: ProcessToken) {
        val enabled = Settings.Secure.getInt(contentResolver, SETTING_DEPTH_ENABLED, 0)
        if (enabled != 1) {
            Log.d(TAG, "Depth clock disabled (enabled=$enabled), clearing stale mask")
            if (isCurrentProcess(token)) {
                clearAllMasks()
            }
            return
        }

        val wm = WallpaperManager.getInstance(this)
        val info = wm.wallpaperInfo
        val isLiveWallpaper = info != null
        val isEffectsWallpaper = info?.component?.packageName == packageName
        val hasLockWallpaper = hasLockWallpaper(wm)
        Log.d(TAG, "wallpaperInfo=${info?.component}, isLive=$isLiveWallpaper")

        if (
            info != null &&
                info.component.packageName == packageName &&
                info.component.className.endsWith("MagicPortraitService") &&
                !hasLockWallpaper
        ) {
            Log.d(TAG, "MagicPortraitService active (manages own depth mask), skipping")
            return
        }

        val targets = DisplayUtils.getDisplayLayoutTargets(this)
        if (isCurrentProcess(token)) clearAllMasks()

        val wallpaper =
            loadWallpaperBitmap(wm, isLiveWallpaper, isEffectsWallpaper, hasLockWallpaper)
                ?: run {
                    Log.w(TAG, "Could not load wallpaper bitmap")
                    return
                }
        val bitmap = wallpaper.bitmap
        Log.d(TAG, "Loaded wallpaper bitmap: ${bitmap.width}x${bitmap.height}")

        val segmenter = PortraitSegmenter(this)
        segmenter.init()
        Log.d(TAG, "Segmenter initialized, running segmentation...")

        try {
            val fg = segmenter.segment(bitmap)
            try {
                Log.d(
                    TAG,
                    "Segmentation result: " +
                        if (fg == null) "fg=null" else "fg=${fg.width}x${fg.height}",
                )
                if (fg != null) {
                    val cropRects = resolveCropRects(wm, wallpaper, targets)
                    val paths =
                        targets.mapIndexed { index, target ->
                            target to extractPath(fg, cropRects[index])
                        }
                    if (isCurrentProcess(token)) {
                        publishPaths(paths)
                    } else {
                        Log.d(TAG, "Skipping stale depth result")
                    }
                } else {
                    Log.d(TAG, "No subject detected in wallpaper")
                }
            } finally {
                fg?.recycle()
            }
        } finally {
            segmenter.release()
            if (!bitmap.isRecycled) bitmap.recycle()
        }
    }

    private fun isCurrentProcess(token: ProcessToken): Boolean {
        return token == currentProcessToken && !Thread.currentThread().isInterrupted
    }

    private fun loadWallpaperBitmap(
        wm: WallpaperManager,
        isLiveWallpaper: Boolean,
        isEffectsWallpaper: Boolean,
        hasLockWallpaper: Boolean,
    ): LoadedWallpaper? {
        if (hasLockWallpaper) {
            loadCroppedWallpaperFile(wm, WallpaperManager.FLAG_LOCK)?.let { return it }
            return null
        }

        if (isLiveWallpaper && isEffectsWallpaper) {
            val deBitmap = loadFromDeStorage()
            if (deBitmap != null) return LoadedWallpaper(deBitmap, null)
        }

        if (!isLiveWallpaper) {
            loadCroppedWallpaperFile(wm, WallpaperManager.FLAG_SYSTEM)?.let { return it }
        }

        try {
            val drawable = wm.getDrawable()
            Log.d(TAG, "WallpaperManager drawable: ${drawable?.javaClass?.simpleName}")
            if (drawable is BitmapDrawable && drawable.bitmap != null) {
                val bmp = drawable.bitmap.copy(Bitmap.Config.ARGB_8888, false)
                Log.d(
                    TAG,
                    "WM bitmap: ${if (bmp != null) "${bmp.width}x${bmp.height}" else "null"}",
                )
                return LoadedWallpaper(bmp, null)
            }
        } catch (e: Exception) {
            Log.w(TAG, "WallpaperManager load failed", e)
        }

        if (!isLiveWallpaper) {
            val deBitmap = loadFromDeStorage()
            if (deBitmap != null) return LoadedWallpaper(deBitmap, null)
        }

        return null
    }

    private fun loadCroppedWallpaperFile(
        wm: WallpaperManager,
        which: Int,
    ): LoadedWallpaper? {
        return try {
            wm.getWallpaperFile(which, true)?.use { pfd ->
                BitmapFactory.decodeFileDescriptor(pfd.fileDescriptor)?.let { bitmap ->
                    Log.d(TAG, "Wallpaper crop file ($which): ${bitmap.width}x${bitmap.height}")
                    LoadedWallpaper(bitmap, which)
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Wallpaper file load failed ($which)", e)
            null
        }
    }

    private fun hasLockWallpaper(wm: WallpaperManager): Boolean {
        return try {
            wm.getWallpaperFile(WallpaperManager.FLAG_LOCK)?.use { true } == true
        } catch (_: Exception) {
            false
        }
    }

    private fun loadFromDeStorage(): Bitmap? {
        return try {
            val deCtx = createDeviceProtectedStorageContext()
            val file = File(deCtx.filesDir, DE_PHOTO_FILE)
            if (file.exists()) {
                val bmp = BitmapFactory.decodeFile(file.absolutePath)
                if (bmp != null) {
                    Log.d(TAG, "DE storage bitmap: ${bmp.width}x${bmp.height}")
                }
                bmp
            } else {
                Log.d(TAG, "DE storage $DE_PHOTO_FILE not found")
                null
            }
        } catch (e: Exception) {
            Log.w(TAG, "DE storage load failed", e)
            null
        }
    }

    private fun resolveCropRects(
        wm: WallpaperManager,
        wallpaper: LoadedWallpaper,
        targets: List<DisplayLayoutTarget>,
    ): List<Rect> {
        return targets.map { target ->
            val storedHint =
                wallpaper.which?.let { which ->
                    try {
                        wm.getBitmapCrops(listOf(target.displaySize), which, false)
                            .takeIf { it.isNotEmpty() }?.first()
                    } catch (_: Exception) {
                        null
                    }
                }
            val matchedCrop =
                centerCropRect(
                    wallpaper.bitmap.width,
                    wallpaper.bitmap.height,
                    target.displaySize.x,
                    target.displaySize.y,
                )
            if (storedHint == null) {
                return@map matchedCrop
            }
            val clamped = clampCrop(storedHint, wallpaper.bitmap)
            val visibleH = matchedCrop.height().coerceAtMost(clamped.height())
            val cropTop = clamped.top.coerceAtMost(wallpaper.bitmap.height - visibleH)
            val cropBottom = (cropTop + visibleH).coerceAtMost(wallpaper.bitmap.height)
            Rect(
                clamped.left,
                cropTop,
                clamped.right,
                cropBottom,
            ).also { r -> if (r.width() < 1 || r.height() < 1) return@map matchedCrop }
        }
    }

    private fun extractPath(foreground: Bitmap, cropRect: Rect): String? {
        val cropped =
            if (
                cropRect.left == 0 &&
                    cropRect.top == 0 &&
                    cropRect.right == foreground.width &&
                    cropRect.bottom == foreground.height
            ) {
                foreground
            } else {
                Bitmap.createBitmap(
                    foreground,
                    cropRect.left,
                    cropRect.top,
                    cropRect.width(),
                    cropRect.height(),
                )
            }
        return try {
            // Taps are stored over the whole photo; this extraction sees only the crop.
            DepthMaskUtils.extractSubjectPath(cropped,
                DepthMaskUtils.Options.parse(Settings.Secure.getString(contentResolver, SETTING_DEPTH_OPTIONS))
                    .inCrop(cropRect, foreground.width, foreground.height))
        } finally {
            if (cropped !== foreground) cropped.recycle()
        }
    }

    private fun publishPaths(
        paths: List<Pair<DisplayLayoutTarget, String?>>,
    ) {
        paths.forEach { (target, path) ->
            val maskSetting = target.layout.getSettingName(SETTING_DEPTH_MASK)
            val boundsSetting = target.layout.getSettingName(SETTING_DEPTH_BOUNDS)
            Settings.Secure.putString(contentResolver, maskSetting, path)
            Settings.Secure.putString(contentResolver, boundsSetting, null)
            Log.d(
                TAG,
                "Published ${target.layout} depth path " +
                    if (path == null) "without a subject" else "(${path.length} chars)",
            )
        }
    }

    private fun centerCropRect(srcW: Int, srcH: Int, dstW: Int, dstH: Int): Rect {
        if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0) {
            return Rect(0, 0, srcW, srcH)
        }
        val srcAspect = srcW.toFloat() / srcH
        val dstAspect = dstW.toFloat() / dstH
        return if (srcAspect > dstAspect) {
            val cropW = (srcH * dstAspect).roundToInt().coerceAtMost(srcW)
            val left = (srcW - cropW) / 2
            Rect(left, 0, left + cropW, srcH)
        } else {
            val cropH = (srcW / dstAspect).roundToInt().coerceAtMost(srcH)
            val top = (srcH - cropH) / 2
            Rect(0, top, srcW, top + cropH)
        }
    }

    private fun clampCrop(crop: Rect, bitmap: Bitmap): Rect {
        val clamped =
            Rect(
                crop.left.coerceIn(0, bitmap.width),
                crop.top.coerceIn(0, bitmap.height),
                crop.right.coerceIn(0, bitmap.width),
                crop.bottom.coerceIn(0, bitmap.height),
            )
        return if (clamped.width() > 0 && clamped.height() > 0) {
            clamped
        } else {
            Rect(0, 0, bitmap.width, bitmap.height)
        }
    }

    private fun clearAllMasks() {
        DisplayLayout.values().forEach {
            clearMask(it.getSettingName(SETTING_DEPTH_MASK))
            clearMask(it.getSettingName(SETTING_DEPTH_BOUNDS))
        }
    }

    private fun clearMask(setting: String) {
        try {
            Settings.Secure.putString(contentResolver, setting, null)
        } catch (_: Exception) {}
    }

    private data class LoadedWallpaper(
        val bitmap: Bitmap,
        val which: Int?,
    )

    private data class ProcessToken(val marker: Any = Any())
}
