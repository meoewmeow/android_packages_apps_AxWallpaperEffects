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

import android.app.WallpaperColors
import android.app.wallpaper.WallpaperDescription
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.BitmapFactory
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.SurfaceHolder
import com.android.axion.wallpapereffects.renderer.CinematicRenderer
import com.google.android.torus.core.wallpaper.LiveWallpaper
import com.google.android.torus.core.wallpaper.listener.LiveWallpaperEventListener
import java.io.File

private const val TAG = "CinematicService"
private const val ACTION_RELOAD = "com.android.axion.wallpapereffects.RELOAD_WALLPAPER"
private const val GYRO_SENSITIVITY = 2.4f
private const val GYRO_MAX_OFFSET = 0.45f
private const val GYRO_DECAY = 0.92f
// Keys of Rime's tilt, sent by SystemUI with the lock screen layout command.
private const val EXTRA_SPATIAL_X = "rime_spatial_x"
private const val EXTRA_SPATIAL_Y = "rime_spatial_y"
private const val EXTERNAL_HOLD_MS = 1500L

class CinematicService : LiveWallpaper() {

    override fun getWallpaperEngine(
        context: Context,
        surfaceHolder: SurfaceHolder,
        wallpaperDescription: WallpaperDescription?,
    ) = CinematicEngine(context, surfaceHolder)
}

class CinematicEngine(context: Context, surfaceHolder: SurfaceHolder) :
    GLTorusEngine(context, surfaceHolder), LiveWallpaperEventListener, SensorEventListener {

    private var myRenderer: CinematicRenderer? = null
    private var sensorManager: SensorManager? = null
    private var gyroSensor: Sensor? = null
    private var lastTimestamp = 0L

    private var gyroX = 0f
    private var gyroY = 0f

    private var scrollOffset = 0f
    private var isResumed = false
    /** False between going-to-sleep and waking: AOD can make the wallpaper visible, and resume() must not restart the gyro then. */
    private var isAwake = true
    /**
     * Rime's lock screen drives the tilt itself and sends it here, so the clock's depth cut-out
     * moves with the subject exactly. While it does, this engine's own gyro is ignored.
     */
    private var externalX = 0f
    private var externalY = 0f
    private var externalUntil = 0L

    private val receiver =
        object : BroadcastReceiver() {
            override fun onReceive(ctx: Context, intent: Intent) {
                if (intent.action == ACTION_RELOAD) {
                    myRenderer?.reloadTexture()
                    requestRender()
                }
            }
        }

    override fun create(isFirstActiveInstance: Boolean) {
        myRenderer = CinematicRenderer(context).also { it.onLayersReady = { requestRender() } }
        initGL(myRenderer!!)

        val filter = IntentFilter(ACTION_RELOAD)
        context.registerReceiver(receiver, filter, Context.RECEIVER_EXPORTED)

        sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        gyroSensor = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        Log.d(TAG, "create: gyroSensor=${gyroSensor != null}")

        notifyWallpaperColorsChanged()
    }

    override fun resume() {
        super.resume()
        isResumed = true
        if (isAwake) gyroSensor?.let {
            sensorManager?.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        updateRendererOffset()
        requestRender()
    }

    override fun pause() {
        isResumed = false
        sensorManager?.unregisterListener(this)
        super.pause()
    }

    override fun destroy(isLastActiveInstance: Boolean) {
        sensorManager?.unregisterListener(this)
        try {
            context.unregisterReceiver(receiver)
        } catch (_: Exception) {}
        myRenderer?.cleanup()
        super.destroy(isLastActiveInstance)
    }

    override fun onOffsetChanged(xOffset: Float, xOffsetStep: Float) {
        scrollOffset = xOffset - 0.5f
        updateRendererOffset()
        requestRender()
    }

    override fun onZoomChanged(zoomLevel: Float) {}

    override fun onWallpaperReapplied() {
        myRenderer?.reloadTexture()
        requestRender()
    }

    override fun computeWallpaperColors(): WallpaperColors? {
        return try {
            val deCtx = context.createDeviceProtectedStorageContext()
            val file = File(deCtx.filesDir, "wallpaper.jpg")
            if (file.exists()) {
                val opts = BitmapFactory.Options().apply { inSampleSize = 4 }
                val bitmap = BitmapFactory.decodeFile(file.absolutePath, opts)
                val colors = bitmap?.let { WallpaperColors.fromBitmap(it) }
                bitmap?.recycle()
                colors
            } else null
        } catch (_: Exception) {
            null
        }
    }

    override fun onWake(extras: Bundle) {
        isAwake = true
        if (isResumed) {
            gyroSensor?.let {
                sensorManager?.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
            }
        }
    }

    override fun onSleep(extras: Bundle) {
        isAwake = false
        sensorManager?.unregisterListener(this)
        gyroX = 0f
        gyroY = 0f
        lastTimestamp = 0L
        updateRendererOffset()
        requestRender()
    }

    override fun onLockscreenLayoutChanged(extras: Bundle) {
        if (!extras.containsKey(EXTRA_SPATIAL_X)) return
        externalX = extras.getFloat(EXTRA_SPATIAL_X)
        externalY = extras.getFloat(EXTRA_SPATIAL_Y)
        externalUntil = SystemClock.uptimeMillis() + EXTERNAL_HOLD_MS
        updateRendererOffset()
        requestRender()
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_GYROSCOPE) return
        if (SystemClock.uptimeMillis() < externalUntil) { lastTimestamp = 0L; return }

        val now = event.timestamp
        if (lastTimestamp != 0L) {
            val dt = (now - lastTimestamp) / 1_000_000_000f

            gyroX += event.values[1] * dt * GYRO_SENSITIVITY
            gyroY += event.values[0] * dt * GYRO_SENSITIVITY

            gyroX = gyroX.coerceIn(-GYRO_MAX_OFFSET, GYRO_MAX_OFFSET)
            gyroY = gyroY.coerceIn(-GYRO_MAX_OFFSET, GYRO_MAX_OFFSET)

            gyroX *= GYRO_DECAY
            gyroY *= GYRO_DECAY

            updateRendererOffset()
            requestRender()
        }
        lastTimestamp = now
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun updateRendererOffset() {
        val r = myRenderer ?: return
        if (SystemClock.uptimeMillis() < externalUntil) {
            r.offsetX = externalX
            r.offsetY = externalY
            return
        }
        r.offsetX = gyroX + scrollOffset * 0.25f
        r.offsetY = gyroY
    }
}
