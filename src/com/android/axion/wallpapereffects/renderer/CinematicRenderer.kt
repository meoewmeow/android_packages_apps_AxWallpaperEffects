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

package com.android.axion.wallpapereffects.renderer

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.drawable.BitmapDrawable
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.util.Log
import com.android.axion.wallpapereffects.R
import com.android.axion.wallpapereffects.util.*
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

private const val TAG = "CinematicRenderer"
private const val DEFAULT_PARALLAX_STRENGTH = 0.085f
private const val DEFAULT_VIGNETTE_STRENGTH = 0.5f
/** 1 while the Spatial subject is drawn as its own moving layer; read by Rime's lock screen. */
private const val SPATIAL_LAYERS_KEY = "rime_spatial_layers"

class CinematicRenderer(private val context: Context) : GLSurfaceView.Renderer {

    var offsetX: Float = 0f
    var offsetY: Float = 0f
    private var spatial = false
    private var spatialStrength = 1f
    @Volatile var needsReload: Boolean = false
    /** Asks the engine for a frame once spatial layers built in the background are ready. */
    var onLayersReady: (() -> Unit)? = null

    private var programId = 0
    private var colorTextureId = 0
    private var depthTextureId = 0
    private var bgTextureId = 0
    private var alphaTextureId = 0
    private var surfaceWidth = 0
    private var surfaceHeight = 0

    private val depthEstimator = DepthEstimator(context)

    private val vertices =
        floatArrayOf(-1f, -1f, 0f, 1f, 1f, -1f, 1f, 1f, -1f, 1f, 0f, 0f, 1f, 1f, 1f, 0f)
    private lateinit var vertexBuffer: FloatBuffer

    fun reloadTexture() {
        needsReload = true
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        vertexBuffer =
            ByteBuffer.allocateDirect(vertices.size * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(vertices)
        vertexBuffer.position(0)

        val vertSource = loadRawResource(context, R.raw.passthrough_vert)
        val fragSource = loadRawResource(context, R.raw.cinematic_frag)
        programId = createProgram(vertSource, fragSource)
        Log.d(TAG, "Program: $programId")

        depthEstimator.init()
        loadAndApplyTextures()
    }

    private fun loadAndApplyTextures() {
        if (colorTextureId != 0) {
            GLES30.glDeleteTextures(2, intArrayOf(colorTextureId, depthTextureId), 0)
            colorTextureId = 0
            depthTextureId = 0
        }
        deleteSpatialTextures()

        // Read once per texture load: the Rime page sends RELOAD_WALLPAPER after changing them.
        spatial = android.provider.Settings.Secure.getInt(context.contentResolver, "ax_spatial_mode", 0) == 1
        spatialStrength = android.provider.Settings.Secure.getFloat(context.contentResolver, "ax_spatial_strength", 0.25f)
            .coerceIn(0.25f, 2f).takeIf { spatial } ?: 1f
        val wallpaper = loadFixedWallpaper()
        Log.d(TAG, "Loaded wallpaper: ${wallpaper.width}x${wallpaper.height}")

        colorTextureId = uploadTexture(wallpaper)
        if (spatial) {
            val photo = File(context.createDeviceProtectedStorageContext().filesDir, "wallpaper.jpg")
            val layers = try { SpatialLayers.cached(context, photo) } catch (t: Throwable) { null }
            // Segmentation and inpainting take seconds and must not run on this GL thread: build
            // them aside, then reload. Until then, or if it fails, the anchored warp stands in.
            if (layers == null && SpatialLayers.shouldBuild(context, photo)) {
                Thread {
                    if (SpatialLayers.build(context, photo)) {
                        needsReload = true
                        onLayersReady?.invoke()
                    }
                }.start()
            }
            if (layers != null) {
                bgTextureId = uploadTexture(layers.background)
                alphaTextureId = uploadTexture(layers.subjectAlpha)
                layers.background.recycle()
                layers.subjectAlpha.recycle()
            }
        }
        // SystemUI moves the clock's cut-out with the subject only while the subject really moves.
        publishLayered(spatial && bgTextureId != 0 && alphaTextureId != 0)

        val rawDepth = depthEstimator.estimateDepth(wallpaper)
        if (rawDepth != null) {
            val depthMap = blurDepthMap(rawDepth, 16)
            rawDepth.recycle()
            depthTextureId = uploadTexture(depthMap)
            depthMap.recycle()
            Log.d(TAG, "Depth texture uploaded (blurred)")
        } else {

            val fallback = createFallbackDepthMap(wallpaper.width, wallpaper.height)
            depthTextureId = uploadTexture(fallback)
            fallback.recycle()
            Log.w(TAG, "Using fallback depth map")
        }

        wallpaper.recycle()
        Log.d(TAG, "Textures ready: color=$colorTextureId depth=$depthTextureId")
    }

    private fun createFallbackDepthMap(width: Int, height: Int): Bitmap {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)
        val cx = width / 2f
        val cy = height / 2f
        val maxDist = Math.sqrt((cx * cx + cy * cy).toDouble()).toFloat()
        for (y in 0 until height) {
            for (x in 0 until width) {
                val dx = x - cx
                val dy = y - cy
                val dist = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()
                val depth = (1f - dist / maxDist).coerceIn(0f, 1f)
                val v = (depth * 255f).toInt()
                pixels[y * width + x] = Color.argb(255, v, v, v)
            }
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height)
        return bmp
    }

    private fun blurDepthMap(src: Bitmap, radius: Int): Bitmap {
        val w = src.width
        val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        val depth = IntArray(w * h)
        for (i in pixels.indices) depth[i] = Color.red(pixels[i])

        val tmp = IntArray(w * h)

        for (y in 0 until h) {
            var sum = 0
            val rowOff = y * w

            for (x in -radius..radius) {
                sum += depth[rowOff + x.coerceIn(0, w - 1)]
            }
            tmp[rowOff] = sum / (2 * radius + 1)
            for (x in 1 until w) {
                sum += depth[rowOff + (x + radius).coerceAtMost(w - 1)]
                sum -= depth[rowOff + (x - radius - 1).coerceAtLeast(0)]
                tmp[rowOff + x] = sum / (2 * radius + 1)
            }
        }

        val out = IntArray(w * h)

        for (x in 0 until w) {
            var sum = 0
            for (y in -radius..radius) {
                sum += tmp[y.coerceIn(0, h - 1) * w + x]
            }
            out[x] = sum / (2 * radius + 1)
            for (y in 1 until h) {
                sum += tmp[(y + radius).coerceAtMost(h - 1) * w + x]
                sum -= tmp[(y - radius - 1).coerceAtLeast(0) * w + x]
                out[y * w + x] = sum / (2 * radius + 1)
            }
        }

        propagateDepthUp(out, w, h)

        for (i in out.indices) {
            val v = out[i].coerceIn(0, 255)
            pixels[i] = Color.argb(255, v, v, v)
        }

        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    private fun propagateDepthUp(depth: IntArray, w: Int, h: Int) {
        for (x in 0 until w) {
            var carry = 0
            for (y in h - 1 downTo 0) {
                val idx = y * w + x
                val d = depth[idx]
                if (d >= carry) {
                    carry = d
                } else if (carry - d < 55) {
                    val boosted = (d * 3 + carry) / 4
                    depth[idx] = boosted.coerceIn(0, 255)
                    carry = boosted
                } else {
                    carry = d
                }
                carry = carry * 97 / 100
            }
        }
    }

    private fun loadFixedWallpaper(): Bitmap {
        try {
            val deCtx = context.createDeviceProtectedStorageContext()
            val file = File(deCtx.filesDir, "wallpaper.jpg")
            if (file.exists()) {
                val bitmap = BitmapFactory.decodeFile(file.absolutePath)
                if (bitmap != null) {
                    Log.d(TAG, "Wallpaper loaded from DE storage")
                    return bitmap
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load from DE storage", e)
        }
        try {
            val wm = android.app.WallpaperManager.getInstance(context)
            val drawable = wm.getDrawable()
            if (drawable is BitmapDrawable && drawable.bitmap != null) {
                Log.d(TAG, "Wallpaper loaded from WallpaperManager")
                return drawable.bitmap.copy(Bitmap.Config.ARGB_8888, false)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load from WallpaperManager", e)
        }
        val fallback = Bitmap.createBitmap(1080, 1920, Bitmap.Config.ARGB_8888)
        fallback.eraseColor(Color.BLUE)
        return fallback
    }

    private fun uploadTexture(bitmap: Bitmap): Int {
        val textureHandle = IntArray(1)
        GLES30.glGenTextures(1, textureHandle, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureHandle[0])
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_S,
            GLES30.GL_CLAMP_TO_EDGE,
        )
        GLES30.glTexParameteri(
            GLES30.GL_TEXTURE_2D,
            GLES30.GL_TEXTURE_WRAP_T,
            GLES30.GL_CLAMP_TO_EDGE,
        )
        GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bitmap, 0)
        return textureHandle[0]
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        surfaceWidth = width
        surfaceHeight = height
        GLES30.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        if (needsReload) {
            needsReload = false
            loadAndApplyTextures()
        }

        if (colorTextureId == 0 || depthTextureId == 0) {
            GLES30.glClearColor(0f, 0f, 0f, 1f)
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
            return
        }

        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glUseProgram(programId)

        GLES30.glUniform2f(GLES30.glGetUniformLocation(programId, "uOffset"), offsetX, offsetY)
        val layered = spatial && bgTextureId != 0 && alphaTextureId != 0
        // Spatial without its layers (still being built, or no subject) shows the photo still and
        // unzoomed, so it matches the clock's depth cut-out; a warp there smeared the subject.
        val still = spatial && !layered
        GLES30.glUniform1f(
            GLES30.glGetUniformLocation(programId, "uParallaxStrength"),
            if (still) 0f else DEFAULT_PARALLAX_STRENGTH * spatialStrength,
        )
        GLES30.glUniform1f(GLES30.glGetUniformLocation(programId, "uZoom"), if (still) 1f else 1.12f)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(programId, "uAnchor"), if (still) 1f else 0f)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(programId, "uSpatial"), if (layered) 1f else 0f)
        if (layered) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, bgTextureId)
            GLES30.glUniform1i(GLES30.glGetUniformLocation(programId, "uTextureBg"), 2)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE3)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, alphaTextureId)
            GLES30.glUniform1i(GLES30.glGetUniformLocation(programId, "uTextureAlpha"), 3)
        }
        GLES30.glUniform1f(
            GLES30.glGetUniformLocation(programId, "uVignetteStrength"),
            DEFAULT_VIGNETTE_STRENGTH,
        )

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, colorTextureId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(programId, "uTextureColor"), 0)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, depthTextureId)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(programId, "uTextureDepth"), 1)

        val aPosLoc = GLES30.glGetAttribLocation(programId, "aPosition")
        val aTexLoc = GLES30.glGetAttribLocation(programId, "aTexCoord")
        drawQuad(aPosLoc, aTexLoc)
    }

    private fun drawQuad(aPosLoc: Int, aTexLoc: Int) {
        vertexBuffer.position(0)
        GLES30.glVertexAttribPointer(aPosLoc, 2, GLES30.GL_FLOAT, false, 4 * 4, vertexBuffer)
        GLES30.glEnableVertexAttribArray(aPosLoc)
        vertexBuffer.position(2)
        GLES30.glVertexAttribPointer(aTexLoc, 2, GLES30.GL_FLOAT, false, 4 * 4, vertexBuffer)
        GLES30.glEnableVertexAttribArray(aTexLoc)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        GLES30.glDisableVertexAttribArray(aPosLoc)
        GLES30.glDisableVertexAttribArray(aTexLoc)
    }

    private fun publishLayered(layered: Boolean) {
        try {
            android.provider.Settings.Secure.putInt(context.contentResolver, SPATIAL_LAYERS_KEY, if (layered) 1 else 0)
        } catch (t: Throwable) {
            Log.w(TAG, "Could not publish spatial layer state", t)
        }
    }

    private fun deleteSpatialTextures() {
        if (bgTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(bgTextureId), 0)
        if (alphaTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(alphaTextureId), 0)
        bgTextureId = 0
        alphaTextureId = 0
    }

    fun cleanup() {
        deleteSpatialTextures()
        if (colorTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(colorTextureId), 0)
        if (depthTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(depthTextureId), 0)
        colorTextureId = 0
        depthTextureId = 0
        if (programId != 0) {
            GLES30.glDeleteProgram(programId)
            programId = 0
        }
        depthEstimator.release()
    }
}
