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

package com.android.axion.wallpapereffects.util

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.util.Log
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate

private const val TAG = "DepthEstimator"
private const val MODEL_FILE = "depth_model.tflite"

class DepthEstimator(private val context: Context) {
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    fun init() {
        try {
            val fd = context.assets.openFd(MODEL_FILE)
            val stream = FileInputStream(fd.fileDescriptor)
            val channel = stream.channel
            val model =
                channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)

            val options = Interpreter.Options().apply { setNumThreads(4) }
            try {
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
                Log.d(TAG, "GPU delegate enabled")
            } catch (e: Exception) {
                Log.w(TAG, "GPU delegate unavailable, using CPU", e)
                gpuDelegate = null
            }

            interpreter = Interpreter(model, options)
            val inputShape = interpreter!!.getInputTensor(0).shape()
            val outputShape = interpreter!!.getOutputTensor(0).shape()
            Log.d(
                TAG,
                "Model loaded — input: ${inputShape.contentToString()}, output: ${outputShape.contentToString()}",
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load depth model", e)
        }
    }

    fun estimateDepth(bitmap: Bitmap): Bitmap? {
        val interp = interpreter ?: return null
        return try {
            val inputShape = interp.getInputTensor(0).shape()
            val inputH = inputShape[1]
            val inputW = inputShape[2]

            val resized = Bitmap.createScaledBitmap(bitmap, inputW, inputH, true)
            val inputBuffer =
                ByteBuffer.allocateDirect(inputH * inputW * 3 * 4).order(ByteOrder.nativeOrder())
            val pixels = IntArray(inputW * inputH)
            resized.getPixels(pixels, 0, inputW, 0, 0, inputW, inputH)
            if (resized !== bitmap) resized.recycle()

            for (pixel in pixels) {
                inputBuffer.putFloat(Color.red(pixel) / 255f)
                inputBuffer.putFloat(Color.green(pixel) / 255f)
                inputBuffer.putFloat(Color.blue(pixel) / 255f)
            }
            inputBuffer.rewind()

            val outputShape = interp.getOutputTensor(0).shape()
            val outputH = outputShape[1]
            val outputW = outputShape[2]
            val outputBuffer =
                ByteBuffer.allocateDirect(outputH * outputW * 4).order(ByteOrder.nativeOrder())

            interp.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()

            val depthValues = FloatArray(outputH * outputW)
            var minDepth = Float.MAX_VALUE
            var maxDepth = Float.NEGATIVE_INFINITY
            for (i in depthValues.indices) {
                val d = outputBuffer.getFloat()
                depthValues[i] = d
                if (d < minDepth) minDepth = d
                if (d > maxDepth) maxDepth = d
            }

            val range = maxDepth - minDepth
            val resultPixels = IntArray(outputH * outputW)
            for (i in depthValues.indices) {
                val normalized = if (range > 0f) (depthValues[i] - minDepth) / range else 0.5f
                val v = (normalized * 255f).toInt().coerceIn(0, 255)
                resultPixels[i] = Color.argb(255, v, v, v)
            }

            val depthBitmap = Bitmap.createBitmap(outputW, outputH, Bitmap.Config.ARGB_8888)
            depthBitmap.setPixels(resultPixels, 0, outputW, 0, 0, outputW, outputH)

            Log.d(TAG, "Depth estimated: ${outputW}x${outputH}, range [$minDepth, $maxDepth]")
            depthBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Depth estimation failed", e)
            null
        }
    }

    fun release() {
        try {
            interpreter?.close()
        } catch (_: Exception) {}
        try {
            gpuDelegate?.close()
        } catch (_: Exception) {}
        interpreter = null
        gpuDelegate = null
    }
}

private const val INPAINT_MODEL_FILE = "migan_inpaint_512_fp16.tflite"
private const val INPAINT_SIZE = 512

/**
 * MI-GAN (Picsart AI Research, ICCV 2023, MIT licence; see LICENSE-MIGAN in this app) fills a
 * hole in a photo. Only the generator is shipped: the crop around the hole, the 512x512 resize
 * and the blend back, which the original ONNX pipeline did, happen here.
 *
 * Input NHWC [1,512,512,4]: mask - 0.5, then RGB scaled to -1..1 and multiplied by the mask
 * (mask 1 = known, 0 = hole). Output NHWC [1,512,512,3] in 0..255.
 */
class Inpainter(private val context: Context) {
    private var interpreter: Interpreter? = null

    fun init(): Boolean {
        if (interpreter != null) return true
        return try {
            val fd = context.assets.openFd(INPAINT_MODEL_FILE)
            val model = FileInputStream(fd.fileDescriptor).channel
                .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            interpreter = Interpreter(model, Interpreter.Options().apply { setNumThreads(4) })
            true
        } catch (e: Exception) {
            Log.e("Inpainter", "Failed to load inpainting model", e)
            false
        }
    }

    /** Returns a copy of [image] with [hole] (true = fill, image-sized) filled, or null on failure. */
    fun fill(image: Bitmap, hole: BooleanArray): Bitmap? {
        val interp = interpreter ?: return null
        val w = image.width
        val h = image.height
        var left = w; var top = h; var right = -1; var bottom = -1
        for (y in 0 until h) for (x in 0 until w) if (hole[y * w + x]) {
            if (x < left) left = x; if (x > right) right = x
            if (y < top) top = y; if (y > bottom) bottom = y
        }
        val result = image.copy(Bitmap.Config.ARGB_8888, true)
        if (right < 0) return result
        if (minOf(w, h) < 64) return null
        // A crop around the whole hole with context on every side, resized to the model's 512x512
        // as the original pipeline did. It must cover the entire hole: anything left outside it
        // stays in the background and shows as a second copy of the subject when it moves.
        val marginX = maxOf((right - left) / 8, 24)
        val marginY = maxOf((bottom - top) / 8, 24)
        val cropLeft = (left - marginX).coerceAtLeast(0)
        val cropTop = (top - marginY).coerceAtLeast(0)
        val cropW = (right + marginX).coerceAtMost(w - 1) - cropLeft + 1
        val cropH = (bottom + marginY).coerceAtMost(h - 1) - cropTop + 1
        return try {
            val crop = Bitmap.createBitmap(image, cropLeft, cropTop, cropW, cropH)
            val scaled = Bitmap.createScaledBitmap(crop, INPAINT_SIZE, INPAINT_SIZE, true)
            val pixels = IntArray(INPAINT_SIZE * INPAINT_SIZE)
            scaled.getPixels(pixels, 0, INPAINT_SIZE, 0, 0, INPAINT_SIZE, INPAINT_SIZE)
            if (scaled !== crop) scaled.recycle()
            crop.recycle()
            val input = ByteBuffer.allocateDirect(INPAINT_SIZE * INPAINT_SIZE * 4 * 4).order(ByteOrder.nativeOrder())
            val scaleX = cropW.toFloat() / INPAINT_SIZE
            val scaleY = cropH.toFloat() / INPAINT_SIZE
            for (y in 0 until INPAINT_SIZE) for (x in 0 until INPAINT_SIZE) {
                val sx = (cropLeft + (x * scaleX).toInt()).coerceAtMost(w - 1)
                val sy = (cropTop + (y * scaleY).toInt()).coerceAtMost(h - 1)
                val known = if (hole[sy * w + sx]) 0f else 1f
                val p = pixels[y * INPAINT_SIZE + x]
                input.putFloat(known - 0.5f)
                input.putFloat((Color.red(p) / 255f * 2f - 1f) * known)
                input.putFloat((Color.green(p) / 255f * 2f - 1f) * known)
                input.putFloat((Color.blue(p) / 255f * 2f - 1f) * known)
            }
            input.rewind()
            val output = ByteBuffer.allocateDirect(INPAINT_SIZE * INPAINT_SIZE * 3 * 4).order(ByteOrder.nativeOrder())
            interp.run(input, output)
            output.rewind()
            val filled = IntArray(INPAINT_SIZE * INPAINT_SIZE) {
                val r = output.getFloat().toInt().coerceIn(0, 255)
                val g = output.getFloat().toInt().coerceIn(0, 255)
                val b = output.getFloat().toInt().coerceIn(0, 255)
                Color.argb(255, r, g, b)
            }
            val patch = Bitmap.createBitmap(filled, INPAINT_SIZE, INPAINT_SIZE, Bitmap.Config.ARGB_8888)
            val back = Bitmap.createScaledBitmap(patch, cropW, cropH, true)
            if (back !== patch) patch.recycle()
            // Only hole pixels change; everything known keeps the photo's full resolution.
            val row = IntArray(cropW)
            val target = IntArray(cropW)
            for (y in 0 until cropH) {
                back.getPixels(row, 0, cropW, 0, y, cropW, 1)
                result.getPixels(target, 0, cropW, cropLeft, cropTop + y, cropW, 1)
                for (x in 0 until cropW) if (hole[(cropTop + y) * w + cropLeft + x]) target[x] = row[x]
                result.setPixels(target, 0, cropW, cropLeft, cropTop + y, cropW, 1)
            }
            back.recycle()
            result
        } catch (e: Exception) {
            Log.e("Inpainter", "Inpainting failed", e)
            result.recycle()
            null
        }
    }

    fun release() {
        try { interpreter?.close() } catch (_: Exception) {}
        interpreter = null
    }
}

/**
 * The two layers of a spatial scene: the photo with its subject filled in (the background that
 * moves), and the subject's soft alpha (the part that stays still). Cached per photo, because
 * segmentation and inpainting take seconds.
 */
object SpatialLayers {
    data class Layers(val background: Bitmap, val subjectAlpha: Bitmap)

    private fun files(context: Context) = context.createDeviceProtectedStorageContext().filesDir.let { dir ->
        Triple(java.io.File(dir, "spatial_key.txt"), java.io.File(dir, "spatial_bg.png"), java.io.File(dir, "spatial_alpha.png"))
    }
    // v2: the fill covers the whole subject (v1 left parts of tall subjects unfilled).
    // The Best model makes a different cut, so it has its own cache entry.
    private fun key(context: Context, photo: java.io.File) =
        "${photo.length()}:${photo.lastModified()}:v2" + if (wantsBest(context)) ":best" else ""
    private val building = java.util.concurrent.atomic.AtomicBoolean(false)
    @Volatile private var failedKey: String? = null

    /** Whether a build for [photo] should start: none running, and it has not already failed. */
    fun shouldBuild(context: Context, photo: java.io.File): Boolean = photo.exists() && failedKey != key(context, photo) && !building.get()

    /** The cached layers for [photo], or null when they have not been built for this photo yet. Fast. */
    fun cached(context: Context, photo: java.io.File): Layers? {
        val (keyFile, bgFile, alphaFile) = files(context)
        if (!keyFile.exists() || keyFile.readText() != key(context, photo)) return null
        val bg = BitmapFactory.decodeFile(bgFile.path)
        val alpha = BitmapFactory.decodeFile(alphaFile.path)
        if (bg != null && alpha != null) return Layers(bg, alpha)
        bg?.recycle(); alpha?.recycle()
        return null
    }

    /**
     * Builds and caches the layers for [photo]. Slow (seconds): never on a render thread. One build
     * at a time per process; a photo that fails (no subject, out of memory) is not retried, so a
     * failure cannot turn into a rebuild loop or a crash loop of this persistent app.
     */
    fun build(context: Context, photo: java.io.File): Boolean {
        if (!building.compareAndSet(false, true)) return false
        val key = key(context, photo)
        try {
            val (keyFile, bgFile, alphaFile) = files(context)
            val image = BitmapFactory.decodeFile(photo.path) ?: return false.also { failedKey = key }
            val built = try { build(context, image) } finally { image.recycle() } ?: return false.also { failedKey = key }
            try {
                keyFile.delete()
                fun write(target: java.io.File, bitmap: Bitmap) {
                    val partial = java.io.File(target.path + ".partial")
                    partial.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
                    if (!partial.renameTo(target)) throw java.io.IOException("rename failed: $target")
                }
                write(bgFile, built.background)
                write(alphaFile, built.subjectAlpha)
                // Written last: readers trust the files only once the key matches.
                keyFile.writeText(key)
                return true
            } finally {
                built.background.recycle()
                built.subjectAlpha.recycle()
            }
        } catch (t: Throwable) {
            Log.e("SpatialLayers", "Spatial layers failed", t)
            failedKey = key
            return false
        } finally {
            building.set(false)
        }
    }

    private fun build(context: Context, image: Bitmap): Layers? {
        val segmenter = PortraitSegmenter(context)
        val foreground = try {
            segmenter.init()
            segmenter.segment(image)
        } finally {
            segmenter.release()
        } ?: return null
        val w = image.width
        val h = image.height
        val fg = if (foreground.width == w && foreground.height == h) foreground
            else Bitmap.createScaledBitmap(foreground, w, h, true).also { foreground.recycle() }
        val pixels = IntArray(w * h)
        fg.getPixels(pixels, 0, w, 0, 0, w, h)
        fg.recycle()
        // Grey alpha texture of the subject, soft edges kept.
        val alpha = IntArray(w * h) { i -> val a = Color.alpha(pixels[i]); Color.argb(255, a, a, a) }
        // The hole reaches past the subject's edge by the most the background can move, so no
        // trace of the subject is left in the background when it slides out from behind it.
        val hole = BooleanArray(w * h) { Color.alpha(pixels[it]) > 24 }
        val grow = (w * 0.035f).toInt().coerceAtLeast(4)
        dilate(hole, w, h, grow)
        val inpainter = Inpainter(context)
        val background = try {
            if (inpainter.init()) inpainter.fill(image, hole) else null
        } finally {
            inpainter.release()
        } ?: return null
        return Layers(background, Bitmap.createBitmap(alpha, w, h, Bitmap.Config.ARGB_8888))
    }

    /** Separable square dilation by [r] pixels: fast enough for a full-size wallpaper. */
    private fun dilate(mask: BooleanArray, w: Int, h: Int, r: Int) {
        val tmp = BooleanArray(w * h)
        val dist = IntArray(maxOf(w, h))
        for (y in 0 until h) {
            var d = r + 1
            for (x in 0 until w) { d = if (mask[y * w + x]) 0 else d + 1; dist[x] = d }
            d = r + 1
            for (x in w - 1 downTo 0) { d = if (mask[y * w + x]) 0 else d + 1; tmp[y * w + x] = minOf(d, dist[x]) <= r }
        }
        for (x in 0 until w) {
            var d = r + 1
            for (y in 0 until h) { d = if (tmp[y * w + x]) 0 else d + 1; dist[y] = d }
            d = r + 1
            for (y in h - 1 downTo 0) { d = if (tmp[y * w + x]) 0 else d + 1; mask[y * w + x] = minOf(d, dist[y]) <= r }
        }
    }
}
