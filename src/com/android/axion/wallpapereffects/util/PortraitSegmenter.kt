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
import android.graphics.*
import android.graphics.Bitmap.Config.ARGB_8888
import android.util.Base64
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min
import kotlin.math.roundToInt
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod
import org.tensorflow.lite.support.image.ops.TensorOperatorWrapper
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

private const val TAG = "PortraitSegmenter"
private const val DEBUG = false
private const val RAID_MODEL = "mobile_bg_removal_mosaic_dm1_w_metadata.f16.tflite"
private const val MATTING_MODEL = "portrait_matting_mask_1024_768.tflite"
private const val FG_ESTIMATION_MODEL = "foreground_estimator_5680_512_512.tflite"
private const val MODELS_DIR = "segmentation_models_v2"

class PortraitSegmenter(private val context: Context) {
    private var raidModelFile: File? = null
    private var mattingModelFile: File? = null
    private var fgEstModelFile: File? = null
    private var hasFullPipeline = false
    private val interpreterOptions = Interpreter.Options().apply { setNumThreads(4) }

    fun init() {
        try {
            raidModelFile = cacheModel(RAID_MODEL)
            if (DEBUG) Log.d(TAG, "RAID model ready: ${raidModelFile!!.length()} bytes")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to cache RAID model", e)
            return
        }
        try {
            mattingModelFile = cacheModel(MATTING_MODEL)
            if (DEBUG) Log.d(TAG, "Matting model ready: ${mattingModelFile!!.length()} bytes")
        } catch (e: Exception) {
            Log.w(TAG, "Matting model not available", e)
        }
        try {
            fgEstModelFile = cacheModel(FG_ESTIMATION_MODEL)
            if (DEBUG) Log.d(TAG, "FG estimation model ready: ${fgEstModelFile!!.length()} bytes")
        } catch (e: Exception) {
            Log.w(TAG, "FG estimation model not available", e)
        }
        hasFullPipeline = mattingModelFile != null && fgEstModelFile != null
        if (DEBUG) {
            Log.d(
                TAG,
                "Pipeline mode: ${if (hasFullPipeline) "FULL 3-stage" else "RAID-only fallback"}",
            )
        }
    }

    fun isReady(): Boolean = raidModelFile != null

    private fun cacheModel(fileName: String): File {
        val deContext = context.createDeviceProtectedStorageContext()
        val modelsDir = File(deContext.cacheDir.absolutePath, MODELS_DIR)
        modelsDir.mkdirs()
        val cachedModel = File(modelsDir, fileName)
        if (!cachedModel.exists() || cachedModel.length() == 0L) {
            cachedModel.delete()
            context.assets.open(fileName).use { input ->
                FileOutputStream(cachedModel).use { output -> input.copyTo(output) }
            }
            Log.d(TAG, "Cached model $fileName: ${cachedModel.length()} bytes")
        }
        return cachedModel
    }

    /**
     * [best] runs BiRefNet first (Rime's "Best" depth option); null reads it from the applied
     * depth options. Any failure there falls back to the fast pipeline below.
     */
    fun segment(bitmap: Bitmap, best: Boolean? = null): Bitmap? {
        if (best ?: wantsBest(context)) BestCutter.cut(context, bitmap)?.let { return it }
        val raidFile = raidModelFile ?: return null

        if (!hasFullPipeline) {
            return fallbackSegment(raidFile, bitmap)
        }

        return try {
            val startTime = System.currentTimeMillis()

            val raidOutputBuffers = HashMap<Int, TensorBuffer>()
            val raidInputImage: TensorImage
            val coarseMask: Bitmap
            Interpreter(raidFile, interpreterOptions).use { raid ->
                raidInputImage = convertRaidInput(raid, bitmap)
                runModel(raid, arrayOf(raidInputImage.tensorBuffer.buffer), raidOutputBuffers)
                coarseMask = convertRaidOutput(raidOutputBuffers, raid)
            }
            if (DEBUG) {
                Log.d(
                    TAG,
                    "Stage 1 RAID: coarse mask ${coarseMask.width}x${coarseMask.height}" +
                        " (${System.currentTimeMillis() - startTime}ms)",
                )
            }

            val t2 = System.currentTimeMillis()
            val mattingOutputBuffers = HashMap<Int, TensorBuffer>()
            val confidenceBitmap: Bitmap
            val refinedAlpha: Bitmap
            val confidenceOutputShape: IntArray
            val confidenceOutputIndex: Int
            val alphaOutputShape: IntArray
            val alphaOutputIndex: Int
            Interpreter(mattingModelFile!!, interpreterOptions).use { matting ->
                val mattingInputImage = convertMattingImageInput(matting, bitmap)
                val mattingMaskBuffer = convertMaskToInputBuffer(matting, coarseMask)
                runModel(
                    matting,
                    arrayOf(mattingInputImage.tensorBuffer.buffer, mattingMaskBuffer),
                    mattingOutputBuffers,
                )
                val outputIndexes = getMattingOutputIndexes(matting)
                confidenceOutputIndex = outputIndexes.first
                alphaOutputIndex = outputIndexes.second
                confidenceOutputShape = matting.getOutputTensor(confidenceOutputIndex).shape()
                alphaOutputShape = matting.getOutputTensor(alphaOutputIndex).shape()
            }
            val mattingResult =
                convertMattingOutput(
                    mattingOutputBuffers,
                    confidenceOutputIndex,
                    confidenceOutputShape,
                    alphaOutputIndex,
                    alphaOutputShape,
                )
            confidenceBitmap = mattingResult.first
            refinedAlpha = mattingResult.second
            coarseMask.recycle()
            if (DEBUG) {
                Log.d(
                    TAG,
                    "Stage 2 Matting: alpha ${refinedAlpha.width}x${refinedAlpha.height}" +
                        " (${System.currentTimeMillis() - t2}ms)",
                )
            }

            val alphaMatte =
                Bitmap.createScaledBitmap(refinedAlpha, bitmap.width, bitmap.height, true)
            refinedAlpha.recycle()
            confidenceBitmap.recycle()

            val roi = computeAlphaBoundingBox(alphaMatte)
            if (roi == null) {
                if (DEBUG) Log.d(TAG, "No subject detected in bounding box")
                alphaMatte.recycle()
                return null
            }

            val bw = roi.width()
            val bh = roi.height()
            val roiImg = Bitmap.createBitmap(bitmap, roi.left, roi.top, bw, bh)
            val roiMask = Bitmap.createBitmap(alphaMatte, roi.left, roi.top, bw, bh)
            alphaMatte.recycle()

            val cleanFgRoi: Bitmap
            val fgEstFile = fgEstModelFile
            if (fgEstFile != null && fgEstFile.exists()) {
                val t3 = System.currentTimeMillis()
                val fgEstOutputBuffers = HashMap<Int, TensorBuffer>()
                Interpreter(fgEstFile, interpreterOptions).use { fgEst ->
                    val fgEstInputImage = convertRoiImageInput(fgEst, roiImg)
                    val fgEstMaskBuffer = convertRoiMaskInput(fgEst, roiMask)
                    runModel(
                        fgEst,
                        arrayOf(fgEstInputImage.tensorBuffer.buffer, fgEstMaskBuffer),
                        fgEstOutputBuffers,
                    )
                }
                if (DEBUG) Log.d(TAG, "Stage 3 FGEstimation: complete (${System.currentTimeMillis() - t3}ms)")

                val fgOutputTensor = fgEstOutputBuffers[0]!!
                val cleanFg512 = convertFgOutput(fgOutputTensor, fgOutputTensor.shape)
                cleanFgRoi = Bitmap.createScaledBitmap(cleanFg512, bw, bh, true)
                if (cleanFgRoi !== cleanFg512) cleanFg512.recycle()
            } else {
                cleanFgRoi = roiImg
            }

            val finalBmp = composeFinalImage(bitmap.width, bitmap.height, roi, cleanFgRoi, roiMask)
            if (cleanFgRoi !== roiImg) cleanFgRoi.recycle()
            roiImg.recycle()
            roiMask.recycle()

            if (DEBUG) {
                Log.d(
                    TAG,
                    "Full pipeline complete: ${finalBmp.width}x${finalBmp.height} " +
                        "ROI [${roi.left}, ${roi.top}, $bw, $bh] " +
                        "(${System.currentTimeMillis() - startTime}ms)",
                )
            }
            finalBmp
        } catch (e: Exception) {
            Log.e(TAG, "Full pipeline FAILED at runtime, falling back to RAID-only", e)
            fallbackSegment(raidFile, bitmap)
        }
    }

    private fun runModel(
        interp: Interpreter,
        inputBuffers: Array<ByteBuffer>,
        outputBuffers: HashMap<Int, TensorBuffer>,
    ) {
        outputBuffers.clear()
        val outputs = HashMap<Int, Any>()
        for (i in 0 until interp.outputTensorCount) {
            val buf =
                TensorBuffer.createFixedSize(interp.getOutputTensor(i).shape(), DataType.FLOAT32)
            outputBuffers[i] = buf
            outputs[i] = buf.buffer
        }
        interp.runForMultipleInputsOutputs(inputBuffers, outputs)
    }

    private fun scaleDownSmooth(src: Bitmap, targetW: Int, targetH: Int): Bitmap {
        var current = src
        var curW = src.width
        var curH = src.height

        while (curW > targetW * 2 && curH > targetH * 2) {
            curW /= 2
            curH /= 2
            val next = Bitmap.createScaledBitmap(current, curW, curH, true)
            if (current !== src) current.recycle()
            current = next
        }

        val finalBmp = Bitmap.createScaledBitmap(current, targetW, targetH, true)
        if (current !== src) current.recycle()
        return finalBmp
    }

    private fun convertRaidInput(interp: Interpreter, bitmap: Bitmap): TensorImage {
        val shape = interp.getInputTensor(0).shape()
        val scaled = scaleDownSmooth(bitmap, shape[2], shape[1])
        val processor =
            ImageProcessor.Builder()
                .add(TensorOperatorWrapper(CastOp(DataType.FLOAT32)))
                .build()
        val tensorImage = TensorImage(DataType.UINT8)
        tensorImage.load(scaled)
        val result = processor.process(tensorImage)
        if (scaled !== bitmap) scaled.recycle()
        return result
    }

    private fun convertRaidOutput(
        outputBuffers: HashMap<Int, TensorBuffer>,
        interp: Interpreter,
    ): Bitmap {
        val shape = interp.getOutputTensor(1).shape()
        val tensorBuffer = outputBuffers[1]!!

        val processor =
            TensorProcessor.Builder()
                .add(NormalizeOp(0f, 0.003921569f))
                .add(CastOp(DataType.UINT8))
                .build()
        val processed = processor.process(tensorBuffer)

        val h = shape[1]
        val w = shape[2]
        processed.buffer.rewind()
        val mask = Bitmap.createBitmap(w, h, Bitmap.Config.ALPHA_8)
        mask.copyPixelsFromBuffer(processed.buffer)
        return mask
    }

    private fun convertMattingImageInput(interp: Interpreter, bitmap: Bitmap): TensorImage {
        val shape = interp.getInputTensor(0).shape()
        val scaled = scaleDownSmooth(bitmap, shape[2], shape[1])
        val processor =
            ImageProcessor.Builder()
                .add(TensorOperatorWrapper(CastOp(DataType.FLOAT32)))
                .add(
                    TensorOperatorWrapper(
                        NormalizeOp(floatArrayOf(0f, 0f, 0f), floatArrayOf(255f, 255f, 255f))
                    )
                )
                .build()
        val tensorImage = TensorImage(DataType.UINT8)
        tensorImage.load(scaled)
        val result = processor.process(tensorImage)
        if (scaled !== bitmap) scaled.recycle()
        return result
    }

    private fun convertMaskToInputBuffer(interp: Interpreter, mask: Bitmap): ByteBuffer {
        val shape = interp.getInputTensor(1).shape()
        val scaled = Bitmap.createScaledBitmap(mask, shape[2], shape[1], true)
        val tensorBuffer = TensorBuffer.createFixedSize(shape, DataType.UINT8)
        tensorBuffer.buffer.rewind()
        scaled.copyPixelsToBuffer(tensorBuffer.buffer)
        if (scaled !== mask) scaled.recycle()

        val processor =
            TensorProcessor.Builder()
                .add(CastOp(DataType.FLOAT32))
                .add(NormalizeOp(0f, 255f))
                .build()
        return processor.process(tensorBuffer).buffer
    }

    private fun convertMattingOutput(
        outputBuffers: HashMap<Int, TensorBuffer>,
        confidenceOutputIndex: Int,
        confidenceShape: IntArray,
        alphaOutputIndex: Int,
        alphaShape: IntArray,
    ): Pair<Bitmap, Bitmap> {
        val confidenceBitmap = convertConfidenceOutput(
            outputBuffers[confidenceOutputIndex]!!,
            confidenceShape,
        )
        val refinedAlpha = convertAlphaOutput(outputBuffers[alphaOutputIndex]!!, alphaShape)
        return Pair(confidenceBitmap, refinedAlpha)
    }

    private fun convertConfidenceOutput(tensorBuffer: TensorBuffer, shape: IntArray): Bitmap {
        val h = shape[1]
        val w = shape[2]
        val buffer = tensorBuffer.buffer
        buffer.rewind()

        val totalFloats = shape.reduce { acc, v -> acc * v }
        val confidenceBuffer =
            ByteBuffer.allocateDirect(4 * totalFloats).order(ByteOrder.nativeOrder())

        while (buffer.hasRemaining()) {
            val f = buffer.float
            confidenceBuffer.putInt(Color.argb(if (f > 0.99f) 1f else 0f, 0f, 0f, 0f))
        }

        confidenceBuffer.rewind()
        val confidenceBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        confidenceBitmap.copyPixelsFromBuffer(confidenceBuffer)
        return confidenceBitmap
    }

    private fun convertAlphaOutput(tensorBuffer: TensorBuffer, shape: IntArray): Bitmap {
        val h = shape[1]
        val w = shape[2]
        val buffer = tensorBuffer.buffer
        buffer.rewind()

        val totalFloats = shape.reduce { acc, v -> acc * v }
        val alphaBuffer = ByteBuffer.allocateDirect(totalFloats).order(ByteOrder.nativeOrder())
        var minF = Float.MAX_VALUE
        var maxF = -Float.MAX_VALUE

        while (buffer.hasRemaining()) {
            val f = buffer.float
            if (f < minF) minF = f
            if (f > maxF) maxF = f
            val alpha = (f.coerceIn(0f, 1f) * 255f).roundToInt()
            alphaBuffer.put(alpha.toByte())
        }
        if (DEBUG) Log.d(TAG, "convertAlphaOutput: shape=${shape[1]}x${shape[2]}, floatRange=[$minF, $maxF]")

        alphaBuffer.rewind()
        val refinedAlpha = Bitmap.createBitmap(w, h, Bitmap.Config.ALPHA_8)
        refinedAlpha.copyPixelsFromBuffer(alphaBuffer)
        return refinedAlpha
    }

    private fun computeAlphaBoundingBox(alphaBmp: Bitmap): Rect? {
        val w = alphaBmp.width
        val h = alphaBmp.height
        val pixels = IntArray(w * h)
        alphaBmp.getPixels(pixels, 0, w, 0, 0, w, h)

        val visited = BooleanArray(pixels.size)
        val queue = IntArray(pixels.size)
        val labels = IntArray(pixels.size)
        var label = 0
        var bestLabel = 0
        var bestCount = 0

        for (start in pixels.indices) {
            if (Color.alpha(pixels[start]) <= 32 || visited[start]) continue
            label++
            var head = 0
            var tail = 0
            var count = 0
            visited[start] = true
            queue[tail++] = start

            while (head < tail) {
                val idx = queue[head++]
                labels[idx] = label
                count++
                val x = idx % w
                val y = idx / w
                if (x > 0 && !visited[idx - 1] && Color.alpha(pixels[idx - 1]) > 32) {
                    visited[idx - 1] = true
                    queue[tail++] = idx - 1
                }
                if (x < w - 1 && !visited[idx + 1] && Color.alpha(pixels[idx + 1]) > 32) {
                    visited[idx + 1] = true
                    queue[tail++] = idx + 1
                }
                if (y > 0 && !visited[idx - w] && Color.alpha(pixels[idx - w]) > 32) {
                    visited[idx - w] = true
                    queue[tail++] = idx - w
                }
                if (y < h - 1 && !visited[idx + w] && Color.alpha(pixels[idx + w]) > 32) {
                    visited[idx + w] = true
                    queue[tail++] = idx + w
                }
            }

            if (count > bestCount) {
                bestCount = count
                bestLabel = label
            }
        }

        val total = w * h
        if (DEBUG) Log.d(TAG, "computeAlphaBoundingBox: w=$w, h=$h, labels=$label, bestLabel=$bestLabel, bestCount=$bestCount, total=$total, p0Alpha=${Color.alpha(pixels[0])}, centerAlpha=${Color.alpha(pixels[h/2 * w + w/2])}")
        if (bestCount == 0 || bestCount.toFloat() / total < MIN_SUBJECT_RATIO) {
            return null
        }

        var minX = w
        var maxX = -1
        var minY = h
        var maxY = -1

        for (i in pixels.indices) {
            if (labels[i] == bestLabel) {
                val x = i % w
                val y = i / w
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
            } else {
                pixels[i] = 0
            }
        }

        if (DEBUG) Log.d(TAG, "computeAlphaBoundingBox: bbox=[$minX, $minY, ${maxX - minX + 1}, ${maxY - minY + 1}]")
        if (maxX < minX || maxY < minY) return null
        alphaBmp.setPixels(pixels, 0, w, 0, 0, w, h)
        return Rect(minX, minY, maxX + 1, maxY + 1)
    }

    private fun convertRoiImageInput(interp: Interpreter, roi: Bitmap): TensorImage {
        val shape = interp.getInputTensor(0).shape()
        val processor =
            ImageProcessor.Builder()
                .add(ResizeOp(shape[1], shape[2], ResizeMethod.BILINEAR))
                .add(TensorOperatorWrapper(CastOp(DataType.FLOAT32)))
                .add(
                    TensorOperatorWrapper(
                        NormalizeOp(floatArrayOf(0f, 0f, 0f), floatArrayOf(255f, 255f, 255f))
                    )
                )
                .build()
        val tensorImage = TensorImage(DataType.UINT8)
        tensorImage.load(roi)
        return processor.process(tensorImage)
    }

    private fun convertRoiMaskInput(interp: Interpreter, mask: Bitmap): ByteBuffer {
        val shape = interp.getInputTensor(1).shape()
        val scaled = Bitmap.createScaledBitmap(mask, shape[2], shape[1], true)
        val tensorBuffer = TensorBuffer.createFixedSize(shape, DataType.UINT8)
        tensorBuffer.buffer.rewind()
        scaled.copyPixelsToBuffer(tensorBuffer.buffer)
        if (scaled !== mask) scaled.recycle()

        val processor =
            TensorProcessor.Builder()
                .add(CastOp(DataType.FLOAT32))
                .add(NormalizeOp(0f, 255f))
                .build()
        return processor.process(tensorBuffer).buffer
    }

    private fun convertFgOutput(tensorBuffer: TensorBuffer, shape: IntArray): Bitmap {
        val h = shape[1]
        val w = shape[2]
        val buffer = tensorBuffer.buffer
        buffer.rewind()

        val pixels = IntArray(w * h)
        var i = 0
        while (buffer.hasRemaining() && i < pixels.size) {
            val rf = buffer.float
            val gf = buffer.float
            val bf = buffer.float
            val r = (rf.coerceIn(0f, 1f) * 255f).roundToInt()
            val g = (gf.coerceIn(0f, 1f) * 255f).roundToInt()
            val b = (bf.coerceIn(0f, 1f) * 255f).roundToInt()
            pixels[i] = Color.rgb(r, g, b)
            i++
        }

        val result = Bitmap.createBitmap(w, h, ARGB_8888)
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    private fun composeFinalImage(
        origW: Int,
        origH: Int,
        roi: Rect,
        cleanFgRoi: Bitmap,
        roiMask: Bitmap,
    ): Bitmap {
        val bw = roi.width()
        val bh = roi.height()
        val fgPixels = IntArray(bw * bh)
        val maskPixels = IntArray(bw * bh)
        cleanFgRoi.getPixels(fgPixels, 0, bw, 0, 0, bw, bh)
        roiMask.getPixels(maskPixels, 0, bw, 0, 0, bw, bh)

        val outPixels = IntArray(origW * origH)
        for (y in 0 until bh) {
            val outRow = (roi.top + y) * origW + roi.left
            val roiRow = y * bw
            for (x in 0 until bw) {
                val idx = roiRow + x
                val a = Color.alpha(maskPixels[idx])
                if (a > 0) {
                    val rgb = fgPixels[idx]
                    val r = (Color.red(rgb) * a + 127) / 255
                    val g = (Color.green(rgb) * a + 127) / 255
                    val b = (Color.blue(rgb) * a + 127) / 255
                    outPixels[outRow + x] = (a shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }

        val result = Bitmap.createBitmap(origW, origH, ARGB_8888)
        result.setPixels(outPixels, 0, origW, 0, 0, origW, origH)
        return result
    }

    private fun fallbackSegment(raidFile: File, bitmap: Bitmap): Bitmap? {
        return try {
            val raidOutputBuffers = HashMap<Int, TensorBuffer>()
            val maskIdx: Int
            val shape: IntArray
            Interpreter(raidFile, interpreterOptions).use { interp ->
                val inputImage = convertRaidInput(interp, bitmap)
                runModel(interp, arrayOf(inputImage.tensorBuffer.buffer), raidOutputBuffers)
                maskIdx = 1
                shape = interp.getOutputTensor(maskIdx).shape()
            }

            val maskH = shape[1]
            val maskW = shape[2]
            val buf = raidOutputBuffers[maskIdx]!!.buffer
            buf.rewind()

            val maskValues = FloatArray(maskW * maskH)
            for (p in maskValues.indices) {
                val raw = buf.float
                maskValues[p] =
                    when {
                        raw < 0.05f -> 0f
                        raw > 0.95f -> 1f
                        else -> raw
                    }
            }

            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, maskW, maskH, true)
            val srcPixels = IntArray(maskW * maskH)
            scaledBitmap.getPixels(srcPixels, 0, maskW, 0, 0, maskW, maskH)
            scaledBitmap.recycle()

            val resultPixels = IntArray(maskW * maskH)
            for (i in srcPixels.indices) {
                val alpha = (maskValues[i].coerceIn(0f, 1f) * 255f).roundToInt()
                resultPixels[i] =
                    Color.argb(
                        alpha,
                        Color.red(srcPixels[i]),
                        Color.green(srcPixels[i]),
                        Color.blue(srcPixels[i]),
                    )
            }

            val result = Bitmap.createBitmap(maskW, maskH, Bitmap.Config.ARGB_8888)
            result.setPixels(resultPixels, 0, maskW, 0, 0, maskW, maskH)
            cleanupForeground(result)
            val scaled = Bitmap.createScaledBitmap(result, bitmap.width, bitmap.height, true)
            if (scaled !== result) result.recycle()

            Log.d(TAG, "Fallback segmentation complete")
            scaled
        } catch (e: Exception) {
            Log.e(TAG, "Fallback segmentation failed", e)
            null
        }
    }

    private fun cleanupForeground(bitmap: Bitmap): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val alpha = IntArray(w * h)
        for (i in pixels.indices) {
            val a = Color.alpha(pixels[i])
            alpha[i] = if (a < 25) 0 else a
        }

        val eroded = IntArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                var minA = alpha[y * w + x]
                for (dy in -1..1) {
                    for (dx in -1..1) {
                        val nx = x + dx
                        val ny = y + dy
                        if (nx in 0 until w && ny in 0 until h) {
                            minA = min(minA, alpha[ny * w + nx])
                        }
                    }
                }
                eroded[y * w + x] = minA
            }
        }

        for (y in 0 until h) {
            for (x in 0 until w) {
                val idx = y * w + x
                val a = eroded[idx]
                if (a == 0) {
                    pixels[idx] = 0
                } else if (a < 220) {
                    var sr = 0
                    var sg = 0
                    var sb = 0
                    var cnt = 0
                    for (dy in -2..2) {
                        for (dx in -2..2) {
                            val nx = x + dx
                            val ny = y + dy
                            if (nx in 0 until w && ny in 0 until h) {
                                val nIdx = ny * w + nx
                                if (eroded[nIdx] >= 220) {
                                    sr += Color.red(pixels[nIdx])
                                    sg += Color.green(pixels[nIdx])
                                    sb += Color.blue(pixels[nIdx])
                                    cnt++
                                }
                            }
                        }
                    }
                    pixels[idx] =
                        if (cnt > 0) {
                            Color.argb(a, sr / cnt, sg / cnt, sb / cnt)
                        } else {
                            Color.argb(
                                a,
                                Color.red(pixels[idx]),
                                Color.green(pixels[idx]),
                                Color.blue(pixels[idx]),
                            )
                        }
                } else {
                    pixels[idx] =
                        Color.argb(
                            a,
                            Color.red(pixels[idx]),
                            Color.green(pixels[idx]),
                            Color.blue(pixels[idx]),
                        )
                }
            }
        }

        bitmap.setPixels(pixels, 0, w, 0, 0, w, h)
        return bitmap
    }

    private fun getMattingOutputIndexes(interp: Interpreter): Pair<Int, Int> {
        var alphaOutputIndex = 0
        var alphaOutputSize = 0
        var confidenceOutputIndex = 0
        var confidenceOutputSize = Int.MAX_VALUE
        for (i in 0 until interp.outputTensorCount) {
            val outputSize = interp.getOutputTensor(i).shape().fold(1) { acc, value ->
                acc * value
            }
            if (outputSize > alphaOutputSize) {
                alphaOutputIndex = i
                alphaOutputSize = outputSize
            }
            if (outputSize < confidenceOutputSize) {
                confidenceOutputIndex = i
                confidenceOutputSize = outputSize
            }
        }
        return Pair(confidenceOutputIndex, alphaOutputIndex)
    }

    fun extractCompactMask(segmented: Bitmap): String? {
        if (segmented.width <= 0 || segmented.height <= 0) return null

        val aspectRatio = segmented.height.toFloat() / segmented.width
        val compactW = COMPACT_MASK_WIDTH
        val compactH = (compactW * aspectRatio).roundToInt().coerceIn(1, 255)

        val scaled = Bitmap.createScaledBitmap(segmented, compactW, compactH, true)
        val pixels = IntArray(compactW * compactH)
        scaled.getPixels(pixels, 0, compactW, 0, 0, compactW, compactH)
        if (scaled !== segmented) scaled.recycle()

        var nonZeroCount = 0
        val alphaBytes = ByteArray(compactW * compactH)
        for (i in pixels.indices) {
            val alpha = Color.alpha(pixels[i])
            alphaBytes[i] = alpha.toByte()
            if (alpha > 10) nonZeroCount++
        }

        val totalPixels = compactW * compactH
        if (nonZeroCount.toFloat() / totalPixels < MIN_SUBJECT_RATIO) {
            Log.d(TAG, "No significant subject detected ($nonZeroCount/$totalPixels fg pixels)")
            return null
        }

        val payload = ByteArray(2 + alphaBytes.size)
        payload[0] = compactW.toByte()
        payload[1] = compactH.toByte()
        System.arraycopy(alphaBytes, 0, payload, 2, alphaBytes.size)

        val encoded = Base64.encodeToString(payload, Base64.NO_WRAP)
        Log.d(
            TAG,
            "Compact mask: ${compactW}x${compactH}, $nonZeroCount fg pixels, ${encoded.length} chars",
        )
        return encoded
    }

    fun release() {

        raidModelFile = null
        mattingModelFile = null
        fgEstModelFile = null
        hasFullPipeline = false
    }

    companion object {
        private const val COMPACT_MASK_WIDTH = 64
        private const val MIN_SUBJECT_RATIO = 0.02f
    }
}

/** Whether the applied depth options ask for the Best model. */
internal fun wantsBest(context: Context): Boolean =
    android.provider.Settings.Secure.getString(context.contentResolver, "ax_depth_options")?.contains("quality=best") == true

/**
 * BiRefNet-lite (MIT, github.com/ZhengPeng7/BiRefNet), converted to TFLite with the sigmoid
 * baked in: input [1,1024,1024,3] RGB normalised with ImageNet mean/std, output a [0,1] mask.
 * Returns the photo with the mask as its alpha, like the fast pipeline's foreground.
 */
internal object BestCutter {
    private const val TAG = "BestCutter"
    private const val MODEL = "birefnet_lite_1024_fp16.tflite"
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    @Synchronized
    fun cut(context: Context, bitmap: Bitmap): Bitmap? {
        // A native crash cannot be caught. Each start is counted and a finished run clears the
        // count; two unfinished runs in a row on this install (a crash, not just one reboot or
        // kill) switch it to the fast model for good instead of crashing again.
        val prefs = context.createDeviceProtectedStorageContext().getSharedPreferences("best_cutter", Context.MODE_PRIVATE)
        val install = runCatching { context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime }.getOrDefault(0L)
        val unfinished = if (prefs.getLong("install", -1L) == install) prefs.getInt("unfinished", 0) else 0
        if (unfinished >= 2) {
            Log.w(TAG, "Earlier runs did not finish; using the fast model")
            return null
        }
        prefs.edit().putLong("install", install).putInt("unfinished", unfinished + 1).commit()
        val started = System.currentTimeMillis()
        try {
            val model = context.assets.openFd(MODEL).use { fd ->
                java.io.FileInputStream(fd.fileDescriptor).channel.use { channel ->
                    channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
            Interpreter(model, Interpreter.Options().apply { setNumThreads(4); setUseXNNPACK(true) }).use { interpreter ->
                val inShape = interpreter.getInputTensor(0).shape()
                val nchw = inShape.size == 4 && inShape[1] == 3
                val h = if (nchw) inShape[2] else inShape[1]
                val w = if (nchw) inShape[3] else inShape[2]
                val scaled = Bitmap.createScaledBitmap(bitmap, w, h, true)
                val pixels = IntArray(w * h)
                scaled.getPixels(pixels, 0, w, 0, 0, w, h)
                if (scaled !== bitmap) scaled.recycle()
                val input = ByteBuffer.allocateDirect(4 * w * h * 3).order(ByteOrder.nativeOrder())
                val floats = input.asFloatBuffer()
                if (nchw) {
                    for (c in 0 until 3) for (i in pixels.indices) {
                        val v = (pixels[i] shr (16 - 8 * c)) and 0xFF
                        floats.put((v / 255f - MEAN[c]) / STD[c])
                    }
                } else {
                    for (p in pixels) for (c in 0 until 3) {
                        val v = (p shr (16 - 8 * c)) and 0xFF
                        floats.put((v / 255f - MEAN[c]) / STD[c])
                    }
                }
                val outShape = interpreter.getOutputTensor(0).shape()
                val outCount = outShape.fold(1) { a, b -> a * b }
                if (outCount != w * h) { Log.w(TAG, "Unexpected output ${outShape.contentToString()}"); return null }
                val output = ByteBuffer.allocateDirect(4 * outCount).order(ByteOrder.nativeOrder())
                interpreter.run(input, output)
                output.rewind()
                val mask = output.asFloatBuffer()
                // A single-channel mask has the same order either way.
                val alpha = IntArray(w * h) { i -> val a = (mask.get(i).coerceIn(0f, 1f) * 255f + 0.5f).toInt(); a shl 24 }
                val small = Bitmap.createBitmap(alpha, w, h, ARGB_8888)
                val full = Bitmap.createScaledBitmap(small, bitmap.width, bitmap.height, true)
                if (full !== small) small.recycle()
                // The photo, with the mask as its alpha.
                val result = bitmap.copy(ARGB_8888, true)
                val photo = IntArray(bitmap.width * bitmap.height)
                result.getPixels(photo, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
                val a = IntArray(photo.size)
                full.getPixels(a, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
                full.recycle()
                for (i in photo.indices) photo[i] = (a[i] and 0xFF000000.toInt()) or (photo[i] and 0x00FFFFFF)
                result.setPixels(photo, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
                Log.d(TAG, "Cut ${bitmap.width}x${bitmap.height} in ${System.currentTimeMillis() - started} ms")
                return result
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Best model failed; using the fast model", t)
            return null
        } finally {
            // Reached unless the process died: the run finished, whatever its result.
            prefs.edit().putInt("unfinished", 0).commit()
        }
    }
}
