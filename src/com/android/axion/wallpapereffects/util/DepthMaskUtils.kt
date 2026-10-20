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

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Path
import android.graphics.PathIterator
import android.graphics.Region
import android.util.Base64
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt
import kotlin.math.sqrt

object DepthMaskUtils {

    private const val TAG = "DepthMaskUtils"
    private const val COMPACT_MASK_WIDTH = 128
    private const val PATH_EXTRACT_WIDTH = 512
    private const val MIN_SUBJECT_RATIO = 0.02f
    private const val MASK_ALPHA_THRESHOLD = 128
    private const val PATH_SIMPLIFY_EPSILON = 1.0f
    private const val SMOOTHING_PASSES = 2
    private const val NORMALIZE_RANGE = 10000f

    fun extractCompactMask(foreground: Bitmap): String? {
        if (foreground.width <= 0 || foreground.height <= 0) return null

        val aspectRatio = foreground.height.toFloat() / foreground.width
        val compactW = COMPACT_MASK_WIDTH
        val compactH = (compactW * aspectRatio).roundToInt().coerceIn(1, 255)

        val scaled = Bitmap.createScaledBitmap(foreground, compactW, compactH, true)
        val pixels = IntArray(compactW * compactH)
        scaled.getPixels(pixels, 0, compactW, 0, 0, compactW, compactH)
        if (scaled !== foreground) scaled.recycle()

        var nonZeroCount = 0
        val alphaBytes = ByteArray(compactW * compactH)
        for (i in pixels.indices) {
            val alpha = Color.alpha(pixels[i])
            val thresholded = if (alpha > MASK_ALPHA_THRESHOLD) 255 else 0
            alphaBytes[i] = thresholded.toByte()
            if (thresholded > 0) nonZeroCount++
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

    /**
     * Tunables for the subject outline, chosen per lock screen preset. Encoded as
     * "subject=all;edge=loose;detail=fine" in Settings.Secure ax_depth_options; unknown or
     * missing keys keep Axion's original behaviour.
     */
    data class Options(
        val allSubjects: Boolean = false,
        /** Pixels at the 512-wide extraction size: negative shrinks the outline, positive grows it. */
        val growPx: Int = 0,
        val threshold: Int = MASK_ALPHA_THRESHOLD,
        val epsilon: Float = PATH_SIMPLIFY_EPSILON,
        val smoothing: Int = SMOOTHING_PASSES,
        /** Tapped points, 0..1 over the bitmap given to extractSubjectPath; keep only parts there. */
        val picks: List<FloatArray> = emptyList(),
    ) {
        /** [picks] given over a whole photo, re-expressed over its [crop] (pixels of a [w]x[h] photo). */
        fun inCrop(crop: android.graphics.Rect, w: Int, h: Int): Options = copy(picks = picks.mapNotNull { p ->
            val x = (p[0] * w - crop.left) / crop.width().coerceAtLeast(1)
            val y = (p[1] * h - crop.top) / crop.height().coerceAtLeast(1)
            if (x in 0f..1f && y in 0f..1f) floatArrayOf(x, y) else null
        })

        companion object {
            fun parse(encoded: String?): Options {
                val values = encoded.orEmpty().split(';').mapNotNull { part ->
                    part.split('=', limit = 2).takeIf { it.size == 2 }?.let { it[0].trim() to it[1].trim() }
                }.toMap()
                return Options(
                    allSubjects = values["subject"] == "all",
                    growPx = when (values["edge"]) { "tight" -> -2; "loose" -> 3; else -> 0 },
                    threshold = when (values["edge"]) { "tight" -> 170; "loose" -> 90; else -> MASK_ALPHA_THRESHOLD },
                    epsilon = when (values["detail"]) { "fine" -> 0.5f; "smooth" -> 2.0f; else -> PATH_SIMPLIFY_EPSILON },
                    smoothing = when (values["detail"]) { "fine" -> 1; "smooth" -> 3; else -> SMOOTHING_PASSES },
                    picks = values["pick"].orEmpty().split(',').mapNotNull { pair ->
                        val xy = pair.split(':').mapNotNull { it.toFloatOrNull() }
                        if (xy.size == 2) floatArrayOf(xy[0], xy[1]) else null
                    },
                )
            }
        }
    }

    fun extractSubjectPath(foreground: Bitmap, options: Options = Options()): String? {
        if (foreground.width <= 0 || foreground.height <= 0) return null

        val aspectRatio = foreground.height.toFloat() / foreground.width
        val extractW = PATH_EXTRACT_WIDTH
        val extractH = (extractW * aspectRatio).roundToInt().coerceAtLeast(1)

        val scaled = Bitmap.createScaledBitmap(foreground, extractW, extractH, true)
        val pixels = IntArray(extractW * extractH)
        scaled.getPixels(pixels, 0, extractW, 0, 0, extractW, extractH)
        if (scaled !== foreground) scaled.recycle()

        val mask = BooleanArray(extractW * extractH)
        var nonZeroCount = 0
        for (i in pixels.indices) {
            if (Color.alpha(pixels[i]) > options.threshold) {
                mask[i] = true
                nonZeroCount++
            }
        }
        if (nonZeroCount.toFloat() / pixels.size < MIN_SUBJECT_RATIO) {
            Log.d(TAG, "No significant subject for path ($nonZeroCount/${pixels.size} fg pixels)")
            return null
        }

        val subjectCount = when {
            options.picks.isNotEmpty() -> keepPickedComponents(mask, extractW, extractH, options.picks)
                .takeIf { it > 0 } ?: keepLargestComponent(mask, extractW, extractH)
            options.allSubjects -> nonZeroCount
            else -> keepLargestComponent(mask, extractW, extractH)
        }
        if (subjectCount.toFloat() / pixels.size < MIN_SUBJECT_RATIO) {
            Log.d(TAG, "No significant connected subject ($subjectCount/${pixels.size} fg pixels)")
            return null
        }

        fillInternalHoles(mask, extractW, extractH)
        if (options.growPx != 0) morph(mask, extractW, extractH, options.growPx)

        val region = Region()
        for (y in 0 until extractH) {
            var spanStart = -1
            for (x in 0 until extractW) {
                if (mask[y * extractW + x]) {
                    if (spanStart == -1) spanStart = x
                } else {
                    if (spanStart != -1) {
                        region.op(spanStart, y, x, y + 1, Region.Op.UNION)
                        spanStart = -1
                    }
                }
            }
            if (spanStart != -1) {
                region.op(spanStart, y, extractW, y + 1, Region.Op.UNION)
            }
        }

        val boundaryPath = Path()
        if (!region.getBoundaryPath(boundaryPath) || boundaryPath.isEmpty) {
            Log.d(TAG, "Empty boundary path")
            return null
        }

        val contours = extractContours(boundaryPath)
        if (contours.isEmpty()) {
            Log.d(TAG, "No contours extracted")
            return null
        }

        val epsilon = options.epsilon / extractW * NORMALIZE_RANGE
        val simplified =
            contours
                .map { contour ->
                    val normalized =
                        contour.map { pt ->
                            floatArrayOf(
                                pt[0] / extractW * NORMALIZE_RANGE,
                                pt[1] / extractH * NORMALIZE_RANGE,
                            )
                        }
                    val reduced = douglasPeucker(normalized, epsilon)
                    smoothContour(reduced, options.smoothing)
                }
                .filter { it.size >= 3 }

        if (simplified.isEmpty()) {
            Log.d(TAG, "All contours simplified away")
            return null
        }

        val totalPoints = simplified.sumOf { it.size }
        val bufSize = 1 + 4 + 2 + simplified.sumOf { 2 + it.size * 4 }
        val buf = ByteBuffer.allocate(bufSize).order(ByteOrder.BIG_ENDIAN)

        buf.put(0x01.toByte())
        buf.putShort(extractW.toShort())
        buf.putShort(extractH.toShort())
        buf.putShort(simplified.size.toShort())

        for (contour in simplified) {
            buf.putShort(contour.size.toShort())
            for (pt in contour) {
                buf.putShort(pt[0].roundToInt().coerceIn(0, 10000).toShort())
                buf.putShort(pt[1].roundToInt().coerceIn(0, 10000).toShort())
            }
        }

        val encoded = Base64.encodeToString(buf.array(), Base64.NO_WRAP)
        Log.d(
            TAG,
            "Subject path: ${extractW}x${extractH}, ${simplified.size} contours, " +
                "$totalPoints points, ${encoded.length} chars",
        )
        return encoded
    }

    /** Grows (positive) or shrinks (negative) the mask by [steps] pixels, one 4-neighbour pass per step. */
    private fun morph(mask: BooleanArray, w: Int, h: Int, steps: Int) {
        val grow = steps > 0
        repeat(kotlin.math.abs(steps)) {
            val source = mask.copyOf()
            for (y in 0 until h) for (x in 0 until w) {
                val i = y * w + x
                if (source[i] == grow) continue
                val touches = (x > 0 && source[i - 1] == grow) || (x < w - 1 && source[i + 1] == grow) ||
                    (y > 0 && source[i - w] == grow) || (y < h - 1 && source[i + w] == grow)
                if (touches) mask[i] = grow
            }
        }
    }

    private fun extractContours(path: Path): List<List<FloatArray>> {
        val contours = mutableListOf<MutableList<FloatArray>>()
        var current: MutableList<FloatArray>? = null
        val pts = FloatArray(8)

        val iterator = path.getPathIterator()
        while (iterator.hasNext()) {
            when (iterator.next(pts, 0)) {
                PathIterator.VERB_MOVE -> {
                    current = mutableListOf(floatArrayOf(pts[0], pts[1]))
                    contours.add(current)
                }
                PathIterator.VERB_LINE -> {
                    current?.add(floatArrayOf(pts[2], pts[3]))
                }
                PathIterator.VERB_CLOSE -> {
                    current = null
                }
            }
        }

        return contours
    }

    private fun douglasPeucker(points: List<FloatArray>, epsilon: Float): List<FloatArray> {
        if (points.size < 3) return points

        var maxDist = 0f
        var maxIdx = 0
        val first = points.first()
        val last = points.last()

        for (i in 1 until points.size - 1) {
            val dist = perpendicularDistance(points[i], first, last)
            if (dist > maxDist) {
                maxDist = dist
                maxIdx = i
            }
        }

        return if (maxDist > epsilon) {
            val left = douglasPeucker(points.subList(0, maxIdx + 1), epsilon)
            val right = douglasPeucker(points.subList(maxIdx, points.size), epsilon)
            left.dropLast(1) + right
        } else {
            listOf(first, last)
        }
    }

    private fun smoothContour(points: List<FloatArray>, iterations: Int): List<FloatArray> {
        if (points.size < 3 || iterations <= 0) return points
        var current = points
        for (it in 0 until iterations) {
            if (current.size >= 200) break
            val smoothed = mutableListOf(floatArrayOf(0f, 0f)).apply { clear() }
            val n = current.size
            for (i in 0 until n) {
                val p0 = current[i]
                val p1 = current[(i + 1) % n]
                val qx = 0.75f * p0[0] + 0.25f * p1[0]
                val qy = 0.75f * p0[1] + 0.25f * p1[1]
                val rx = 0.25f * p0[0] + 0.75f * p1[0]
                val ry = 0.25f * p0[1] + 0.75f * p1[1]
                smoothed.add(floatArrayOf(qx, qy))
                smoothed.add(floatArrayOf(rx, ry))
            }
            current = smoothed
        }
        return current
    }

    private fun perpendicularDistance(
        point: FloatArray,
        lineStart: FloatArray,
        lineEnd: FloatArray,
    ): Float {
        val dx = lineEnd[0] - lineStart[0]
        val dy = lineEnd[1] - lineStart[1]
        val lenSq = dx * dx + dy * dy
        if (lenSq == 0f) {
            val px = point[0] - lineStart[0]
            val py = point[1] - lineStart[1]
            return sqrt(px * px + py * py)
        }
        val t = ((point[0] - lineStart[0]) * dx + (point[1] - lineStart[1]) * dy) / lenSq
        val projX = lineStart[0] + t * dx
        val projY = lineStart[1] + t * dy
        val px = point[0] - projX
        val py = point[1] - projY
        return sqrt(px * px + py * py)
    }

    private fun fillInternalHoles(mask: BooleanArray, w: Int, h: Int) {
        val visited = BooleanArray(w * h)
        val queue = ArrayDeque<Int>(w * 2 + h * 2)

        for (x in 0 until w) {
            val top = x
            if (!mask[top] && !visited[top]) {
                visited[top] = true
                queue.add(top)
            }
            val bottom = (h - 1) * w + x
            if (!mask[bottom] && !visited[bottom]) {
                visited[bottom] = true
                queue.add(bottom)
            }
        }
        for (y in 1 until h - 1) {
            val left = y * w
            if (!mask[left] && !visited[left]) {
                visited[left] = true
                queue.add(left)
            }
            val right = y * w + w - 1
            if (!mask[right] && !visited[right]) {
                visited[right] = true
                queue.add(right)
            }
        }

        while (queue.isNotEmpty()) {
            val idx = queue.removeFirst()
            val x = idx % w
            val y = idx / w
            if (x > 0) {
                val n = idx - 1
                if (!mask[n] && !visited[n]) {
                    visited[n] = true
                    queue.add(n)
                }
            }
            if (x < w - 1) {
                val n = idx + 1
                if (!mask[n] && !visited[n]) {
                    visited[n] = true
                    queue.add(n)
                }
            }
            if (y > 0) {
                val n = idx - w
                if (!mask[n] && !visited[n]) {
                    visited[n] = true
                    queue.add(n)
                }
            }
            if (y < h - 1) {
                val n = idx + w
                if (!mask[n] && !visited[n]) {
                    visited[n] = true
                    queue.add(n)
                }
            }
        }

        for (i in mask.indices) {
            if (!mask[i] && !visited[i]) {
                mask[i] = true
            }
        }
    }

    /**
     * Keeps the connected parts under the tapped points; a tap just beside a part (within 4% of
     * the width) still takes it. Returns the kept pixel count, 0 when no tap hit anything (the
     * mask is then left untouched).
     */
    private fun keepPickedComponents(mask: BooleanArray, w: Int, h: Int, picks: List<FloatArray>): Int {
        val labels = IntArray(mask.size)
        val queue = IntArray(mask.size)
        var label = 0
        for (start in mask.indices) {
            if (!mask[start] || labels[start] != 0) continue
            label++
            var head = 0
            var tail = 0
            labels[start] = label
            queue[tail++] = start
            while (head < tail) {
                val idx = queue[head++]
                val x = idx % w
                val y = idx / w
                for (next in intArrayOf(if (x > 0) idx - 1 else -1, if (x < w - 1) idx + 1 else -1,
                        if (y > 0) idx - w else -1, if (y < h - 1) idx + w else -1)) {
                    if (next >= 0 && mask[next] && labels[next] == 0) { labels[next] = label; queue[tail++] = next }
                }
            }
        }
        val radius = (w * 0.04f).toInt().coerceAtLeast(2)
        val kept = HashSet<Int>()
        for (pick in picks) {
            val px = (pick[0] * w).toInt().coerceIn(0, w - 1)
            val py = (pick[1] * h).toInt().coerceIn(0, h - 1)
            var best = -1
            var bestDistance = Int.MAX_VALUE
            for (dy in -radius..radius) for (dx in -radius..radius) {
                val x = px + dx
                val y = py + dy
                if (x !in 0 until w || y !in 0 until h) continue
                val l = labels[y * w + x]
                val d = dx * dx + dy * dy
                if (l != 0 && d < bestDistance) { best = l; bestDistance = d }
            }
            if (best > 0) kept += best
        }
        if (kept.isEmpty()) return 0
        var count = 0
        for (i in mask.indices) {
            mask[i] = labels[i] in kept
            if (mask[i]) count++
        }
        return count
    }

    private fun keepLargestComponent(mask: BooleanArray, w: Int, h: Int): Int {
        val visited = BooleanArray(mask.size)
        val labels = IntArray(mask.size)
        val queue = IntArray(mask.size)
        var label = 0
        var bestLabel = 0
        var bestCount = 0

        for (start in mask.indices) {
            if (!mask[start] || visited[start]) continue

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
                if (x > 0) {
                    tail = addComponentPixel(idx - 1, mask, visited, queue, tail)
                }
                if (x < w - 1) {
                    tail = addComponentPixel(idx + 1, mask, visited, queue, tail)
                }
                if (y > 0) {
                    tail = addComponentPixel(idx - w, mask, visited, queue, tail)
                }
                if (y < h - 1) {
                    tail = addComponentPixel(idx + w, mask, visited, queue, tail)
                }
            }

            if (count > bestCount) {
                bestCount = count
                bestLabel = label
            }
        }

        if (bestCount == 0) return 0

        for (i in mask.indices) {
            if (labels[i] != bestLabel) {
                mask[i] = false
            }
        }

        return bestCount
    }

    private fun addComponentPixel(
        idx: Int,
        mask: BooleanArray,
        visited: BooleanArray,
        queue: IntArray,
        tail: Int,
    ): Int {
        if (!mask[idx] || visited[idx]) return tail
        visited[idx] = true
        queue[tail] = idx
        return tail + 1
    }
}
