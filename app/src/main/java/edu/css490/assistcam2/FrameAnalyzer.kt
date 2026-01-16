package edu.css490.assistcam2

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

enum class Mode {
    BASELINE_FULL,
    SKIP_K2,
    SKIP_MOTION,
    SKIP_MOTION_TRACK
}

class FrameAnalyzer(
    private val detectorHelper: DetectorHelper,
    private val updateStats: (String) -> Unit,
    private val mode: Mode = Mode.BASELINE_FULL,
    private val motionGate: MotionGate? = null,
    private val onDetections: (List<Detection>, Int, Int) -> Unit
) : ImageAnalysis.Analyzer {

    private var frameIndex = 0
    private var inferenceCalls = 0
    private val latenciesMs = mutableListOf<Long>()
    private var lastDetections: List<Detection> = emptyList()
    private var startTimeNs: Long = 0L

    override fun analyze(image: ImageProxy) {
        frameIndex++
        if (startTimeNs == 0L) startTimeNs = System.nanoTime()

        val bitmap = image.toBitmap()
        val frameW = bitmap.width
        val frameH = bitmap.height

        var motionScore: Float? = null
        var dxPx: Float? = null
        var dyPx: Float? = null
        var propagated: Boolean? = null

        val needInference: Boolean = when (mode) {
            Mode.BASELINE_FULL -> true
            Mode.SKIP_K2 -> (frameIndex % 2 == 0)

            Mode.SKIP_MOTION,
            Mode.SKIP_MOTION_TRACK -> {
                val gate = motionGate
                if (gate == null) true else {
                    val d = gate.update(bitmap)
                    motionScore = d.emaMotion
                    dxPx = d.dxPx
                    dyPx = d.dyPx
                    d.runInference
                }
            }
        }

        if (needInference) {
            val t0 = System.nanoTime()
            lastDetections = detectorHelper.detect(bitmap)
            val t1 = System.nanoTime()
            val latencyMs = (t1 - t0) / 1_000_000
            latenciesMs.add(latencyMs)
            inferenceCalls++
            propagated = false
        } else {
            // SKIP path
            if (mode == Mode.SKIP_MOTION_TRACK && dxPx != null && dyPx != null) {
                val shifted = shiftDetections(lastDetections, dxPx!!, dyPx!!, frameW, frameH)
                lastDetections = shifted          // important: accumulate motion across skipped frames
                propagated = true
            } else {
                propagated = true // (reusing last detections)
            }
        }

        val usedDetections = lastDetections
        onDetections(usedDetections, frameW, frameH)

        // Stats
        val elapsedSec = (System.nanoTime() - startTimeNs) / 1_000_000_000.0
        val fps = if (elapsedSec > 0) frameIndex / elapsedSec else 0.0
        val callsPerSec = if (elapsedSec > 0) inferenceCalls / elapsedSec else 0.0

        if (latenciesMs.size >= 10) {
            val sorted = latenciesMs.sorted()
            val median = sorted[sorted.size / 2]
            val p90 = sorted[(sorted.size * 9) / 10]

            val text = buildString {
                appendLine("mode=$mode")
                appendLine("frames=$frameIndex calls=$inferenceCalls")
                appendLine("fps=%.1f calls/s=%.1f".format(fps, callsPerSec))
                appendLine("lat med=${median}ms p90=${p90}ms")
                appendLine("objs=${usedDetections.size}")
                if (motionScore != null) appendLine("emaMotion=%.2f".format(motionScore))
                if (dxPx != null && dyPx != null) appendLine("shift=(%.1f, %.1f)px".format(dxPx, dyPx))
                if (propagated != null) appendLine("propagated=$propagated")
            }
            updateStats(text)
        }

        image.close()
    }

    private fun shiftDetections(
        dets: List<Detection>,
        dx: Float,
        dy: Float,
        w: Int,
        h: Int
    ): List<Detection> {
        val maxX = (w - 1).toFloat()
        val maxY = (h - 1).toFloat()
        return dets.mapNotNull { d ->
            val x1 = (d.xmin + dx).coerceIn(0f, maxX)
            val y1 = (d.ymin + dy).coerceIn(0f, maxY)
            val x2 = (d.xmax + dx).coerceIn(0f, maxX)
            val y2 = (d.ymax + dy).coerceIn(0f, maxY)
            if (x2 <= x1 || y2 <= y1) null else d.copy(xmin = x1, ymin = y1, xmax = x2, ymax = y2)
        }
    }
}
