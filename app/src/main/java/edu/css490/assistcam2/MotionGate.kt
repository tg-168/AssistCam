package edu.css490.assistcam2

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

data class MotionDecision(
    val runInference: Boolean,
    val meanDiff: Float,
    val emaMotion: Float,
    val dxPx: Float,
    val dyPx: Float,
    val framesSinceLastInference: Int
)

class MotionGate(
    // Gate (cheap) resolution
    private val gateW: Int = 64,
    private val gateH: Int = 36,

    // Translation (less quantized) resolution
    private val transW: Int = 320,
    private val transH: Int = 180,

    private val alpha: Float = 0.2f,
    private val threshold: Float = 5.0f,
    private val maxSkipFrames: Int = 8,

    // clamp translation in TRANS pixels (avoid crazy jumps)
    private val maxShiftTrans: Float = 24f,

    // if edge energy is too low, translation is unreliable → return 0
    private val minEdgeEnergy: Float = 2_000f
) {
    private var prevGateGray: ByteArray? = null
    private var prevTransGray: ByteArray? = null

    private var emaMotion: Float = 0f
    private var framesSinceLastInference: Int = 0

    fun reset() {
        prevGateGray = null
        prevTransGray = null
        emaMotion = 0f
        framesSinceLastInference = 0
    }

    fun update(bitmap: Bitmap): MotionDecision {
        val firstFrame = (prevGateGray == null || prevTransGray == null)

        // --- Gate gray ---
        val gateGray = toGraySmall(bitmap, gateW, gateH, filter = false)
        val prevG = prevGateGray

        var sumDiff = 0L
        if (prevG != null) {
            val n = gateW * gateH
            for (i in 0 until n) {
                val a = gateGray[i].toInt() and 0xFF
                val b = prevG[i].toInt() and 0xFF
                sumDiff += abs(a - b)
            }
        }
        val meanDiff = if (prevG != null) sumDiff.toFloat() / (gateW * gateH).toFloat() else 0f

        emaMotion = if (firstFrame) meanDiff else alpha * meanDiff + (1f - alpha) * emaMotion

        val mustRunBecauseMotion = emaMotion > threshold
        val mustRunBecauseGap = framesSinceLastInference >= maxSkipFrames
        val runInference = firstFrame || mustRunBecauseMotion || mustRunBecauseGap

        // --- Translation gray ---
        val transGray = toGraySmall(bitmap, transW, transH, filter = false)
        val prevT = prevTransGray

        var dxFull = 0f
        var dyFull = 0f

        // estimate translation ONLY when skipping inference
        if (!runInference && prevT != null) {
            val (dxT, dyT, edgeEnergy) = estimateTranslationByEdgeCentroid(prevT, transGray, transW, transH)

            if (edgeEnergy >= minEdgeEnergy) {
                val dxTc = dxT.coerceIn(-maxShiftTrans, maxShiftTrans)
                val dyTc = dyT.coerceIn(-maxShiftTrans, maxShiftTrans)

                val sx = bitmap.width.toFloat() / transW.toFloat()
                val sy = bitmap.height.toFloat() / transH.toFloat()
                dxFull = dxTc * sx
                dyFull = dyTc * sy
            } else {
                // too little texture/edges → translation unreliable
                dxFull = 0f
                dyFull = 0f
            }
        }

        // update state
        prevGateGray = gateGray
        prevTransGray = transGray
        if (runInference) framesSinceLastInference = 0 else framesSinceLastInference++

        return MotionDecision(
            runInference = runInference,
            meanDiff = meanDiff,
            emaMotion = emaMotion,
            dxPx = dxFull,
            dyPx = dyFull,
            framesSinceLastInference = framesSinceLastInference
        )
    }

    private fun toGraySmall(src: Bitmap, w: Int, h: Int, filter: Boolean): ByteArray {
        val small = Bitmap.createScaledBitmap(src, w, h, filter)
        val size = w * h
        val pixels = IntArray(size)
        small.getPixels(pixels, 0, w, 0, 0, w, h)
        small.recycle()

        val gray = ByteArray(size)
        for (i in 0 until size) {
            val c = pixels[i]
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            val gv = (0.299f * r + 0.587f * g + 0.114f * b).toInt().coerceIn(0, 255)
            gray[i] = gv.toByte()
        }
        return gray
    }

    /**
     * Subpixel-ish translation estimate from the shift of the "center of mass" of edge energy.
     * Returns (dx, dy, totalEdgeEnergy) in TRANS pixels.
     */
    private fun estimateTranslationByEdgeCentroid(
        prev: ByteArray,
        curr: ByteArray,
        w: Int,
        h: Int
    ): Triple<Float, Float, Float> {
        // Compute cheap edge magnitude: |Ix|+|Iy| using central differences.
        fun edgeAt(arr: ByteArray, x: Int, y: Int): Int {
            val xm1 = max(0, x - 1)
            val xp1 = min(w - 1, x + 1)
            val ym1 = max(0, y - 1)
            val yp1 = min(h - 1, y + 1)

            val cL = arr[y * w + xm1].toInt() and 0xFF
            val cR = arr[y * w + xp1].toInt() and 0xFF
            val cU = arr[ym1 * w + x].toInt() and 0xFF
            val cD = arr[yp1 * w + x].toInt() and 0xFF

            return abs(cR - cL) + abs(cD - cU)
        }

        var sumWPrev = 0f
        var sumXPrev = 0f
        var sumYPrev = 0f

        var sumWCurr = 0f
        var sumXCurr = 0f
        var sumYCurr = 0f

        // Avoid borders (derivatives are noisy there)
        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val eP = edgeAt(prev, x, y).toFloat()
                val eC = edgeAt(curr, x, y).toFloat()

                sumWPrev += eP
                sumXPrev += eP * x.toFloat()
                sumYPrev += eP * y.toFloat()

                sumWCurr += eC
                sumXCurr += eC * x.toFloat()
                sumYCurr += eC * y.toFloat()
            }
        }

        val edgeEnergy = min(sumWPrev, sumWCurr)

        if (sumWPrev <= 1e-3f || sumWCurr <= 1e-3f) {
            return Triple(0f, 0f, 0f)
        }

        val cxPrev = sumXPrev / sumWPrev
        val cyPrev = sumYPrev / sumWPrev
        val cxCurr = sumXCurr / sumWCurr
        val cyCurr = sumYCurr / sumWCurr

        val dx = cxCurr - cxPrev
        val dy = cyCurr - cyPrev

        return Triple(dx, dy, edgeEnergy)
    }
}
