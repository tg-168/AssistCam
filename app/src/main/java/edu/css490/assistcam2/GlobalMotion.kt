package edu.css490.assistcam2

import kotlin.math.abs

object GlobalMotion {

    // Returns (dx, dy) in SMALL-FRAME pixels (downsampled coords)
    fun estimateTranslationSAD(
        prev: ByteArray,
        curr: ByteArray,
        w: Int,
        h: Int,
        maxShift: Int = 4
    ): Pair<Int, Int> {
        var bestDx = 0
        var bestDy = 0
        var bestCost = Long.MAX_VALUE

        for (dy in -maxShift..maxShift) {
            for (dx in -maxShift..maxShift) {
                var cost = 0L

                val xStart = maxOf(0, dx)
                val xEnd = minOf(w - 1, w - 1 + dx)
                val yStart = maxOf(0, dy)
                val yEnd = minOf(h - 1, h - 1 + dy)

                for (y in yStart..yEnd) {
                    val py = y - dy
                    val rowOffC = y * w
                    val rowOffP = py * w
                    for (x in xStart..xEnd) {
                        val px = x - dx
                        val c = curr[rowOffC + x].toInt() and 0xFF
                        val p = prev[rowOffP + px].toInt() and 0xFF
                        cost += abs(c - p)
                    }
                }

                if (cost < bestCost) {
                    bestCost = cost
                    bestDx = dx
                    bestDy = dy
                }
            }
        }
        return Pair(bestDx, bestDy)
    }
}
