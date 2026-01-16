package edu.css490.assistcam2

fun Detection.shiftByPixels(dx: Float, dy: Float, w: Int, h: Int): Detection {
    // Handles either normalized [0..1] OR pixel coords by checking scale
    val normalized = (this.xmax <= 2f && this.ymax <= 2f)

    fun clamp(v: Float, lo: Float, hi: Float) = v.coerceIn(lo, hi)

    return if (normalized) {
        val ndx = dx / w.toFloat()
        val ndy = dy / h.toFloat()
        this.copy(
            xmin = clamp(xmin + ndx, 0f, 1f),
            ymin = clamp(ymin + ndy, 0f, 1f),
            xmax = clamp(xmax + ndx, 0f, 1f),
            ymax = clamp(ymax + ndy, 0f, 1f)
        )
    } else {
        val wf = w.toFloat()
        val hf = h.toFloat()
        this.copy(
            xmin = clamp(xmin + dx, 0f, wf),
            ymin = clamp(ymin + dy, 0f, hf),
            xmax = clamp(xmax + dx, 0f, wf),
            ymax = clamp(ymax + dy, 0f, hf)
        )
    }
}
