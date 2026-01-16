package edu.css490.assistcam2

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    private val textPaint = Paint().apply {
        color = Color.YELLOW
        textSize = 32f
        style = Paint.Style.FILL
    }

    private var detections: List<Detection> = emptyList()
    private var imageWidth: Int = 0
    private var imageHeight: Int = 0

    fun setDetections(dets: List<Detection>, imgW: Int, imgH: Int) {
        detections = dets
        imageWidth = imgW
        imageHeight = imgH
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (imageWidth <= 0 || imageHeight <= 0 || detections.isEmpty()) return

        val viewW = width.toFloat()
        val viewH = height.toFloat()
        val srcW = imageWidth.toFloat()
        val srcH = imageHeight.toFloat()

        // Match PreviewView default: FILL_CENTER (center crop)
        val scale = maxOf(viewW / srcW, viewH / srcH)
        val scaledW = srcW * scale
        val scaledH = srcH * scale

        // How much we crop off after scaling
        val cropX = (scaledW - viewW) / 2f
        val cropY = (scaledH - viewH) / 2f

        for (d in detections) {
            // Model coords are in [0, imageW/H] pixels (from DetectorHelper)
            val left = d.xmin * scale - cropX
            val top = d.ymin * scale - cropY
            val right = d.xmax * scale - cropX
            val bottom = d.ymax * scale - cropY

            // Skip boxes that ended up completely off-screen
            if (right < 0 || bottom < 0 || left > viewW || top > viewH) continue

            canvas.drawRect(left, top, right, bottom, boxPaint)
            canvas.drawText(
                "${d.label} ${"%.2f".format(d.score)}",
                left + 4f,
                top + 28f,
                textPaint
            )
        }
    }
}
