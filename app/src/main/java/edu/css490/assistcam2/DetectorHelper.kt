package edu.css490.assistcam2

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.detector.ObjectDetector

data class Detection(
    val label: String,
    val score: Float,
    val xmin: Float,
    val ymin: Float,
    val xmax: Float,
    val ymax: Float
)

class DetectorHelper(context: Context) {

    private val detector: ObjectDetector

    init {
        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setMaxResults(20)
            .setScoreThreshold(0.3f)
            .build()

        detector = ObjectDetector.createFromFileAndOptions(
            context,
            "4.tflite",   // must match your asset filename
            options
        )
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        val image = TensorImage.fromBitmap(bitmap)
        val results = detector.detect(image)

        val out = mutableListOf<Detection>()
        for (obj in results) {
            val cat = obj.categories.firstOrNull() ?: continue
            val box = obj.boundingBox
            out += Detection(
                label = cat.label,
                score = cat.score,
                xmin = box.left,
                ymin = box.top,
                xmax = box.right,
                ymax = box.bottom
            )
        }
        return out
    }
}
