package edu.css490.assistcam2

import android.content.Context
import java.io.File
import java.io.FileWriter

class EvalLogger(context: Context, filename: String) {

    private val file: File
    private val writer: FileWriter

    init {
        val dir = context.getExternalFilesDir(null) ?: context.filesDir
        file = File(dir, filename)
        writer = FileWriter(file, false) // overwrite
    }

    fun logFrame(
        frameIdx: Int,
        mode: Mode,
        inferenceRan: Boolean,
        latencyMs: Long?,
        detections: List<Detection>,
        motionScore: Float? = null,
        isKeyFrame: Boolean? = null,
        dxPx: Float? = null,
        dyPx: Float? = null,
        propagated: Boolean? = null
    ) {
        val sb = StringBuilder()
        sb.append("{")
        sb.append("\"frame_idx\":$frameIdx,")
        sb.append("\"mode\":\"$mode\",")
        sb.append("\"inference_ran\":$inferenceRan,")
        sb.append("\"latency_ms\":${latencyMs ?: "null"},")
        sb.append("\"motion_score\":${motionScore ?: "null"},")
        sb.append("\"is_keyframe\":${isKeyFrame ?: "null"},")
        sb.append("\"dx_px\":${dxPx ?: "null"},")
        sb.append("\"dy_px\":${dyPx ?: "null"},")
        sb.append("\"propagated\":${propagated ?: "null"},")
        sb.append("\"detections\":[")
        detections.forEachIndexed { i, d ->
            if (i > 0) sb.append(",")
            sb.append("{")
            sb.append("\"label\":\"${d.label}\",")
            sb.append("\"score\":${d.score},")
            sb.append("\"xmin\":${d.xmin},")
            sb.append("\"ymin\":${d.ymin},")
            sb.append("\"xmax\":${d.xmax},")
            sb.append("\"ymax\":${d.ymax}")
            sb.append("}")
        }
        sb.append("]}")
        sb.append("\n")
        writer.write(sb.toString())
        writer.flush()
    }


    fun close() {
        writer.flush()
        writer.close()
    }

    fun getFilePath(): String = file.absolutePath
}
