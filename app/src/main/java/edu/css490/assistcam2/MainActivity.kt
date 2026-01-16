package edu.css490.assistcam2

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import edu.css490.assistcam2.databinding.ActivityMainBinding
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val cameraPermission = Manifest.permission.CAMERA
    private val requestCodeCamera = 42
    private lateinit var analysisExecutor: ExecutorService

    private lateinit var detectorHelper: DetectorHelper

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        analysisExecutor = Executors.newSingleThreadExecutor()
        detectorHelper = DetectorHelper(this)

        // CAMERA demo
        // if (ContextCompat.checkSelfPermission(this, cameraPermission) == PackageManager.PERMISSION_GRANTED) {
        //     startCamera()
        // } else {
        //     ActivityCompat.requestPermissions(this, arrayOf(cameraPermission), requestCodeCamera)
        // }

        // OFFLINE eval
        runOfflineEval(Mode.SKIP_MOTION) // BASELINE_FULL / SKIP_MOTION / SKIP_MOTION_TRACK
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    val mode = Mode.SKIP_MOTION // BASELINE_FULL / SKIP_MOTION / SKIP_MOTION_TRACK

                    val motionGate =
                        if (mode == Mode.SKIP_MOTION || mode == Mode.SKIP_MOTION_TRACK) {
                            MotionGate(
                                gateW = 64, gateH = 36,
                                transW = 160, transH = 90,
                                alpha = 0.2f,
                                threshold = 5.0f,
                                maxSkipFrames = 8,
                                maxShiftTrans = 8f,
                            )
                        } else null

                    it.setAnalyzer(
                        analysisExecutor,
                        FrameAnalyzer(
                            detectorHelper = detectorHelper,
                            updateStats = { text ->
                                runOnUiThread { binding.fpsText.text = text }
                            },
                            mode = mode,
                            motionGate = motionGate,
                            onDetections = { dets, imgW, imgH ->
                                runOnUiThread {
                                    binding.detectionOverlay.setDetections(dets, imgW, imgH)
                                }
                            }
                        )
                    )
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )
            } catch (exc: Exception) {
                Log.e("AssistCam", "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun runOfflineEval(mode: Mode) {
        Thread {
            val logger = EvalLogger(this, "eval_${mode.name}.jsonl")

            val root = getExternalFilesDir("eval_frames")
            if (root == null) {
                Log.e("Eval", "External files dir is null")
                return@Thread
            }

            val seqDir = File(root, "seq01")
            if (!seqDir.exists() || !seqDir.isDirectory) {
                Log.e("Eval", "Sequence dir not found: ${seqDir.absolutePath}")
                return@Thread
            }

            val frameFiles = seqDir.listFiles { f ->
                f.isFile && (f.extension.equals("jpg", true) || f.extension.equals("png", true))
            }?.sortedBy { it.name } ?: emptyList()

            if (frameFiles.isEmpty()) {
                Log.e("Eval", "No frames found in ${seqDir.absolutePath}")
                return@Thread
            }

            // MotionGate only needed for SKIP_MOTION / SKIP_MOTION_TRACK
            val motionGate = if (mode == Mode.SKIP_MOTION || mode == Mode.SKIP_MOTION_TRACK) {
                MotionGate(
                    gateW = 64, gateH = 36,
                    transW = 160, transH = 90,
                    alpha = 0.2f,
                    threshold = 5.0f,
                    maxSkipFrames = 8,
                    maxShiftTrans = 8f,
                )

            } else null

            fun shiftDetections(
                dets: List<Detection>,
                dx: Float,
                dy: Float,
                w: Int,
                h: Int
            ): List<Detection> {
                val maxX = (w - 1).toFloat().coerceAtLeast(0f)
                val maxY = (h - 1).toFloat().coerceAtLeast(0f)

                return dets.mapNotNull { d ->
                    val x1 = (d.xmin + dx).coerceIn(0f, maxX)
                    val y1 = (d.ymin + dy).coerceIn(0f, maxY)
                    val x2 = (d.xmax + dx).coerceIn(0f, maxX)
                    val y2 = (d.ymax + dy).coerceIn(0f, maxY)
                    if (x2 <= x1 || y2 <= y1) null else d.copy(xmin = x1, ymin = y1, xmax = x2, ymax = y2)
                }
            }

            Log.i("Eval", "Running eval for mode=$mode on ${frameFiles.size} frames")

            var frameIdx = 0
            var inferenceCalls = 0
            val latenciesMs = mutableListOf<Long>()

            // trackedDetections = what we will log/show for each frame
            var trackedDetections: List<Detection> = emptyList()

            val startTimeNs = System.nanoTime()

            for (file in frameFiles) {
                frameIdx++

                val bitmap = BitmapFactory.decodeFile(file.absolutePath)
                if (bitmap == null) {
                    Log.w("Eval", "Failed to decode ${file.absolutePath}, skipping")
                    continue
                }

                val w = bitmap.width
                val h = bitmap.height

                var motionScore: Float? = null   // we log EMA here (stable gating score)
                var isKeyFrame: Boolean? = null
                var dxPx: Float? = null
                var dyPx: Float? = null
                var propagated: Boolean? = false

                // Decide inference
                val needInference: Boolean = when (mode) {
                    Mode.BASELINE_FULL -> {
                        isKeyFrame = true
                        true
                    }

                    Mode.SKIP_K2 -> {
                        val kf = (frameIdx % 2 == 0)
                        isKeyFrame = kf
                        kf
                    }

                    Mode.SKIP_MOTION, Mode.SKIP_MOTION_TRACK -> {
                        val gate = motionGate!!
                        val d = gate.update(bitmap)
                        motionScore = d.emaMotion
                        isKeyFrame = d.runInference
                        dxPx = d.dxPx
                        dyPx = d.dyPx
                        d.runInference
                    }
                }

                var latencyMs: Long? = null

                if (needInference) {
                    val t0 = System.nanoTime()
                    trackedDetections = detectorHelper.detect(bitmap)
                    val t1 = System.nanoTime()

                    latencyMs = (t1 - t0) / 1_000_000
                    latenciesMs.add(latencyMs)
                    inferenceCalls++

                    propagated = false
                } else {
                    // No inference: reuse (SKIP_MOTION) or shift-propagate (SKIP_MOTION_TRACK)
                    propagated = false

                    if (mode == Mode.SKIP_MOTION_TRACK && dxPx != null && dyPx != null && trackedDetections.isNotEmpty()) {
                        val dx = dxPx!!
                        val dy = dyPx!!

                        // Only call it "propagated" if the shift is meaningfully non-zero
                        val eps = 0.5f
                        if (kotlin.math.abs(dx) + kotlin.math.abs(dy) > eps) {
                            trackedDetections = shiftDetections(trackedDetections, dx, dy, w, h)
                            propagated = true
                        }
                    }
                }

                logger.logFrame(
                    frameIdx = frameIdx,
                    mode = mode,
                    inferenceRan = needInference,
                    latencyMs = latencyMs,
                    detections = trackedDetections,
                    motionScore = motionScore,
                    isKeyFrame = isKeyFrame,
                    dxPx = dxPx,
                    dyPx = dyPx,
                    propagated = propagated
                )
            }

            logger.close()

            val elapsedSec = (System.nanoTime() - startTimeNs) / 1_000_000_000.0
            val fps = if (elapsedSec > 0) frameIdx / elapsedSec else 0.0
            val callsPerSec = if (elapsedSec > 0) inferenceCalls / elapsedSec else 0.0

            val stats = "mode=$mode frames=$frameIdx calls=$inferenceCalls fps=%.2f calls/s=%.2f"
                .format(fps, callsPerSec)

            Log.i("Eval", stats)
            Log.i("Eval", "Eval log written to: ${logger.getFilePath()}")

            runOnUiThread {
                binding.fpsText.text = stats
            }
        }.start()
    }
}
