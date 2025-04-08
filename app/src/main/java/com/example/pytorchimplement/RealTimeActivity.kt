package com.example.pytorchimplement

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.camera.view.PreviewView
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.ViewGroup
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.graphics.Path
import android.graphics.DashPathEffect
import android.view.Surface
import android.graphics.Rect
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import com.example.pytorchimplement.BuildConfig

class RealTimeActivity : AppCompatActivity(), ImageAnalyzer.AnalysisListener {

    private lateinit var cameraExecutor: ExecutorService
    private var isFrontCamera = false
    private var imageCapture: ImageCapture? = null
    private var camera: Camera? = null
    private var imageAnalyzer: ImageAnalyzer? = null
    private lateinit var resultTextView: TextView
    private lateinit var detailsTextView: TextView
    private lateinit var boxOverlay: BoxOverlay
    private val lastUIUpdateTime = AtomicLong(0)
    private val uiUpdateInterval = 200L // 5 updates per second

    // Define required permissions based on Android version
    private val REQUIRED_PERMISSIONS = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        // Android 13+ (API 33+)
        arrayOf(
            Manifest.permission.CAMERA
            // READ_MEDIA_IMAGES is only needed if you're accessing the gallery
            // Manifest.permission.READ_MEDIA_IMAGES
        )
    } else
        // Android 10+ (API 29+)
        arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.READ_EXTERNAL_STORAGE
            // WRITE_EXTERNAL_STORAGE doesn't give general write access on API 29+
        )

    // Camera-only permission launcher (primary use case)
    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            Log.d(TAG, "Camera permission granted, starting camera")
            startCamera()
        } else {
            Log.d(TAG, "Camera permission denied")
            Toast.makeText(this, "Camera permission is required for this app", Toast.LENGTH_LONG).show()
        }
    }

    // Multiple permissions launcher (if we need more permissions)
    private val requestPermissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        // Log each permission result
        permissions.forEach { (permission, isGranted) ->
            Log.d(TAG, "Permission $permission: ${if (isGranted) "GRANTED" else "DENIED"}")
        }

        // Check if at least camera permission is granted
        val cameraPermissionGranted = permissions[Manifest.permission.CAMERA] ?: false

        if (cameraPermissionGranted) {
            Log.d(TAG, "Camera permission granted, starting camera")
            startCamera()
        } else {
            Log.d(TAG, "Camera permission denied")
            Toast.makeText(this, "Camera permission is required for this app", Toast.LENGTH_LONG).show()
            showPermissionRationale()
        }
    }

    private val DEBUG = try {
        BuildConfig.DEBUG
    } catch (e: Exception) {
        Log.e(TAG, "Error accessing BuildConfig.DEBUG: ${e.message}")
        false // Default to false (production mode) if BuildConfig isn't available
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_real_time_activity)

        // Set up the result text view
        resultTextView = findViewById<TextView>(R.id.result_text)
        detailsTextView = findViewById<TextView>(R.id.details_text)

        // Make analysis displays visible by default with initial text
        resultTextView.text = "LIVE DETECTION: Starting camera..."
        resultTextView.visibility = View.VISIBLE

        detailsTextView.text = "DETECTED ACNE TYPES:\nWaiting for analysis..."
        detailsTextView.visibility = View.VISIBLE

        // Set up the bounding box overlay
        val previewView = findViewById<PreviewView>(R.id.view_finder)
        boxOverlay = BoxOverlay(this)

        // Enable hardware acceleration for better performance
        boxOverlay.setLayerType(View.LAYER_TYPE_HARDWARE, null)

        // Add BoxOverlay with MATCH_PARENT to ensure it covers the entire preview
        (previewView.parent as ViewGroup).addView(boxOverlay, ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))

        // Add a post layout listener to ensure we get the correct preview size
        previewView.viewTreeObserver.addOnGlobalLayoutListener {
            val width = previewView.width
            val height = previewView.height
            if (width > 0 && height > 0) {
                if (DEBUG) {
                    Log.d(TAG, "PreviewView size changed: ${width}x${height}")
                }
                boxOverlay.setPreviewSize(width, height)
                boxOverlay.setCameraFacing(isFrontCamera)
            }
        }

        // Set up the buttons with proper UI elements
        val severityButton = findViewById<Button>(R.id.redirect_severity)
        val switchCameraButton = findViewById<ImageButton>(R.id.switch_camera)
        switchCameraButton.visibility = View.INVISIBLE

        severityButton.setOnClickListener {
            val intent = Intent(this, SeverityActivity::class.java)
            startActivity(intent)
        }

        switchCameraButton.setOnClickListener {
            isFrontCamera = !isFrontCamera
            boxOverlay.setCameraFacing(isFrontCamera)
            startCamera()
        }

        // Request permissions before starting camera
        if (isCameraPermissionGranted()) {
            Log.d(TAG, "Camera permission already granted, starting camera")
            startCamera()

            // Show instructions Toast
            Toast.makeText(
                this,
                "Real-time acne detection is running with ONNX model.",
                Toast.LENGTH_LONG
            ).show()
        } else {
            Log.d(TAG, "Requesting camera permission")
            // For simplicity, just request camera permission since that's the essential one
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        // Initialize with a larger thread pool for parallel operations
        cameraExecutor = Executors.newFixedThreadPool(2)
    }
    @SuppressLint("MissingSuperCall")
    override fun onBackPressed() {
        // Do Here what ever you want do on back press;
    }
    // Save bitmap to a temporary file
    private fun saveBitmapToTempFile(bitmap: Bitmap): File {
        val cachePath = File(cacheDir, "images")
        cachePath.mkdirs()

        // Create a unique filename based on timestamp
        val fileName = "captured_image_${System.currentTimeMillis()}.jpg"
        val file = File(cachePath, fileName)

        // Compress and save the bitmap to the file
        FileOutputStream(file).use { out ->
            // Use a lower quality (70) to reduce file size
            bitmap.compress(Bitmap.CompressFormat.JPEG, 70, out)
            out.flush()
        }

        return file
    }

    private fun processBitmapForModel(bitmap: Bitmap): Bitmap {
        // Prepare the bitmap for the model
        // If front camera, flip the image horizontally
        if (isFrontCamera) {
            val matrix = Matrix()
            matrix.preScale(-1.0f, 1.0f)
            return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, false)
        }

        return bitmap
    }

    // Check if camera permission is granted
    private fun isCameraPermissionGranted(): Boolean {
        val cameraPermissionGranted = ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        if (!cameraPermissionGranted) {
            Log.d(TAG, "Camera permission not granted")
        }

        return cameraPermissionGranted
    }

    // Check all permissions
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        val isGranted = ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        if (!isGranted) {
            Log.d(TAG, "Permission not granted: $it")
        }
        isGranted
    }

    private fun requestPermissions() {
        requestPermissionsLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private fun showPermissionRationale() {
        Toast.makeText(
            this,
            "Camera permission is required for real-time analysis",
            Toast.LENGTH_LONG
        ).show()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                val previewView = findViewById<PreviewView>(R.id.view_finder)

                // Use moderate resolutions that work well
                val previewResolution = android.util.Size(640, 640)
                val analysisResolution = android.util.Size(640, 640) // Moderate size for reliable detection

                // Configure preview with square aspect ratio
                val preview = Preview.Builder()
                    .setTargetResolution(previewResolution)
                    .setTargetRotation(Surface.ROTATION_0)
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                // For image capture, use the model's preferred size
                imageCapture = ImageCapture.Builder()
                    .setTargetResolution(previewResolution)
                    .setTargetRotation(Surface.ROTATION_0)
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()

                // Basic image analysis setup that works reliably
                val imageAnalysis = ImageAnalysis.Builder()
                    .setTargetResolution(analysisResolution)
                    .setTargetRotation(Surface.ROTATION_0)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build()

                // Initialize the image analyzer
                imageAnalyzer = ImageAnalyzer(this, this)
                // Use executor service for analysis
                imageAnalysis.setAnalyzer(cameraExecutor, imageAnalyzer!!)

                // Select back or front camera
                val cameraSelector = if (isFrontCamera) {
                    CameraSelector.DEFAULT_FRONT_CAMERA
                } else {
                    CameraSelector.DEFAULT_BACK_CAMERA
                }

                // Unbind any existing use cases before rebinding
                cameraProvider.unbindAll()

                // Bind all use cases to camera lifecycle
                camera = cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageCapture,
                    imageAnalysis
                )

                // Add debug info about camera resolution
                if (DEBUG) {
                    val cameraInfo = camera?.cameraInfo
                    if (cameraInfo != null) {
                        Log.d(TAG, "Camera sensor rotation: ${cameraInfo.sensorRotationDegrees}°")
                    }
                }

                // Update overlay with preview size - with better logging
                previewView.post {
                    val width = previewView.width
                    val height = previewView.height

                    if (width > 0 && height > 0) {
                        if (DEBUG) {
                            Log.d(TAG, "PreviewView dimensions: ${width}x${height}, aspect ratio: ${width.toFloat()/height}")
                        }
                        boxOverlay.setPreviewSize(width, height)
                        boxOverlay.setCameraFacing(isFrontCamera)

                        // Update the image analyzer with the guide box information
                        val guideRect = boxOverlay.getModelToScreenRect()
                        imageAnalyzer?.setGuideBoxInfo(
                            guideRect.left.toFloat() / width,
                            guideRect.top.toFloat() / height,
                            guideRect.width().toFloat() / width,
                            guideRect.height().toFloat() / height
                        )

                        // Force a redraw of any existing detections
                        if (imageAnalyzer?.lastAnalysisResult != null) {
                            val lastResult = imageAnalyzer?.lastAnalysisResult
                            if (lastResult != null && lastResult.detections.isNotEmpty()) {
                                boxOverlay.setDetections(lastResult.detections, lastResult.timestamp, lastResult.inferenceTimeMs)
                            }
                        }
                    } else {
                        Log.w(TAG, "PreviewView dimensions not available: ${width}x${height}")
                    }
                }

                // Set up tap to focus
                previewView.setOnTouchListener { view, event ->
                    try {
                        // Get focus MeteringPoint
                        val factory = previewView.meteringPointFactory
                        val point = factory.createPoint(event.x, event.y)

                        // Build focus action
                        val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF)
                            .setAutoCancelDuration(3, java.util.concurrent.TimeUnit.SECONDS)
                            .build()

                            // Execute focus action
                            camera?.cameraControl?.startFocusAndMetering(action)

                            view.performClick()
                            true
                    } catch (e: Exception) {
                        Log.e(TAG, "Cannot focus: ${e.message}")
                        false
                    }
                }

            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
                Toast.makeText(this, "Error starting camera: ${exc.message}", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onAnalysisComplete(result: ImageAnalyzer.AnalysisResult, inferenceTimeMs: Long) {
        // Use a simple approach - filter detections, then update UI
        val filteredDetections = result.detections.filter { detection ->
            val x = detection.boundingBox.x
            val y = detection.boundingBox.y

            // Basic range check
            x >= 0 && x <= 1 && y >= 0 && y <= 1
        }

        // Simple throttling based on fixed interval
        val currentTime = System.currentTimeMillis()
        val timeSinceLastUpdate = currentTime - lastUIUpdateTime.get()

        // Update if it's been at least 100ms since last update
        if (timeSinceLastUpdate >= 100) {
            // Update overlay with filtered detections
            boxOverlay.setDetections(filteredDetections, result.timestamp, inferenceTimeMs)

            // Update UI text
            runOnUiThread {
                // Update header text with detection status
                resultTextView.text = "LIVE DETECTION: Acne Classification"

                // Create the text for acne counts
                val countsText = StringBuilder("DETECTED ACNE TYPES:\n")
                var totalCount = 0
//
                // Display counts for each acne type
                result.acneCounts.forEach { (type, count) ->
                    if (count > 0) {
                        countsText.append("• ${type.capitalize()}\n")
                        totalCount += count
                    }
                }

                // Add total count and processing time
                countsText.append("\nTIME: ${inferenceTimeMs}ms")

                detailsTextView.text = countsText.toString()
//            }
//
//            // Update the timestamp
//            lastUIUpdateTime.set(currentTime)
        }
            }
        // Only log in debug mode
        if (DEBUG) {
            Log.d(TAG, "Frame processed: Acne Types=${result.acneCounts}, " +
                    "Total Detections=${filteredDetections.size}, " +
                    "Time=${inferenceTimeMs}ms")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        imageAnalyzer?.close()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "RealTimeActivity"
    }

    /**
     * Custom view for drawing bounding boxes overlay on the camera preview
     */
    inner class BoxOverlay(context: Context) : SurfaceView(context), SurfaceHolder.Callback {
        // Pre-allocated Paint objects for better performance
        private val paint = Paint().apply {
            isAntiAlias = true
            style = Paint.Style.STROKE
            strokeWidth = 10f
        }

        private val textPaint = Paint().apply {
            isAntiAlias = true
            style = Paint.Style.FILL
            color = Color.WHITE
            textSize = 52f
        }

        private val backgroundPaint = Paint().apply {
            style = Paint.Style.FILL
            color = Color.parseColor("#80000000") // Semi-transparent black
        }

        private val debugPaint = Paint().apply {
            style = Paint.Style.FILL
            color = Color.YELLOW
            textSize = 36f
        }

        private val guidePaint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeWidth = 8f
            pathEffect = DashPathEffect(floatArrayOf(50f, 30f), 0f)
            alpha = 150 // Semi-transparent
        }

        // Pre-allocated Paint objects for different acne classes for better performance
        private val acneClassColors = mapOf(
            "comedone" to Paint().apply {
                color = Color.YELLOW
                style = Paint.Style.STROKE
                strokeWidth = 10f
                isAntiAlias = true
            },
            "pustule" to Paint().apply {
                color = Color.RED
                style = Paint.Style.STROKE
                strokeWidth = 10f
                isAntiAlias = true
            },
            "papule" to Paint().apply {
                color = Color.MAGENTA
                style = Paint.Style.STROKE
                strokeWidth = 10f
                isAntiAlias = true
            },
            "nodule" to Paint().apply {
                color = Color.GREEN
                style = Paint.Style.STROKE
                strokeWidth = 10f
                isAntiAlias = true
            }
        )

        private val acneClassFillColors = mapOf(
            "comedone" to Paint().apply {
                color = Color.YELLOW
                style = Paint.Style.FILL
                alpha = 60
            },
            "pustule" to Paint().apply {
                color = Color.RED
                style = Paint.Style.FILL
                alpha = 60
            },
            "papule" to Paint().apply {
                color = Color.MAGENTA
                style = Paint.Style.FILL
                alpha = 60
            },
            "nodule" to Paint().apply {
                color = Color.GREEN
                style = Paint.Style.FILL
                alpha = 60
            }
        )

        private var detections: List<ImageAnalyzer.Detection> = emptyList()
        private var previewWidth = 0
        private var previewHeight = 0
        private var isFrontCamera = false
        private var lastDrawTime = 0L
        private var processTimeMs = 0L
        private var frameTimestamp = 0L
        private var drawingEnabled = true

        // Frame rate control
        private val targetDrawInterval = 33L // ~30fps

        // Caching for better performance
        private var cachedGuideRect: Rect? = null
        private var lastPreviewWidth = 0
        private var lastPreviewHeight = 0

        // Model dimensions - these match what we set in the camera configuration
        private val MODEL_WIDTH = 640
        private val MODEL_HEIGHT = 640

        init {
            setZOrderOnTop(true)
            holder.setFormat(android.graphics.PixelFormat.TRANSPARENT)
            holder.addCallback(this)
            setWillNotDraw(false) // Ensure onDraw is called

            // Enable hardware acceleration for better performance
            setLayerType(View.LAYER_TYPE_HARDWARE, null)

            // Log initialization
            Log.d(TAG, "BoxOverlay initialized")
        }



        fun setDetections(newDetections: List<ImageAnalyzer.Detection>, timestamp: Long, inferenceTimeMs: Long = 0) {
            // Keep this simple - just log and update
            Log.d(TAG, "Received ${newDetections.size} detections to draw")

            detections = newDetections
            frameTimestamp = timestamp
            processTimeMs = inferenceTimeMs
            lastDrawTime = System.currentTimeMillis()


            // Force a redraw
            try {
                drawDetections(true)
            } catch (e: Exception) {
                Log.e(TAG, "Error drawing detections: ${e.message}")
            }

            // Also request a redraw through the View system
            invalidate()
            postInvalidate()
        }

        fun setPreviewSize(width: Int, height: Int) {
            Log.d(TAG, "BoxOverlay size set to ${width}x${height}")
            previewWidth = width
            previewHeight = height

            // Clear cached guide rect when size changes
            if (width != lastPreviewWidth || height != lastPreviewHeight) {
                cachedGuideRect = null
            }
        }

        fun setCameraFacing(front: Boolean) {
            isFrontCamera = front
            // Clear cached guide rect when camera switches
            cachedGuideRect = null
        }

        // Override onDraw as a backup drawing mechanism
        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)

            // If detections are available but surface drawing failed, try drawing here
            if (detections.isNotEmpty()) {
                try {
                    // Draw guide box
                    drawFaceGuide(canvas)

                    // Draw detections
                    detections.forEach { detection ->
                        drawBoxForDetection(canvas, detection)
                    }

                    Log.d(TAG, "Drew ${detections.size} detections via onDraw")
                } catch (e: Exception) {
                    Log.e(TAG, "Error in onDraw: ${e.message}")
                }
            }
        }

        private fun drawDetections(forceRedraw: Boolean = false) {
            // Skip if drawing is disabled
            if (!drawingEnabled) {
                Log.d(TAG, "Drawing is disabled, skipping")
                return
            }

            // Throttle drawing to target frame rate, unless forced
            val currentTime = System.currentTimeMillis()
            if (!forceRedraw && currentTime - lastDrawTime < targetDrawInterval) {
                return
            }

            if (!holder.surface.isValid) {
                Log.d(TAG, "Cannot draw - surface is not valid")
                return
            }

            val canvas = holder.lockCanvas() ?: return
            try {
                // Clear the canvas
                canvas.drawColor(Color.TRANSPARENT, android.graphics.PorterDuff.Mode.CLEAR)

                // Always draw the face guide frame
                drawFaceGuide(canvas)

                // Draw each detection box
                if (detections.isEmpty()) {
                    // Skip drawing "waiting" text for cleaner UI
                } else {
                    // Draw all detections
                    detections.forEach { detection ->
                        drawBoxForDetection(canvas, detection)
                    }

//                    // Draw a simplified info panel in the top corner
//                    val infoBackgroundPaint = Paint().apply {
//                        style = Paint.Style.FILL
//                        color = Color.parseColor("#AA000000") // Semi-transparent black
//                    }
//
//                    // Prepare text
//                    val countText = "${detections.size} detections"
//                    val timeText = "${processTimeMs}ms"
//
//
//                    textPaint.textSize = 40f
//                    val textWidth = Math.max(
//                        textPaint.measureText(countText),
//                        textPaint.measureText(timeText)
//                    ) + 20f
//                    val textHeight = textPaint.textSize * 2 + 20f
//
//                    // Draw info panel background
//                    canvas.drawRect(
//                        20f, 20f,
//                        20f + textWidth,
//                        20f + textHeight,
//                        infoBackgroundPaint
//                    )
//
//                    // Draw text
//                    textPaint.color = Color.WHITE
//                    canvas.drawText(countText, 30f, 20f + textPaint.textSize, textPaint)
//                    canvas.drawText(timeText, 30f, 20f + textPaint.textSize * 2, textPaint)

                    if (DEBUG) {
                        Log.d(TAG, "Successfully drew ${detections.size} detection boxes to surface")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error drawing to surface: ${e.message}")
                e.printStackTrace()
            } finally {
                try {
                    holder.unlockCanvasAndPost(canvas)
                    lastDrawTime = currentTime
                } catch (e: Exception) {
                    Log.e(TAG, "Error posting canvas: ${e.message}")
                }
            }
        }

        private fun drawFaceGuide(canvas: Canvas) {
            try {
                // Get the guide rect that we're using for detection
                val guideRect = getModelToScreenRect()

                // Extract coordinates
                val frameX = guideRect.left
                val frameY = guideRect.top
                val frameSize = guideRect.width()

                // Draw outer dashed frame with improved visibility
                guidePaint.color = Color.WHITE
                guidePaint.strokeWidth = 6f
                guidePaint.pathEffect = DashPathEffect(floatArrayOf(40f, 20f), 0f)
                guidePaint.alpha = 200 // More visible

                canvas.drawRect(
                    frameX.toFloat(),
                    frameY.toFloat(),
                    (frameX + frameSize).toFloat(),
                    (frameY + frameSize).toFloat(),
                    guidePaint
                )

                // Draw corner markers for better alignment guidance
                val cornerSize = frameSize * 0.1f
                val cornerPaint = Paint().apply {
                    color = Color.WHITE
                    style = Paint.Style.STROKE
                    strokeWidth = 8f
                    alpha = 255 // Fully opaque for better visibility
                }

                // Top-left corner
                canvas.drawLine(frameX.toFloat(), frameY.toFloat(),
                                frameX.toFloat() + cornerSize, frameY.toFloat(), cornerPaint)
                canvas.drawLine(frameX.toFloat(), frameY.toFloat(),
                                frameX.toFloat(), frameY.toFloat() + cornerSize, cornerPaint)

                // Top-right corner
                canvas.drawLine((frameX + frameSize).toFloat(), frameY.toFloat(),
                                (frameX + frameSize).toFloat() - cornerSize, frameY.toFloat(), cornerPaint)
                canvas.drawLine((frameX + frameSize).toFloat(), frameY.toFloat(),
                                (frameX + frameSize).toFloat(), frameY.toFloat() + cornerSize, cornerPaint)
            } catch (e: Exception) {
                Log.e(TAG, "Error drawing face guide: ${e.message}")
            }
        }

        fun getModelToScreenRect(): Rect {
            // Return cached rect if dimensions haven't changed
            if (cachedGuideRect != null &&
                previewWidth == lastPreviewWidth &&
                previewHeight == lastPreviewHeight) {
                return cachedGuideRect!!
            }

            // Calculate a square area that matches the model's input dimensions
            val boxSize = Math.min(previewWidth, previewHeight) * 0.9f
            val left = (previewWidth - boxSize) / 2
            val top = (previewHeight - boxSize) / 2

            cachedGuideRect = Rect(
                left.toInt(),
                top.toInt(),
                (left + boxSize).toInt(),
                (top + boxSize).toInt()
            )

            lastPreviewWidth = previewWidth
            lastPreviewHeight = previewHeight

            Log.d(TAG, "Created guide frame: ${cachedGuideRect!!.width()}x${cachedGuideRect!!.height()} at (${cachedGuideRect!!.left},${cachedGuideRect!!.top})")

            return cachedGuideRect!!
        }

        private fun drawBoxForDetection(canvas: Canvas, detection: ImageAnalyzer.Detection) {
            try {
                // Get the guide box area
                val guideRect = getModelToScreenRect()

                // Get normalized coordinates (0-1)
                val normX = detection.boundingBox.x
                val normY = detection.boundingBox.y
                val normWidth = detection.boundingBox.width
                val normHeight = detection.boundingBox.height

                // Convert to screen coordinates
                val screenX = guideRect.left + (normX * guideRect.width())
                val screenY = guideRect.top + (normY * guideRect.height())
                val screenWidth = normWidth * guideRect.width()
                val screenHeight = normHeight * guideRect.height()

                // Default min box size
                val minBoxSize = guideRect.width() * 0.05f
                val finalWidth = Math.max(screenWidth, minBoxSize)
                val finalHeight = Math.max(screenHeight, minBoxSize)

                // POSITION CORRECTION: Add a leftward offset to fix the rightward skew
                // This value may need to be tuned based on testing
                val offsetCorrection = guideRect.width() * 0.05f  // 5% of guide width leftward shift

                // Calculate box coordinates with correction
                var left = (screenX - finalWidth / 2) - offsetCorrection
                var top = screenY - finalHeight / 2
                var right = (screenX + finalWidth / 2) - offsetCorrection
                var bottom = screenY + finalHeight / 2

                // Apply mirroring for front camera if needed
                if (isFrontCamera) {
                    val oldLeft = left
                    left = previewWidth - right
                    right = previewWidth - oldLeft
                }

                // Get paint objects based on class
                val className = detection.className.lowercase()
                val basePaint = when {
                    className.contains("comedone") -> acneClassColors["comedone"]
                    className.contains("pustule") -> acneClassColors["pustule"]
                    className.contains("papule") -> acneClassColors["papule"]
                    className.contains("nodule") -> acneClassColors["nodule"]
                    else -> acneClassColors["comedone"] // default
                } ?: paint

                // Get fill paint
                val fillPaint = when {
                    className.contains("comedone") -> acneClassFillColors["comedone"]
                    className.contains("pustule") -> acneClassFillColors["pustule"]
                    className.contains("papule") -> acneClassFillColors["papule"]
                    className.contains("nodule") -> acneClassFillColors["nodule"]
                    else -> acneClassFillColors["comedone"] // default
                } ?: Paint().apply {
                    style = Paint.Style.FILL
                    color = Color.YELLOW
                    alpha = 60
                }

                // Draw filled area
                canvas.drawRect(left, top, right, bottom, fillPaint)

                // Draw outline
                canvas.drawRect(left, top, right, bottom, basePaint)

                // Display class name and confidence
                val confidence = (detection.confidence * 100).toInt()

                // Simplified class name
                val shortClassName = when {
                    className.contains("comedone") -> "Comedone"
                    className.contains("pustule") -> "Pustule"
                    className.contains("papule") -> "Papule"
                    className.contains("nodule") -> "Nodule"
                    else -> className
                }

                val labelText = "$shortClassName ${confidence}%"
                textPaint.textSize = 30f
                val textWidth = textPaint.measureText(labelText) + 10f
                val textHeight = textPaint.textSize + 5f

                // Draw label background
                canvas.drawRect(left, top - textHeight, left + textWidth, top, backgroundPaint)

                // Draw label text
                canvas.drawText(labelText, left + 5f, top - 5f, textPaint)
            } catch (e: Exception) {
                Log.e(TAG, "Error drawing detection box: ${e.message}")
            }
        }

        override fun surfaceCreated(holder: SurfaceHolder) {
            Log.d(TAG, "BoxOverlay surface created")
            drawingEnabled = true

            // Force a redraw when surface is created
            post {
                try {
                    drawDetections(true)
                } catch (e: Exception) {
                    Log.e(TAG, "Error drawing on surface creation: ${e.message}")
                }
            }
        }

        override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
            Log.d(TAG, "BoxOverlay surface changed: ${width}x${height}")
            if (width > 0 && height > 0) {
                previewWidth = width
                previewHeight = height
                // Clear cached guide rect when surface changes
                cachedGuideRect = null

                // Try to redraw on surface change
                post {
                    try {
                        drawDetections(true)
                    } catch (e: Exception) {
                        Log.e(TAG, "Error drawing on surface change: ${e.message}")
                    }
                }
            }
        }

        override fun surfaceDestroyed(holder: SurfaceHolder) {
            Log.d(TAG, "BoxOverlay surface destroyed")
            drawingEnabled = false
        }

        // Helper function to format floats nicely
        private fun Float.format(digits: Int) = "%.${digits}f".format(this)
    }
}


