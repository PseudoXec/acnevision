package com.example.pytorchimplement

import android.Manifest
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

class RealTimeActivity : AppCompatActivity(), ImageAnalyzer.AnalysisListener {

    private lateinit var cameraExecutor: ExecutorService
    private var isFrontCamera = false
    private var imageCapture: ImageCapture? = null
    private var camera: Camera? = null
    private var imageAnalyzer: ImageAnalyzer? = null
    private lateinit var resultTextView: TextView
    private lateinit var detailsTextView: TextView
    private lateinit var boxOverlay: BoxOverlay

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
                Log.d(TAG, "PreviewView size changed: ${width}x${height}")
                boxOverlay.setPreviewSize(width, height)
                boxOverlay.setCameraFacing(isFrontCamera)
            }
        }
        
        // Set up the buttons with proper UI elements
        val severityButton = findViewById<Button>(R.id.redirect_severity)
        val switchCameraButton = findViewById<ImageButton>(R.id.switch_camera)

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

        cameraExecutor = Executors.newSingleThreadExecutor()
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

                // Get display metrics to understand device screen dimensions
                val displayMetrics = resources.displayMetrics
                Log.d(TAG, "Device screen size: ${displayMetrics.widthPixels}x${displayMetrics.heightPixels}")

                // Force square aspect ratio for preview to match model input
                val targetResolution = android.util.Size(640, 640)
                
                // Configure preview with square aspect ratio
                val preview = Preview.Builder()
                    .setTargetResolution(targetResolution)
                    .setTargetRotation(Surface.ROTATION_0)
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                // For image capture, use the model's preferred size
                imageCapture = ImageCapture.Builder()
                    .setTargetResolution(targetResolution)
                    .setTargetRotation(Surface.ROTATION_0)
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()

                // For analysis, use the exact model input dimensions
                val imageAnalysis = ImageAnalysis.Builder()
                    .setTargetResolution(targetResolution)
                    .setTargetRotation(Surface.ROTATION_0)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    
                // Initialize the image analyzer
                imageAnalyzer = ImageAnalyzer(this, this)
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
                val cameraInfo = camera?.cameraInfo
                if (cameraInfo != null) {
                    Log.d(TAG, "Camera sensor rotation: ${cameraInfo.sensorRotationDegrees}°")
                }
                
                // Update overlay with preview size - with better logging
                previewView.post {
                    val width = previewView.width
                    val height = previewView.height
                    
                    if (width > 0 && height > 0) {
                        Log.d(TAG, "PreviewView dimensions: ${width}x${height}, aspect ratio: ${width.toFloat()/height}")
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
                                boxOverlay.setDetections(lastResult.detections)
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
        // Since the detections are now already relative to the guide box,
        // we only need to filter out any that might be outside the 0-1 range
        val filteredDetections = result.detections.filter { detection ->
            val x = detection.boundingBox.x
            val y = detection.boundingBox.y
            
            // Ensure the detection is within the normalized range (0-1)
            x >= 0 && x <= 1 && y >= 0 && y <= 1
        }
        
        // Update UI with filtered detections
        boxOverlay.setDetections(filteredDetections, inferenceTimeMs)
        
        // Update UI with the analysis result
        runOnUiThread {
            // Update header text with detection status
            resultTextView.text = "LIVE DETECTION: Acne Classification"
            
            // Update acne counts with more detail
            val countsText = StringBuilder("DETECTED ACNE TYPES:\n")
            var totalCount = 0
            
            // Display counts for each acne type
            result.acneCounts.forEach { (type, count) ->
                if (count > 0) {
                    countsText.append("• ${type.capitalize()}: $count\n")
                    totalCount += count
                }
            }
            
            // Add total count
            countsText.append("\nTOTAL ACNE DETECTED: $totalCount")
            
            // Add detection info
            if (filteredDetections.isNotEmpty()) {
                countsText.append("\n\nDetections: ${filteredDetections.size}")
                
                // Show more details about some detections (limit to 3 for readability)
                if (filteredDetections.size <= 3) {
                    countsText.append("\n\nDetailed detections:")
                    filteredDetections.forEachIndexed { index, detection ->
                        val confidence = (detection.confidence * 100).toInt()
                        countsText.append("\n${index+1}. ${detection.className} ($confidence%)")
                    }
                }
            } else {
                countsText.append("\n\nNo detections found")
            }
            
            detailsTextView.text = countsText.toString()
            
            // Log the analysis results
            Log.d(TAG, "Frame processed: Acne Types=${result.acneCounts}, " +
                    "Total=$totalCount, Detections=${filteredDetections.size}")
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
        private val paint = Paint().apply {
            isAntiAlias = true
            style = Paint.Style.STROKE
            strokeWidth = 10f // Increased thickness for better visibility
        }
        
        private val textPaint = Paint().apply {
            isAntiAlias = true
            style = Paint.Style.FILL
            color = Color.WHITE
            textSize = 52f // Larger text for better visibility
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
        
        private var detections: List<ImageAnalyzer.Detection> = emptyList()
        private var previewWidth = 0
        private var previewHeight = 0
        private var isFrontCamera = false
        private var lastDrawTime = 0L
        private var processingTimeMs = 0L
        
        // Model dimensions - these match what we set in the camera configuration
        private val MODEL_WIDTH = 640
        private val MODEL_HEIGHT = 640
        
        init {
            setZOrderOnTop(true)
            holder.setFormat(android.graphics.PixelFormat.TRANSPARENT)
            holder.addCallback(this)
            setWillNotDraw(false) // Ensure onDraw is called
        }

        fun setDetections(newDetections: List<ImageAnalyzer.Detection>, inferenceTimeMs: Long = 0) {
            detections = newDetections

            processingTimeMs = inferenceTimeMs
            lastDrawTime = System.currentTimeMillis()

            // Add debug logs
            Log.d(TAG, "Setting ${detections.size} detections to draw on preview size ${previewWidth}x${previewHeight}")

            // Only log details if there are just a few detections
            if (detections.size <= 5) {
                detections.forEachIndexed { index, detection ->
                    Log.d(TAG, "  - Detection $index: ${detection.className} (${detection.confidence * 100}%), " +
                          "box=[${detection.boundingBox.x.format(3)}, ${detection.boundingBox.y.format(3)}, " +
                          "${detection.boundingBox.width.format(3)}, ${detection.boundingBox.height.format(3)}]")
                }
            }

            // Force immediate redraw
            invalidate()

            // Also draw on the surface if it's valid
            drawDetections()
        }

        fun setPreviewSize(width: Int, height: Int) {
            Log.d(TAG, "BoxOverlay size set to ${width}x${height}")
            previewWidth = width
            previewHeight = height
        }
        
        fun setCameraFacing(front: Boolean) {
            isFrontCamera = front
        }
        
        private fun drawDetections() {
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
                    // Draw "No detections" text for debug
                    textPaint.textSize = 60f
                    canvas.drawText("Waiting for detections...", 50f, 120f, textPaint)

                    debugPaint.textSize = 36f
                    canvas.drawText("Processing time: ${processingTimeMs}ms", 50f, 180f, debugPaint)

                    // Draw the area where detections would appear with aspect ratio correction
                    if (previewWidth > 0 && previewHeight > 0) {
                        // Show the effective area where boxes would be drawn (accounting for aspect ratio)
                        val aspectRatioPaint = Paint().apply {
                            style = Paint.Style.STROKE
                            strokeWidth = 2f
                            color = Color.CYAN
                        }
                        
                        // Draw a rectangle showing the target 640x640 region mapped to screen
                        // drawModelAreaOutline(canvas, aspectRatioPaint)
                    }
                } else {
                    detections.forEach { detection ->
                        drawBoxForDetection(canvas, detection)
                    }
                    
                    // Draw detection count at the top
                    textPaint.textSize = 60f
                    val countText = "Found: ${detections.size} detections"
                    canvas.drawText(countText, 50f, 120f, textPaint)
                    
                    // Draw timestamp and preview info
                    debugPaint.textSize = 30f
                    val timestamp = System.currentTimeMillis()
                    canvas.drawText("Processing time: ${processingTimeMs}ms", 50f, 180f, debugPaint)
                }
            } finally {
                holder.unlockCanvasAndPost(canvas)
            }
        }

        private fun drawFaceGuide(canvas: Canvas) {
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
        }

        private fun drawModelAreaOutline(canvas: Canvas, paint: Paint) {
            val visibleModelRect = getVisibleModelRect()

            if (visibleModelRect.width() == 0 || visibleModelRect.height() == 0) {
                Log.e(TAG, "Error: visibleModelRect has zero width or height!")
                return
            }

            Log.d(TAG, "Drawing Model Area: Left=${visibleModelRect.left}, Top=${visibleModelRect.top}, Right=${visibleModelRect.right}, Bottom=${visibleModelRect.bottom}")

            // Draw the outline rectangle
            canvas.drawRect(
                visibleModelRect.left.toFloat(),
                visibleModelRect.top.toFloat(),
                visibleModelRect.right.toFloat(),
                visibleModelRect.bottom.toFloat(),
                paint
            )

            // Add debug marker lines
            val markerSize = 40f  // Increased size for better visibility

            // Top-left marker
            canvas.drawLine(
                visibleModelRect.left.toFloat(),
                visibleModelRect.top.toFloat(),
                visibleModelRect.left.toFloat() + markerSize,
                visibleModelRect.top.toFloat(),
                paint
            )
            canvas.drawLine(
                visibleModelRect.left.toFloat(),
                visibleModelRect.top.toFloat(),
                visibleModelRect.left.toFloat(),
                visibleModelRect.top.toFloat() + markerSize,
                paint
            )

            // Bottom-right marker
            canvas.drawLine(
                visibleModelRect.right.toFloat(),
                visibleModelRect.bottom.toFloat(),
                visibleModelRect.right.toFloat() - markerSize,
                visibleModelRect.bottom.toFloat(),
                paint
            )
            canvas.drawLine(
                visibleModelRect.right.toFloat(),
                visibleModelRect.bottom.toFloat(),
                visibleModelRect.right.toFloat(),
                visibleModelRect.bottom.toFloat() - markerSize,
                paint
            )

            // Add dimensions text
            val dimensionsText = "${visibleModelRect.width()}x${visibleModelRect.height()}"
            debugPaint.color = Color.CYAN
            debugPaint.textSize = 50f  // Make text more readable
            canvas.drawText(
                dimensionsText,
                visibleModelRect.left.toFloat() + 20,
                visibleModelRect.top.toFloat() + 50,
                debugPaint
            )

            Log.d(TAG, "Model Area Drawn: $dimensionsText")
        }

        fun getModelToScreenRect(): Rect {
            // Calculate a square area that matches the model's input dimensions
            val boxSize = Math.min(previewWidth, previewHeight) * 0.9f
            val left = (previewWidth - boxSize) / 2
            val top = (previewHeight - boxSize) / 2
            
            val rect = Rect(
                left.toInt(), 
                top.toInt(), 
                (left + boxSize).toInt(), 
                (top + boxSize).toInt()
            )
            
            Log.d(TAG, "Face guide frame: ${rect.width()}x${rect.height()} at (${rect.left},${rect.top})")
            return rect
        }

        private fun drawBoxForDetection(canvas: Canvas, detection: ImageAnalyzer.Detection) {
            try {
                // Get the guide box area
                val guideRect = getModelToScreenRect()
                
                // Get normalized coordinates (0-1) - these are now relative to the guide box
                val normX = detection.boundingBox.x
                val normY = detection.boundingBox.y
                val normWidth = detection.boundingBox.width
                val normHeight = detection.boundingBox.height
                
                // Log the normalized coordinates for debugging
                Log.d(TAG, "Drawing detection: ${detection.className} at normalized coords: " +
                      "x=${normX.format(3)}, y=${normY.format(3)}, " +
                      "w=${normWidth.format(3)}, h=${normHeight.format(3)}")

                // Convert normalized coordinates (0-1) directly to screen pixels within the guide box
                // Since the normalized coordinates are now relative to the guide box, this is simpler
                val screenX = guideRect.left + (normX * guideRect.width())
                val screenY = guideRect.top + (normY * guideRect.height())
                val screenWidth = normWidth * guideRect.width()
                val screenHeight = normHeight * guideRect.height()
                
                // Log the screen coordinates for debugging
                Log.d(TAG, "Mapped to screen coords: x=${screenX.toInt()}, y=${screenY.toInt()}, " +
                      "w=${screenWidth.toInt()}, h=${screenHeight.toInt()}")

                // Ensure the bounding box has a minimum size for visibility
                val minBoxSize = Math.min(previewWidth, previewHeight) * 0.03f
                val finalWidth = Math.max(screenWidth, minBoxSize)
                val finalHeight = Math.max(screenHeight, minBoxSize)

                // Calculate the box corners
                var left = screenX - finalWidth / 2
                var top = screenY - finalHeight / 2
                var right = screenX + finalWidth / 2
                var bottom = screenY + finalHeight / 2

                // Apply mirroring correction for front camera
                if (isFrontCamera) {
                    val oldLeft = left
                    left = previewWidth - right
                    right = previewWidth - oldLeft
                }

                // Set bounding box color based on detected class
                val boxColor = when {
                    detection.className.contains("comedone") -> Color.YELLOW
                    detection.className.contains("pustule") -> Color.RED
                    detection.className.contains("papule") -> Color.MAGENTA
                    detection.className.contains("nodule") -> Color.GREEN
                    else -> Color.WHITE
                }

                // Draw bounding box with improved visibility
                paint.color = boxColor
                paint.strokeWidth = 10f  // Thicker for better visibility
                paint.style = Paint.Style.STROKE
                canvas.drawRect(left, top, right, bottom, paint)
                
                // Draw a semi-transparent fill for better visibility
                val fillPaint = Paint().apply {
                    color = boxColor
                    style = Paint.Style.FILL
                    alpha = 60  // Semi-transparent
                }
                canvas.drawRect(left, top, right, bottom, fillPaint)
                
                // Draw label background
                val confidence = (detection.confidence * 100).toInt()
                val labelText = "${detection.className} ${confidence}%"
                textPaint.textSize = 40f  // Larger text for better visibility
                val textWidth = textPaint.measureText(labelText) + 20f
                val textHeight = textPaint.textSize + 10f

                // Position label above the box
                backgroundPaint.color = Color.parseColor("#AA000000") // Semi-transparent black
                canvas.drawRect(left, top - textHeight, left + textWidth, top, backgroundPaint)

                // Draw label text
                textPaint.color = Color.WHITE
                canvas.drawText(labelText, left + 10f, top - 10f, textPaint)
                
                // Log successful drawing
                Log.d(TAG, "Successfully drew bounding box for ${detection.className}")
            } catch (e: Exception) {
                Log.e(TAG, "Error drawing detection box: ${e.message}")
                e.printStackTrace()
            }
        }

        // Add this new method to calculate the visible rectangle of the model's coordinate space
        private fun getVisibleModelRect(): Rect {
            val previewAspectRatio = previewWidth.toFloat() / previewHeight.toFloat()
            val modelAspectRatio = MODEL_WIDTH.toFloat() / MODEL_HEIGHT.toFloat() // 1.0 (square)
            
            // Calculate the rectangle in the preview that represents the visible part of the model
            if (previewAspectRatio > modelAspectRatio) {
                // Preview is wider than the model - use height for scaling, center horizontally
                val effectiveHeight = previewHeight
                val effectiveWidth = effectiveHeight * modelAspectRatio
                val leftPadding = (previewWidth - effectiveWidth) / 2
                
                return Rect(
                    leftPadding.toInt(),
                    0,
                    (leftPadding + effectiveWidth).toInt(),
                    previewHeight
                )
            } else {
                // Preview is taller than the model - use width for scaling, center vertically
                val effectiveWidth = previewWidth
                val effectiveHeight = effectiveWidth / modelAspectRatio
                val topPadding = (previewHeight - effectiveHeight) / 2
                
                return Rect(
                    0,
                    topPadding.toInt(),
                    previewWidth,
                    (topPadding + effectiveHeight).toInt()
                )
            }
        }
        
        override fun surfaceCreated(holder: SurfaceHolder) {
            Log.d(TAG, "BoxOverlay surface created")
            drawDetections()
        }
        
        override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
            Log.d(TAG, "BoxOverlay surface changed: ${width}x${height}")
            if (width > 0 && height > 0) {
                previewWidth = width
                previewHeight = height
                drawDetections()
            }
        }
        
        override fun surfaceDestroyed(holder: SurfaceHolder) {
            Log.d(TAG, "BoxOverlay surface destroyed")
        }

        // Helper function to format floats nicely
        private fun Float.format(digits: Int) = "%.${digits}f".format(this)
    }
}


