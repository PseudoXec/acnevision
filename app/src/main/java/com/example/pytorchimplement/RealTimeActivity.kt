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
    } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        // Android 10+ (API 29+)
        arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.READ_EXTERNAL_STORAGE
            // WRITE_EXTERNAL_STORAGE doesn't give general write access on API 29+
        )
    } else {
        // Older Android versions
        arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        )
    }

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

                // Allow the preview to use native aspect ratio but still feed the model a square input
                val preview = Preview.Builder()
                    // Don't force square preview - let it use the natural camera aspect ratio
                    .setTargetRotation(Surface.ROTATION_0)
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                // For image capture, we can still use the model's preferred size
                imageCapture = ImageCapture.Builder()
                    .setTargetResolution(android.util.Size(640, 640))
                    .setTargetRotation(Surface.ROTATION_0)
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()

                // For analysis, we still want the 640x640 input for the model
                val imageAnalysis = ImageAnalysis.Builder()
                    .setTargetResolution(android.util.Size(640, 640))
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
                    Log.d(TAG, "Camera resolution: ${cameraInfo.sensorRotationDegrees}")
                }
                
                // Update overlay with preview size - with better logging
                previewView.post {
                    val width = previewView.width
                    val height = previewView.height
                    
                    if (width > 0 && height > 0) {
                        Log.d(TAG, "PreviewView dimensions: ${width}x${height}, aspect ratio: ${width.toFloat()/height}")
                        boxOverlay.setPreviewSize(width, height)
                        boxOverlay.setCameraFacing(isFrontCamera)
                        
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

    override fun onAnalysisComplete(result: ImageAnalyzer.AnalysisResult) {
        // Update UI with the analysis result
        runOnUiThread {
            // Update header text with detection status
            resultTextView.text = "LIVE DETECTION: Acne Classification"
            
            // Always show the results - don't hide them
            resultTextView.visibility = View.VISIBLE
            
            // Update acne counts with more detail
            val countsText = StringBuilder("DETECTED ACNE TYPES:\n")
            var totalCount = 0
            
            result.acneCounts.forEach { (type, count) ->
                if (count > 0) {
                    countsText.append("• ${type.capitalize()}: $count\n")
                    totalCount += count
                }
            }
            
            // Add total count
            countsText.append("\nTOTAL ACNE DETECTED: $totalCount")
            
            // Add bounding box info
            if (result.detections.isNotEmpty()) {
                countsText.append("\n\nDetections with bounding boxes: ${result.detections.size}")
                
                // Show more details about some detections
                if (result.detections.size <= 3) {
                    countsText.append("\n\nDetailed detections:")
                    result.detections.forEachIndexed { index, detection ->
                        val box = detection.boundingBox
                        countsText.append("\n${index+1}. ${detection.className} (${(detection.confidence*100).toInt()}%) " +
                                          "at [${String.format("%.2f", box.x)}, ${String.format("%.2f", box.y)}]")
                    }
                }
            } else {
                countsText.append("\n\nNo bounding box detections")
            }
            
            detailsTextView.text = countsText.toString()
            detailsTextView.visibility = View.VISIBLE
            
            // Update bounding box overlay with new detections
            boxOverlay.setDetections(result.detections)
            
            // Log the analysis results with more detail
            Log.d(TAG, "Frame processed: Acne Types=${result.acneCounts}, " +
                    "Total=$totalCount, Detections=${result.detections.size}")
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
        
        private var detections: List<ImageAnalyzer.Detection> = emptyList()
        private var previewWidth = 0
        private var previewHeight = 0
        private var isFrontCamera = false
        private var lastDrawTime = 0L
        
        // Model dimensions - these match what we set in the camera configuration
        private val MODEL_WIDTH = 640
        private val MODEL_HEIGHT = 640
        
        init {
            setZOrderOnTop(true)
            holder.setFormat(android.graphics.PixelFormat.TRANSPARENT)
            holder.addCallback(this)
            setWillNotDraw(false) // Ensure onDraw is called
        }
        
        fun setDetections(newDetections: List<ImageAnalyzer.Detection>) {
            detections = newDetections
            
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
                
                // Draw each detection box
                if (detections.isEmpty()) {
                    // Draw "No detections" text for debug
                    textPaint.textSize = 60f
                    canvas.drawText("Waiting for detections...", 50f, 120f, textPaint)
                    
                    // Add timestamp for debugging
                    val timestamp = System.currentTimeMillis()
                    debugPaint.textSize = 36f
                    canvas.drawText("Last updated: ${timestamp - lastDrawTime}ms ago", 50f, 180f, debugPaint)
                    canvas.drawText("Preview size: ${previewWidth}x${previewHeight}", 50f, 230f, debugPaint)
                    
                    // Draw the area where detections would appear with aspect ratio correction
                    if (previewWidth > 0 && previewHeight > 0) {
                        // Show the effective area where boxes would be drawn (accounting for aspect ratio)
                        val aspectRatioPaint = Paint().apply {
                            style = Paint.Style.STROKE
                            strokeWidth = 2f
                            color = Color.CYAN
                        }
                        
                        // Draw a rectangle showing the target 640x640 region mapped to screen
                        drawModelAreaOutline(canvas, aspectRatioPaint)
                        
                        // Add explanation
                        debugPaint.textSize = 30f
                        canvas.drawText("Model input area (blue)", 50f, 290f, debugPaint)
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
                    canvas.drawText("Last updated: ${timestamp - lastDrawTime}ms ago", 50f, 180f, debugPaint)
                    canvas.drawText("Preview size: ${previewWidth}x${previewHeight}", 50f, 220f, debugPaint)
                    
                    // Draw model area outline to help visualize coordinate mapping
                    val aspectRatioPaint = Paint().apply {
                        style = Paint.Style.STROKE
                        strokeWidth = 2f
                        color = Color.CYAN
                    }
                    drawModelAreaOutline(canvas, aspectRatioPaint)
                }
            } finally {
                holder.unlockCanvasAndPost(canvas)
            }
        }
        
        // Draw outline showing the model's coordinate space mapped to screen space
        private fun drawModelAreaOutline(canvas: Canvas, paint: Paint) {
            val visibleModelRect = getVisibleModelRect()
            
            // Draw the outline
            canvas.drawRect(
                visibleModelRect.left.toFloat(),
                visibleModelRect.top.toFloat(),
                visibleModelRect.right.toFloat(),
                visibleModelRect.bottom.toFloat(),
                paint
            )
            
            // Add additional markers at corners for better debugging
            val markerSize = 20f
            
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
            
            // Add dimensions text
            val dimensionsText = "${visibleModelRect.width()}x${visibleModelRect.height()}"
            debugPaint.color = Color.CYAN
            canvas.drawText(
                dimensionsText,
                visibleModelRect.left.toFloat() + 10,
                visibleModelRect.top.toFloat() + 30,
                debugPaint
            )
        }
        
        // Get the rectangle that represents the model's 640x640 space mapped to screen space
        private fun getModelToScreenRect(): Rect {
            val previewAspectRatio = previewWidth.toFloat() / previewHeight.toFloat()
            
            if (previewAspectRatio > 1.0) {
                // Preview is wider than tall - center horizontally
                val effectiveWidth = previewHeight
                val leftPadding = (previewWidth - effectiveWidth) / 2
                return Rect(leftPadding, 0, leftPadding + effectiveWidth, previewHeight)
            } else {
                // Preview is taller than wide - center vertically
                val effectiveHeight = previewWidth
                val topPadding = (previewHeight - effectiveHeight) / 2
                return Rect(0, topPadding, previewWidth, topPadding + effectiveHeight)
            }
        }
        
        private fun drawBoxForDetection(canvas: Canvas, detection: ImageAnalyzer.Detection) {
            try {
                // Get the normalized coordinates (all in 0-1 range)
                val centerX = detection.boundingBox.x
                val centerY = detection.boundingBox.y
                val width = detection.boundingBox.width
                val height = detection.boundingBox.height
                
                // Log the normalized coordinates
                Log.d(TAG, "Drawing box for detection: centerX=$centerX centerY=$centerY width=$width height=$height")
                
                // Get the visible model area within the preview
                val visibleModelRect = getVisibleModelRect()
                Log.d(TAG, "Visible model rect: $visibleModelRect")
                
                // IMPROVED COORDINATE MAPPING:
                // The model coordinates come from a 640x640 square model where:
                // - (0,0) is the top-left
                // - (1,1) is the bottom-right
                
                // Map normalized model coordinates (0-1) directly to the visible model area
                val screenX = visibleModelRect.left + (centerX * visibleModelRect.width())
                val screenY = visibleModelRect.top + (centerY * visibleModelRect.height())
                
                // Scale the box dimensions proportionally
                val screenWidth = width * visibleModelRect.width()
                val screenHeight = height * visibleModelRect.height()
                
                // Ensure minimum box size for visibility (5% of the smaller screen dimension)
                val minDimension = Math.min(previewWidth, previewHeight) * 0.05f
                val finalWidth = Math.max(screenWidth, minDimension)
                val finalHeight = Math.max(screenHeight, minDimension)
                
                // Calculate the box corners
                var left = screenX - finalWidth / 2
                var top = screenY - finalHeight / 2
                var right = screenX + finalWidth / 2
                var bottom = screenY + finalHeight / 2
                
                Log.d(TAG, "Box mapped to: left=$left, top=$top, right=$right, bottom=$bottom")
                
                // Handle mirroring for front camera
                if (isFrontCamera) {
                    // For front camera, mirror horizontally around the center axis
                    val oldLeft = left
                    left = previewWidth - right
                    right = previewWidth - oldLeft
                    Log.d(TAG, "After mirroring: left=$left, top=$top, right=$right, bottom=$bottom")
                }
                
                // Set color based on acne type
                val boxColor = when {
                    detection.className.contains("comedone") -> Color.YELLOW
                    detection.className.contains("pustule") -> Color.RED
                    detection.className.contains("papule") -> Color.MAGENTA
                    detection.className.contains("nodule") -> Color.GREEN
                    else -> Color.WHITE
                }
                
                // Draw crosshair at the center point for better visualization
                val crosshairSize = 40f
                val crosshairPaint = Paint().apply {
                    color = boxColor
                    strokeWidth = 5f
                    style = Paint.Style.STROKE
                }
                
                // Draw the crosshair at the detection center
                canvas.drawLine(
                    screenX - crosshairSize, 
                    screenY, 
                    screenX + crosshairSize, 
                    screenY, 
                    crosshairPaint
                )
                canvas.drawLine(
                    screenX, 
                    screenY - crosshairSize, 
                    screenX, 
                    screenY + crosshairSize, 
                    crosshairPaint
                )
                
                // Make boxes more visible with increased width
                paint.color = boxColor
                paint.strokeWidth = 8f
                
                // Draw the bounding box
                canvas.drawRect(left, top, right, bottom, paint)
                
                // Add class label with confidence percentage
                val labelText = "${detection.className} ${(detection.confidence * 100).toInt()}%"
                
                // Create background for text
                textPaint.textSize = 46f
                val textWidth = textPaint.measureText(labelText)
                
                // Background for label
                backgroundPaint.color = Color.parseColor("#CC000000") // More opaque black
                canvas.drawRect(
                    left, 
                    bottom, // Place label at bottom of box for better visibility
                    left + textWidth + 20f, 
                    bottom + 60f, 
                    backgroundPaint
                )
                
                // Draw label text
                textPaint.color = Color.WHITE
                canvas.drawText(
                    labelText,
                    left + 10f,
                    bottom + 45f, // Align text inside the background
                    textPaint
                )
                
                // Draw coordinate debug info
                val debugText = "[${centerX.format(2)},${centerY.format(2)}]"
                backgroundPaint.color = Color.parseColor("#AA000000")
                canvas.drawRect(
                    left,
                    bottom + 60f,
                    left + textPaint.measureText(debugText) + 20,
                    bottom + 100f,
                    backgroundPaint
                )
                
                debugPaint.color = Color.YELLOW
                canvas.drawText(
                    debugText,
                    left + 10,
                    bottom + 90f,
                    debugPaint
                )
                
            } catch (e: Exception) {
                Log.e(TAG, "Error drawing detection box: ${e.message}")
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


