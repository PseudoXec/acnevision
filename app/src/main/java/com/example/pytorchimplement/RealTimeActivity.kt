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
        (previewView.parent as ViewGroup).addView(boxOverlay, ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))
        
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
                "Real-time acne detection is running continuously.",
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

                // Set up the preview use case
                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                // Set up the image capture use case
                imageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()

                // Set up image analysis for real-time detection
                val imageAnalysis = ImageAnalysis.Builder()
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
                
                // Update overlay with preview size once view is laid out
                previewView.post {
                    boxOverlay.setPreviewSize(previewView.width, previewView.height)
                    boxOverlay.setCameraFacing(isFrontCamera)
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
        
        init {
            setZOrderOnTop(true)
            holder.setFormat(android.graphics.PixelFormat.TRANSPARENT)
            holder.addCallback(this)
            setWillNotDraw(false) // Ensure onDraw is called
        }
        
        fun setDetections(newDetections: List<ImageAnalyzer.Detection>) {
            detections = newDetections
            
            // Add debug logs
            if (detections.isNotEmpty()) {
                val now = System.currentTimeMillis()
                Log.d(TAG, "Setting ${detections.size} detections to draw, last draw was ${now - lastDrawTime}ms ago")
                lastDrawTime = now
            }
            
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
                } else {
                    detections.forEach { detection ->
                        drawDetection(canvas, detection)
                    }
                    
                    // Draw detection count at the top
                    textPaint.textSize = 60f
                    val countText = "Found: ${detections.size} detections"
                    canvas.drawText(countText, 50f, 120f, textPaint)
                    
                    // Draw timestamp
                    debugPaint.textSize = 36f
                    val timestamp = System.currentTimeMillis()
                    canvas.drawText("Last updated: ${timestamp - lastDrawTime}ms ago", 50f, 180f, debugPaint)
                }
            } finally {
                holder.unlockCanvasAndPost(canvas)
            }
        }
        
        private fun drawDetection(canvas: Canvas, detection: ImageAnalyzer.Detection) {
            if (previewWidth <= 0 || previewHeight <= 0) {
                Log.d(TAG, "Cannot draw - preview size is not set")
                return
            }
            
            // Colors for different acne types
            val boxColor = when (detection.classId) {
                0 -> Color.GREEN      // comedone
                1 -> Color.YELLOW     // pustule
                2 -> Color.BLUE       // papule
                3 -> Color.RED        // nodule
                else -> Color.WHITE   // unknown
            }
            
            // Set the box color
            paint.color = boxColor
            
            // Get normalized coordinates from detection
            var left = (detection.boundingBox.x - detection.boundingBox.width / 2) * previewWidth
            var top = (detection.boundingBox.y - detection.boundingBox.height / 2) * previewHeight
            var right = (detection.boundingBox.x + detection.boundingBox.width / 2) * previewWidth
            var bottom = (detection.boundingBox.y + detection.boundingBox.height / 2) * previewHeight
            
            // Ensure coordinates are within preview bounds
            left = left.coerceIn(0f, previewWidth.toFloat())
            top = top.coerceIn(0f, previewHeight.toFloat())
            right = right.coerceIn(0f, previewWidth.toFloat())
            bottom = bottom.coerceIn(0f, previewHeight.toFloat())
            
            // If front camera, flip horizontally
            if (isFrontCamera) {
                val tmp = left
                left = previewWidth - right
                right = previewWidth - tmp
            }
            
            // Log bounding box dimensions once for debugging
            Log.d(TAG, "Drawing box at [${left.toInt()},${top.toInt()},${right.toInt()},${bottom.toInt()}], " +
                  "original: [${detection.boundingBox.x},${detection.boundingBox.y},${detection.boundingBox.width},${detection.boundingBox.height}]")
            
            // Draw a more visible bounding box with a thicker stroke
            // First draw an outer stroke in black for contrast
            val originalStrokeWidth = paint.strokeWidth
            val originalColor = paint.color
            
            // Draw contrasting outline
            paint.strokeWidth = originalStrokeWidth + 4
            paint.color = Color.BLACK
            canvas.drawRect(left - 2, top - 2, right + 2, bottom + 2, paint)
            
            // Draw main colored box
            paint.strokeWidth = originalStrokeWidth
            paint.color = originalColor
            canvas.drawRect(left, top, right, bottom, paint)
            
            // Prepare text with class name and confidence
            val text = "${detection.className} ${(detection.confidence * 100).toInt()}%"
            
            // Calculate text width and height
            val textWidth = textPaint.measureText(text)
            val textHeight = textPaint.textSize
            
            // Draw background for text
            canvas.drawRect(
                left,
                top - textHeight - 10,
                left + textWidth + 20,
                top,
                backgroundPaint
            )
            
            // Draw detection text
            canvas.drawText(
                text,
                left + 10,
                top - 10,
                textPaint
            )
            
            // Draw confidence and box dimensions for debugging
            val debugText = "conf:${String.format("%.2f", detection.confidence)} " +
                           "dim:${String.format("%.2f", detection.boundingBox.width)}x" +
                           "${String.format("%.2f", detection.boundingBox.height)}"
            
            canvas.drawRect(
                left,
                bottom,
                left + textPaint.measureText(debugText) + 20,
                bottom + textHeight + 10,
                backgroundPaint
            )
            
            canvas.drawText(
                debugText,
                left + 10,
                bottom + textHeight,
                debugPaint
            )
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
    }
}


