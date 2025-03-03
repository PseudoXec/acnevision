package com.example.pytorchimplement

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import android.content.Context
import com.example.pytorchimplement.ImageAnalyzer.Detection
import com.example.pytorchimplement.ImageAnalyzer.BoundingBox
import com.github.mikephil.charting.charts.PieChart
import com.github.mikephil.charting.data.PieData
import com.github.mikephil.charting.data.PieDataSet
import com.github.mikephil.charting.data.PieEntry
import com.github.mikephil.charting.utils.ColorTemplate
import java.util.Arrays
import java.util.ArrayList

/**
 * Activity that displays the results of acne analysis with severity scores and bounding boxes.
 * Now using ONNX Runtime for inference.
 */
class ResultActivity : AppCompatActivity() {
    
    private val TAG = "ResultActivity"
    private lateinit var boxOverlay: BoxOverlay
    private lateinit var imageContainer: FrameLayout
    private lateinit var imageView: ImageView
    
    // UI elements
    private lateinit var severityTextView: TextView
    private lateinit var detailsTextView: TextView
    private lateinit var recommendationsTextView: TextView
    private lateinit var homeButton: Button
    private lateinit var newScanButton: Button
    private lateinit var pieChart: PieChart
    
    // Scale and position factors for drawing bounding boxes correctly
    private var scaleFactor = 1.0f
    private var offsetX = 0f
    private var offsetY = 0f
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)
        
        // Initialize UI elements
        imageView = findViewById(R.id.result_image)
        severityTextView = findViewById(R.id.severity_text)
        detailsTextView = findViewById(R.id.details_text)
        recommendationsTextView = findViewById(R.id.recommendations_text)
        homeButton = findViewById(R.id.home_button)
        newScanButton = findViewById(R.id.new_scan_button)
        pieChart = findViewById(R.id.pie_chart)

        // Get the analysis data from the intent
        val severity = intent.getIntExtra("severity", 0)
        val totalCount = intent.getIntExtra("total_count", 0)
        val comedoneCount = intent.getIntExtra("comedone_count", 0)
        val pustuleCount = intent.getIntExtra("pustule_count", 0)
        val papuleCount = intent.getIntExtra("papule_count", 0)
        val noduleCount = intent.getIntExtra("nodule_count", 0)
        val imageBytes = intent.getByteArrayExtra("image")
        
        // Get detection data for bounding boxes (if available)
        val detections = getDetectionsFromIntent()
        
        // Create the container for the image and box overlay
        setupImageContainer()
        
        // Set severity score
        val severityText = String.format("Acne Severity: %s (%d%%)", 
            when {
                severity < 3 -> "Mild"
                severity < 5 -> "Moderate" 
                severity < 8 -> "Moderately severe"
                else -> "Severe"
            },
            (severity * 10) // Convert to percentage
        )
        severityTextView.text = severityText

        // Set acne summary
        val summaryBuilder = StringBuilder()
        summaryBuilder.append("Acne Breakdown:\n")
        if (comedoneCount > 0) summaryBuilder.append("• Comedones: $comedoneCount (${calculatePercentage(comedoneCount, totalCount)}%)\n")
        if (pustuleCount > 0) summaryBuilder.append("• Pustules: $pustuleCount (${calculatePercentage(pustuleCount, totalCount)}%)\n")
        if (papuleCount > 0) summaryBuilder.append("• Papules: $papuleCount (${calculatePercentage(papuleCount, totalCount)}%)\n")
        if (noduleCount > 0) summaryBuilder.append("• Nodules: $noduleCount (${calculatePercentage(noduleCount, totalCount)}%)\n")
        summaryBuilder.append("\nTotal Acne Count: $totalCount")
        
        // Add information about bounding boxes
        if (detections.isNotEmpty()) {
            summaryBuilder.append("\n\nBounding boxes show acne location and type")
        }
        
        detailsTextView.text = summaryBuilder.toString()
        
        // Set up pie chart for acne distribution
        setupPieChart(comedoneCount, pustuleCount, papuleCount, noduleCount, totalCount)

        // Set recommendations based on severity
        val recommendationsBuilder = StringBuilder("Recommendations:\n\n")
        when {
            severity < 3 -> {
                recommendationsBuilder.append("• Mild acne detected - over-the-counter treatments may be effective\n")
                recommendationsBuilder.append("• Recommended treatments: Benzoyl peroxide, Salicylic acid\n")
                recommendationsBuilder.append("• Maintain a consistent face washing routine\n")
                recommendationsBuilder.append("• Avoid excessive scrubbing or harsh cleansers")
            }
            severity < 5 -> {
                recommendationsBuilder.append("• Moderate acne detected - consider prescription treatments\n")
                recommendationsBuilder.append("• Consider consulting with a dermatologist\n")
                recommendationsBuilder.append("• Recommended treatments: Topical antibiotics, Retinoids\n")
                recommendationsBuilder.append("• Avoid picking or squeezing acne lesions")
            }
            severity < 8 -> {
                recommendationsBuilder.append("• Moderately severe acne detected - prescription treatments recommended\n")
                recommendationsBuilder.append("• Consult with a dermatologist for personalized treatment\n")
                recommendationsBuilder.append("• Recommended treatments: Oral antibiotics, Stronger retinoids\n")
                recommendationsBuilder.append("• Consider lifestyle factors like diet and stress")
            }
            else -> {
                recommendationsBuilder.append("• Severe acne detected - urgent dermatological care recommended\n")
                recommendationsBuilder.append("• See a dermatologist as soon as possible\n")
                recommendationsBuilder.append("• Potential treatments: Isotretinoin, Hormone therapy\n")
                recommendationsBuilder.append("• Monitor for psychological impacts and seek support if needed")
            }
        }
        recommendationsTextView.text = recommendationsBuilder.toString()

        // Display the image if available
        if (imageBytes != null) {
            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            imageView.setImageBitmap(bitmap)
            
            // Wait for the ImageView to be laid out before calculating scale factors
            imageView.post {
                calculateImageScaleFactors(bitmap)
                
                // Set detections for the overlay once scaling is calculated
                if (detections.isNotEmpty()) {
                    boxOverlay.setPreviewSize(imageContainer.width, imageContainer.height)
                    boxOverlay.setDetections(detections, scaleFactor, offsetX, offsetY)
                }
            }
        } else {
            // Hide image container if no image
            imageContainer.visibility = View.GONE
        }

        // Set up button actions
        newScanButton.setOnClickListener {
            val intent = Intent(this, CaptureActivity::class.java)
            startActivity(intent)
            finish()
        }

        homeButton.setOnClickListener {
            val intent = Intent(this, SeverityActivity::class.java)
            startActivity(intent)
            finish()
        }
    }
    
    /**
     * Sets up the pie chart with acne count data
     */
    private fun setupPieChart(comedoneCount: Int, pustuleCount: Int, papuleCount: Int, noduleCount: Int, totalCount: Int) {
        try {
            // Only set up chart if there's data to show
            if (totalCount > 0) {
                val entries = ArrayList<PieEntry>()
                
                // Add non-zero entries
                if (comedoneCount > 0) entries.add(PieEntry(comedoneCount.toFloat(), "Comedones"))
                if (pustuleCount > 0) entries.add(PieEntry(pustuleCount.toFloat(), "Pustules"))
                if (papuleCount > 0) entries.add(PieEntry(papuleCount.toFloat(), "Papules"))
                if (noduleCount > 0) entries.add(PieEntry(noduleCount.toFloat(), "Nodules"))
                
                // Create data set
                val dataSet = PieDataSet(entries, "Acne Types")
                
                // Set colors
                val colors = ArrayList<Int>()
                colors.add(Color.rgb(76, 175, 80))    // Green for comedones
                colors.add(Color.rgb(255, 235, 59))   // Yellow for pustules
                colors.add(Color.rgb(33, 150, 243))   // Blue for papules
                colors.add(Color.rgb(244, 67, 54))    // Red for nodules
                dataSet.colors = colors
                
                // Set chart properties
                val data = PieData(dataSet)
                data.setValueTextSize(12f)
                data.setValueTextColor(Color.WHITE)
                
                // Configure the chart
                pieChart.data = data
                pieChart.description.isEnabled = false
                pieChart.centerText = "Acne Types"
                pieChart.setCenterTextSize(14f)
                pieChart.setEntryLabelColor(Color.WHITE)
                pieChart.legend.textSize = 12f
                pieChart.setDrawEntryLabels(false)
                pieChart.setUsePercentValues(true)
                
                // Animate the chart
                pieChart.animateY(1000)
                
                // Refresh the chart
                pieChart.invalidate()
            } else {
                // If no data, hide the chart
                pieChart.visibility = View.GONE
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error setting up pie chart: ${e.message}")
            pieChart.visibility = View.GONE
        }
    }
    
    /**
     * Calculate percentage for acne type
     */
    private fun calculatePercentage(count: Int, total: Int): Int {
        return if (total > 0) (count * 100) / total else 0
    }
    
    /**
     * Extract detection data from the intent
     */
    private fun getDetectionsFromIntent(): List<Detection> {
        val detectionsResult = mutableListOf<Detection>()
        val detectionsCount = intent.getIntExtra("detections_count", 0)
        
        if (detectionsCount > 0) {
            for (i in 0 until detectionsCount) {
                val classId = intent.getIntExtra("detection_${i}_class_id", 0)
                val className = intent.getStringExtra("detection_${i}_class_name") ?: "unknown"
                val confidence = intent.getFloatExtra("detection_${i}_confidence", 0f)
                val x = intent.getFloatExtra("detection_${i}_x", 0f)
                val y = intent.getFloatExtra("detection_${i}_y", 0f)
                val width = intent.getFloatExtra("detection_${i}_width", 0f)
                val height = intent.getFloatExtra("detection_${i}_height", 0f)
                
                detectionsResult.add(Detection(
                    classId = classId,
                    className = className,
                    confidence = confidence,
                    boundingBox = BoundingBox(x, y, width, height)
                ))
            }
            Log.d(TAG, "Received ${detectionsResult.size} detections for visualization")
        }
        
        return detectionsResult
    }
    
    /**
     * Set up the container for the image and overlay
     */
    private fun setupImageContainer() {
        // Create a FrameLayout container to hold both the image and the overlay
        imageContainer = FrameLayout(this)
        
        // Find the original parent of the ImageView
        val imageParent = imageView.parent as ViewGroup
        val imageIndex = imageParent.indexOfChild(imageView)
        
        // Get the layout params of the original ImageView
        val layoutParams = imageView.layoutParams
        
        // Remove the ImageView from its parent
        imageParent.removeView(imageView)
        
        // Add the container at the same position
        imageParent.addView(imageContainer, imageIndex, layoutParams)
        
        // Add the ImageView to the container
        imageContainer.addView(imageView, ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))
        
        // Create and add the box overlay
        boxOverlay = BoxOverlay(this)
        imageContainer.addView(boxOverlay, ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))
    }
    
    /**
     * Calculate scale factors needed to correctly position bounding boxes
     */
    private fun calculateImageScaleFactors(bitmap: Bitmap) {
        try {
            // Get the matrix from the ImageView
            val matrix = imageView.imageMatrix
            
            // Create an array to hold the values
            val values = FloatArray(9)
            matrix.getValues(values)
            
            // Extract scale values
            val scaleX = values[Matrix.MSCALE_X]
            val scaleY = values[Matrix.MSCALE_Y]
            
            // Extract translation values
            offsetX = values[Matrix.MTRANS_X]
            offsetY = values[Matrix.MTRANS_Y]
            
            // Use the smaller scale to ensure boxes fit within the image
            scaleFactor = minOf(scaleX, scaleY)
            
            Log.d(TAG, "Image scale factors: scale=$scaleFactor, offsetX=$offsetX, offsetY=$offsetY")
        } catch (e: Exception) {
            Log.e(TAG, "Error calculating scale factors: ${e.message}")
            // Use safe defaults if calculation fails
            scaleFactor = 1.0f
            offsetX = 0f
            offsetY = 0f
        }
    }
    
    /**
     * Data classes for detections and bounding boxes
     */
    data class BoundingBox(
        val x: Float,      // center x coordinate (normalized 0-1)
        val y: Float,      // center y coordinate (normalized 0-1)
        val width: Float,  // width of box (normalized 0-1)
        val height: Float  // height of box (normalized 0-1)
    )

    data class Detection(
        val classId: Int,          // class ID (0=comedone, 1=pustule, 2=papule, 3=nodule)
        val className: String,     // human-readable class name
        val confidence: Float,     // detection confidence 0-1
        val boundingBox: BoundingBox // normalized coordinates for the bounding box
    )
    
    /**
     * Custom view for drawing bounding boxes overlay on the static image
     */
    inner class BoxOverlay(context: Context) : SurfaceView(context), SurfaceHolder.Callback {
        private val paint = Paint().apply {
            isAntiAlias = true
            style = Paint.Style.STROKE
            strokeWidth = 8f  // Thicker lines for better visibility
        }
        
        private val textPaint = Paint().apply {
            isAntiAlias = true
            style = Paint.Style.FILL
            color = Color.WHITE
            textSize = 40f  // Larger text for better visibility
        }
        
        private val backgroundPaint = Paint().apply {
            style = Paint.Style.FILL
            color = Color.parseColor("#80000000")  // Semi-transparent black
        }
        
        private var detections: List<Detection> = emptyList()
        private var previewWidth = 0
        private var previewHeight = 0
        private var imageScaleFactor = 1.0f
        private var imageOffsetX = 0f
        private var imageOffsetY = 0f
        
        init {
            setZOrderOnTop(true)
            holder.setFormat(android.graphics.PixelFormat.TRANSPARENT)
            holder.addCallback(this)
            setWillNotDraw(false)  // Ensure onDraw is called
        }
        
        fun setDetections(newDetections: List<Detection>, scaleFactor: Float, offsetX: Float, offsetY: Float) {
            detections = newDetections
            imageScaleFactor = scaleFactor
            imageOffsetX = offsetX
            imageOffsetY = offsetY
            
            Log.d(TAG, "Setting ${detections.size} detections for result image, scale=$scaleFactor, offset=($offsetX,$offsetY)")
            
            drawDetections()
        }
        
        fun setPreviewSize(width: Int, height: Int) {
            previewWidth = width
            previewHeight = height
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
                    // Draw "No detections" text if needed
                    textPaint.textSize = 48f
                    val text = "No acne detected in this region"
                    val x = (previewWidth - textPaint.measureText(text)) / 2
                    canvas.drawText(text, x, previewHeight / 2f, textPaint)
                } else {
                    // Draw each bounding box
                    detections.forEach { detection ->
                        drawDetection(canvas, detection)
                    }
                }
            } finally {
                holder.unlockCanvasAndPost(canvas)
            }
        }
        
        private fun drawDetection(canvas: Canvas, detection: Detection) {
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
            
            try {
                // Get normalized coordinates from detection and apply scaling/offset
                val left = (detection.boundingBox.x - detection.boundingBox.width / 2) * previewWidth * imageScaleFactor + imageOffsetX
                val top = (detection.boundingBox.y - detection.boundingBox.height / 2) * previewHeight * imageScaleFactor + imageOffsetY
                val right = (detection.boundingBox.x + detection.boundingBox.width / 2) * previewWidth * imageScaleFactor + imageOffsetX
                val bottom = (detection.boundingBox.y + detection.boundingBox.height / 2) * previewHeight * imageScaleFactor + imageOffsetY
                
                // Ensure coordinates are within view bounds
                val safeLeft = left.coerceIn(0f, previewWidth.toFloat())
                val safeTop = top.coerceIn(0f, previewHeight.toFloat())
                val safeRight = right.coerceIn(0f, previewWidth.toFloat())
                val safeBottom = bottom.coerceIn(0f, previewHeight.toFloat())
                
                // Draw a more visible bounding box with a thicker stroke
                // First draw an outer stroke in black for contrast
                val originalStrokeWidth = paint.strokeWidth
                val originalColor = paint.color
                
                // Draw contrasting outline
                paint.strokeWidth = originalStrokeWidth + 4
                paint.color = Color.BLACK
                canvas.drawRect(safeLeft - 2, safeTop - 2, safeRight + 2, safeBottom + 2, paint)
                
                // Draw main colored box
                paint.strokeWidth = originalStrokeWidth
                paint.color = originalColor
                canvas.drawRect(safeLeft, safeTop, safeRight, safeBottom, paint)
                
                // Prepare text with class name and confidence
                val text = "${detection.className} ${(detection.confidence * 100).toInt()}%"
                
                // Calculate text width and height
                val textWidth = textPaint.measureText(text)
                val textHeight = textPaint.textSize
                
                // Draw background for text if it fits above the box
                if (safeTop > textHeight + 10) {
                    canvas.drawRect(
                        safeLeft,
                        safeTop - textHeight - 10,
                        safeLeft + textWidth + 20,
                        safeTop,
                        backgroundPaint
                    )
                    
                    // Draw detection text
                    canvas.drawText(
                        text,
                        safeLeft + 10,
                        safeTop - 10,
                        textPaint
                    )
                } else {
                    // Draw inside the box if text doesn't fit above
                    canvas.drawRect(
                        safeLeft + 5,
                        safeTop + 5,
                        safeLeft + textWidth + 25,
                        safeTop + textHeight + 15,
                        backgroundPaint
                    )
                    
                    canvas.drawText(
                        text,
                        safeLeft + 15,
                        safeTop + textHeight + 5,
                        textPaint
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error drawing detection: ${e.message}")
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
    }
}