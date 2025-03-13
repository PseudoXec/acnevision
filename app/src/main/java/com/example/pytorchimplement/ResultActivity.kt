package com.example.pytorchimplement

import android.annotation.SuppressLint
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
import androidx.viewpager2.widget.ViewPager2
import androidx.recyclerview.widget.RecyclerView
import android.view.LayoutInflater
import com.example.pytorchimplement.ImageAnalyzer.Detection
import com.example.pytorchimplement.ImageAnalyzer.BoundingBox
import com.github.mikephil.charting.charts.PieChart
import com.github.mikephil.charting.data.PieData
import com.github.mikephil.charting.data.PieDataSet
import com.github.mikephil.charting.data.PieEntry
import com.github.mikephil.charting.utils.ColorTemplate
import java.util.Arrays
import java.util.ArrayList
import java.io.File
import android.graphics.Rect

/**
 * Activity that displays the results of acne analysis with severity scores and bounding boxes.
 * Now using ONNX Runtime for inference.
 */
class ResultActivity : AppCompatActivity() {
    
    private val TAG = "ResultActivity"
    private lateinit var regionViewPager: ViewPager2
    
    // UI elements
    private lateinit var severityTextView: TextView
    private lateinit var detailsTextView: TextView
    private lateinit var recommendationsTextView: TextView
    private lateinit var homeButton: Button
    private lateinit var newScanButton: Button
    private lateinit var pieChart: PieChart
    
    // Storage directory for analysis results
    private lateinit var storageDir: File
    private var analysisId: String? = null
    
    // Facial regions
    private val regions = listOf("forehead", "nose", "left_cheek", "right_cheek", "chin")
    private val regionDisplayNames = mapOf(
        "forehead" to "Forehead",
        "nose" to "Nose",
        "left_cheek" to "Left Cheek",
        "right_cheek" to "Right Cheek",
        "chin" to "Chin"
    )
    
    // Data class for region images and detections
    data class RegionData(
        val regionId: String,
        val displayName: String,
        val bitmap: Bitmap,
        val detections: List<Detection>,
        val inferenceTimeMs: Long
    )
    
    // Data class for detections
    data class Detection(
        val className: String,
        val confidence: Float,
        val boundingBox: BoundingBox
    )
    
    // Bounding box coordinates class (all values normalized 0-1)
    data class BoundingBox(
        val x: Float, // center x coordinate
        val y: Float, // center y coordinate
        val width: Float, // width of box
        val height: Float // height of box
    )

    // Current region index being displayed
    private var currentRegionIndex = 0
    
    // Map to store region image views
    private val regionImageViews = mutableMapOf<Int, ImageView>()

    @SuppressLint("DefaultLocale")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)
        
        // Initialize storage directory
        storageDir = File(getExternalFilesDir(null), "acne_analysis_results")
        
        // Initialize UI elements
        severityTextView = findViewById(R.id.severity_text)
        detailsTextView = findViewById(R.id.details_text)
        recommendationsTextView = findViewById(R.id.recommendations_text)
        homeButton = findViewById(R.id.home_button)
        newScanButton = findViewById(R.id.new_scan_button)
        pieChart = findViewById(R.id.pie_chart)
        regionViewPager = findViewById(R.id.region_view_pager)
        
        // Set up ViewPager page change listener
        regionViewPager.registerOnPageChangeCallback(object : ViewPager2.OnPageChangeCallback() {
            override fun onPageSelected(position: Int) {
                super.onPageSelected(position)
                currentRegionIndex = position
                // Redraw the current overlay when page changes
                val adapter = regionViewPager.adapter as? RegionPagerAdapter
                adapter?.getOverlayForPosition(position)?.drawDetections()
            }
        })

        // Get the analysis ID from the intent
        analysisId = intent.getStringExtra("analysis_id")
        
        // Get the analysis data from the intent
        val severity = intent.getStringExtra("severity") ?: "Unknown"
        val totalScore = intent.getIntExtra("total_score", 0)
        val totalCount = intent.getIntExtra("total_count", 0)
        val comedoneCount = intent.getIntExtra("comedone_count", 0)
        val pustuleCount = intent.getIntExtra("pustule_count", 0)
        val papuleCount = intent.getIntExtra("papule_count", 0)
        val noduleCount = intent.getIntExtra("nodule_count", 0)
        
        // Get the inference time from intent
        val inferenceTime = intent.getLongExtra("inference_time", 0)
        
        // Set severity text
        val severityText = String.format("Acne Severity: %s (Score: %d)", severity, totalScore)
        severityTextView.text = severityText

        // Set acne summary
        val summaryBuilder = StringBuilder()
        summaryBuilder.append("Acne Breakdown:\n")
        if (comedoneCount > 0) summaryBuilder.append("• Comedones: $comedoneCount (${calculatePercentage(comedoneCount, totalCount)}%)\n")
        if (pustuleCount > 0) summaryBuilder.append("• Pustules: $pustuleCount (${calculatePercentage(pustuleCount, totalCount)}%)\n")
        if (papuleCount > 0) summaryBuilder.append("• Papules: $papuleCount (${calculatePercentage(papuleCount, totalCount)}%)\n")
        if (noduleCount > 0) summaryBuilder.append("• Nodules: $noduleCount (${calculatePercentage(noduleCount, totalCount)}%)\n")
        summaryBuilder.append("\nTotal Acne Count: $totalCount")
        summaryBuilder.append("\nInference Time: ${inferenceTime}ms")
        
        detailsTextView.text = summaryBuilder.toString()
        
        // Set up pie chart for acne distribution
        setupPieChart(comedoneCount, pustuleCount, papuleCount, noduleCount, totalCount)

        // Set recommendations based on severity
        val recommendationsBuilder = StringBuilder("Recommendations:\n\n")
        
        when (severity) {
            "No Acne" -> {
                recommendationsBuilder.append("• No acne detected - maintain your current skincare routine")
            }
            "Mild" -> {
                recommendationsBuilder.append("• Mild acne detected\n")
                recommendationsBuilder.append("• Consider over-the-counter treatments containing benzoyl peroxide or salicylic acid\n")
                recommendationsBuilder.append("• Maintain a consistent cleansing routine\n")
                recommendationsBuilder.append("• Avoid picking or squeezing acne lesions")
            }
            "Moderate" -> {
                recommendationsBuilder.append("• Moderate acne detected - consider prescription treatments\n")
                recommendationsBuilder.append("• Consider consulting with a dermatologist\n")
                recommendationsBuilder.append("• Recommended treatments: Topical antibiotics, Retinoids\n")
                recommendationsBuilder.append("• Avoid picking or squeezing acne lesions")
            }
            "Severe", "Very Severe" -> {
                recommendationsBuilder.append("• Severe acne detected - prescription treatment recommended\n")
                recommendationsBuilder.append("• Consult with a dermatologist as soon as possible\n")
                recommendationsBuilder.append("• Potential treatments may include oral antibiotics, isotretinoin, or hormonal therapy\n")
                recommendationsBuilder.append("• Follow a gentle skincare routine to avoid irritation")
            }
            else -> {
                recommendationsBuilder.append("• Please consult with a dermatologist for personalized advice")
            }
        }
        recommendationsTextView.text = recommendationsBuilder.toString()

        // Load region images and detections
        loadRegionData()

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
    
    private fun loadRegionData() {
        // Check if we have a valid analysis ID
        if (analysisId.isNullOrEmpty()) {
            Log.e(TAG, "No analysis ID provided")
            return
        }
        
        val resultDir = File(storageDir, analysisId!!)
        if (!resultDir.exists() || !resultDir.isDirectory) {
            Log.e(TAG, "Analysis directory not found: ${resultDir.absolutePath}")
            return
        }
        
        // Load data for each region
        val regionDataList = mutableListOf<RegionData>()
        
        regions.forEach { regionId ->
            try {
                // Load image
                val imageFile = File(resultDir, "$regionId.jpg")
                if (imageFile.exists()) {
                    val bitmap = BitmapFactory.decodeFile(imageFile.absolutePath)
                    
                    // Load detections
                    val detections = loadDetections(resultDir, regionId)
                    
                    // Load inference time from summary file
                    var inferenceTimeMs = 0L
                    val summaryFile = File(resultDir, "${regionId}_summary.txt")
                    if (summaryFile.exists()) {
                        // Parse summary file to find inference time
                        summaryFile.readLines().forEach { line ->
                            if (line.contains("Inference Time:")) {
                                val timeStr = line.substringAfter("Inference Time:").trim()
                                inferenceTimeMs = timeStr.replace("ms", "").trim().toLongOrNull() ?: 0L
                            }
                        }
                    }
                    
                    // Add to list
                    regionDataList.add(RegionData(
                        regionId = regionId,
                        displayName = regionDisplayNames[regionId] ?: regionId,
                        bitmap = bitmap,
                        detections = detections,
                        inferenceTimeMs = inferenceTimeMs
                    ))
                    
                    Log.d(TAG, "Loaded region data for $regionId with ${detections.size} detections and ${inferenceTimeMs}ms inference time")
                } else {
                    Log.e(TAG, "Image file not found for region: $regionId")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading region data for $regionId: ${e.message}")
            }
        }
        
        // Set up ViewPager with region data
        if (regionDataList.isNotEmpty()) {
            val adapter = RegionPagerAdapter(regionDataList)
            regionViewPager.adapter = adapter
        }
    }
    
    private fun loadDetections(resultDir: File, regionId: String): List<Detection> {
        val detections = mutableListOf<Detection>()
        
        try {
            val detectionsFile = File(resultDir, "${regionId}_detections.csv")
            if (detectionsFile.exists()) {
                // Skip header line and parse each detection
                detectionsFile.readLines().drop(1).forEach { line ->
                    val parts = line.split(",")
                    if (parts.size >= 6) {
                        try {
                            val className = parts[0]
                            val confidence = parts[1].toFloat()
                            val x = parts[2].toFloat()
                            val y = parts[3].toFloat()
                            val width = parts[4].toFloat()
                            val height = parts[5].toFloat()
                            
                            detections.add(Detection(
                                className = className,
                                confidence = confidence,
                                boundingBox = BoundingBox(x, y, width, height)
                            ))
                        } catch (e: Exception) {
                            Log.e(TAG, "Error parsing detection line: $line")
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading detections for $regionId: ${e.message}")
        }
        
        return detections
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
                colors.add(Color.rgb(255, 255, 0))    // Yellow for comedones
                colors.add(Color.rgb(255, 0, 0))      // Red for pustules
                colors.add(Color.rgb(255, 0, 255))    // Magenta for papules
                colors.add(Color.rgb(0, 255, 0))      // Green for nodules
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
     * Adapter for the region ViewPager
     */
    inner class RegionPagerAdapter(private val regionDataList: List<RegionData>) : 
            RecyclerView.Adapter<RegionPagerAdapter.RegionViewHolder>() {
        
        // Store overlays for each position
        private val overlays = mutableMapOf<Int, BoxOverlay>()
        
        inner class RegionViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
            val regionTitle: TextView = itemView.findViewById(R.id.region_title)
            val regionImage: ImageView = itemView.findViewById(R.id.region_image)
            val regionContainer: FrameLayout = itemView.findViewById(R.id.region_container)
            val regionInferenceTime: TextView = itemView.findViewById(R.id.region_inference_time)
        }
        
        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RegionViewHolder {
            val view = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_region_result, parent, false)
            return RegionViewHolder(view)
        }
        
        override fun onBindViewHolder(holder: RegionViewHolder, position: Int) {
            val regionData = regionDataList[position]
            
            // Set region title
            holder.regionTitle.text = regionData.displayName
            
            // Set inference time
            holder.regionInferenceTime.text = "Inference time: ${regionData.inferenceTimeMs}ms"
            
            // Set region image
            holder.regionImage.setImageBitmap(regionData.bitmap)
            
            // Store reference to the image view
            regionImageViews[position] = holder.regionImage
            
            // Create and add box overlay
            val boxOverlay = BoxOverlay(this@ResultActivity)
            holder.regionContainer.addView(boxOverlay, ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            ))
            
            // Store reference to the overlay
            overlays[position] = boxOverlay
            
            // Set detections on the overlay
            holder.regionImage.post {
                boxOverlay.setPreviewSize(holder.regionContainer.width, holder.regionContainer.height)
                boxOverlay.setDetections(regionData.detections)
            }
        }
        
        override fun getItemCount(): Int = regionDataList.size
        
        // Method to get overlay for a specific position
        fun getOverlayForPosition(position: Int): BoxOverlay? {
            return overlays[position]
        }
    }
    
    /**
     * Custom view for drawing bounding boxes overlay on the static image
     */
    inner class BoxOverlay(context: Context) : SurfaceView(context), SurfaceHolder.Callback {
        private val paint = Paint().apply {
            isAntiAlias = true
            style = Paint.Style.STROKE
            strokeWidth = 4f
        }
        
        private val textPaint = Paint().apply {
            isAntiAlias = true
            style = Paint.Style.FILL
            color = Color.WHITE
            textSize = 30f
        }
        
        private val backgroundPaint = Paint().apply {
            style = Paint.Style.FILL
            color = Color.parseColor("#80000000")  // Semi-transparent black
        }
        
        private var detections: List<Detection> = emptyList()
        private var previewWidth = 0
        private var previewHeight = 0
        private var imageWidth = 0
        private var imageHeight = 0
        private var imageRect = Rect()
        
        init {
            setZOrderOnTop(true)
            holder.setFormat(android.graphics.PixelFormat.TRANSPARENT)
            holder.addCallback(this)
            setWillNotDraw(false)  // Ensure onDraw is called
        }
        
        fun setDetections(newDetections: List<Detection>) {
            detections = newDetections
            Log.d(TAG, "Setting ${detections.size} detections to draw")
            drawDetections()
        }
        
        fun setPreviewSize(width: Int, height: Int) {
            previewWidth = width
            previewHeight = height
            Log.d(TAG, "BoxOverlay size set to ${width}x${height}")
        }
        
        private fun updateImageRect() {
            // Find the actual image dimensions and position within the preview
            val imageView = regionImageViews[currentRegionIndex]
            
            if (imageView != null) {
                // Get the image drawable dimensions
                val drawable = imageView.drawable
                if (drawable != null) {
                    imageWidth = drawable.intrinsicWidth
                    imageHeight = drawable.intrinsicHeight
                    
                    // Get the image matrix to determine actual display size
                    val matrix = imageView.imageMatrix
                    val values = FloatArray(9)
                    matrix.getValues(values)
                    
                    // Calculate the actual displayed image size and position
                    val scaleX = values[Matrix.MSCALE_X]
                    val scaleY = values[Matrix.MSCALE_Y]
                    val transX = values[Matrix.MTRANS_X]
                    val transY = values[Matrix.MTRANS_Y]
                    
                    val displayWidth = imageWidth * scaleX
                    val displayHeight = imageHeight * scaleY
                    
                    // Create a rect representing the image bounds in the view
                    imageRect.set(
                        transX.toInt(),
                        transY.toInt(),
                        (transX + displayWidth).toInt(),
                        (transY + displayHeight).toInt()
                    )
                    
                    Log.d(TAG, "Image rect updated: $imageRect")
                } else {
                    // If no drawable, use the full view bounds
                    imageRect.set(0, 0, previewWidth, previewHeight)
                }
            } else {
                // Fallback to full view bounds
                imageRect.set(0, 0, previewWidth, previewHeight)
            }
        }
        
        fun drawDetections() {
            if (!holder.surface.isValid) {
                Log.d(TAG, "Cannot draw - surface is not valid")
                return
            }
            
            val canvas = holder.lockCanvas() ?: return
            try {
                // Clear the canvas
                canvas.drawColor(Color.TRANSPARENT, android.graphics.PorterDuff.Mode.CLEAR)
                
                // Update image rect
                updateImageRect()
                
                // Draw each detection box
                if (detections.isEmpty()) {
                    // Draw "No detections" text if needed
                    textPaint.textSize = 40f
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
            val boxColor = when {
                detection.className.contains("comedone") -> Color.YELLOW
                detection.className.contains("pustule") -> Color.RED
                detection.className.contains("papule") -> Color.MAGENTA
                detection.className.contains("nodule") -> Color.GREEN
                else -> Color.WHITE
            }
            
            // Set the box color
            paint.color = boxColor
            
            try {
                // Convert normalized coordinates to screen coordinates
                val left = (detection.boundingBox.x - detection.boundingBox.width / 2) * previewWidth
                val top = (detection.boundingBox.y - detection.boundingBox.height / 2) * previewHeight
                val right = (detection.boundingBox.x + detection.boundingBox.width / 2) * previewWidth
                val bottom = (detection.boundingBox.y + detection.boundingBox.height / 2) * previewHeight
                
                // Clip coordinates to image bounds
                val clippedLeft = left.coerceIn(imageRect.left.toFloat(), imageRect.right.toFloat())
                val clippedTop = top.coerceIn(imageRect.top.toFloat(), imageRect.bottom.toFloat())
                val clippedRight = right.coerceIn(imageRect.left.toFloat(), imageRect.right.toFloat())
                val clippedBottom = bottom.coerceIn(imageRect.top.toFloat(), imageRect.bottom.toFloat())
                
                // Only draw if the box is visible (has width and height)
                if (clippedRight > clippedLeft && clippedBottom > clippedTop) {
                    // Draw bounding box
                    canvas.drawRect(clippedLeft, clippedTop, clippedRight, clippedBottom, paint)
                    
                    // Draw a semi-transparent fill
                    val fillPaint = Paint().apply {
                        color = boxColor
                        style = Paint.Style.FILL
                        alpha = 60  // Semi-transparent
                    }
                    canvas.drawRect(clippedLeft, clippedTop, clippedRight, clippedBottom, fillPaint)
                    
                    // Draw center dot if it's within the image bounds
                    val centerX = detection.boundingBox.x * previewWidth
                    val centerY = detection.boundingBox.y * previewHeight
                    
                    if (centerX >= imageRect.left && centerX <= imageRect.right &&
                        centerY >= imageRect.top && centerY <= imageRect.bottom) {
                        val centerPaint = Paint().apply {
                            color = boxColor
                            style = Paint.Style.FILL
                            alpha = 255
                        }
                        canvas.drawCircle(centerX, centerY, 5f, centerPaint)
                    }
                    
                    // Prepare text with class name and confidence
                    val confidence = (detection.confidence * 100).toInt()
                    val text = "${detection.className} ${confidence}%"
                    
                    // Calculate text position - ensure it's within image bounds
                    val textWidth = textPaint.measureText(text)
                    val textHeight = textPaint.textSize
                    
                    // Try to position label above the box, but keep it inside the image
                    var labelLeft = clippedLeft
                    var labelTop = clippedTop - textHeight - 5
                    
                    // If label would be outside the top of the image, position it inside the box at the top
                    if (labelTop < imageRect.top) {
                        labelTop = clippedTop + 5
                    }
                    
                    // If label would extend beyond right edge of image, align right edge with image
                    if (labelLeft + textWidth + 10 > imageRect.right) {
                        labelLeft = imageRect.right - textWidth - 10
                    }
                    
                    // Draw background for text
                    canvas.drawRect(
                        labelLeft,
                        labelTop,
                        labelLeft + textWidth + 10,
                        labelTop + textHeight,
                        backgroundPaint
                    )
                    
                    // Draw text
                    textPaint.color = Color.WHITE
                    canvas.drawText(text, labelLeft + 5, labelTop + textHeight - 5, textPaint)
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