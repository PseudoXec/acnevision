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
import android.graphics.RectF
import android.view.WindowManager
import androidx.fragment.app.DialogFragment
import android.os.Parcelable
import android.os.Parcel
import android.widget.ImageButton
import android.view.Gravity

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
    
    // Custom Parcelable implementation for Detection
    data class Detection(
        val className: String,
        val confidence: Float,
        val boundingBox: BoundingBox
    ) : Parcelable {
        constructor(parcel: Parcel) : this(
            parcel.readString() ?: "",
            parcel.readFloat(),
            parcel.readParcelable(BoundingBox::class.java.classLoader) ?: BoundingBox(0f, 0f, 0f, 0f)
        )

        override fun writeToParcel(parcel: Parcel, flags: Int) {
            parcel.writeString(className)
            parcel.writeFloat(confidence)
            parcel.writeParcelable(boundingBox, flags)
        }

        override fun describeContents(): Int {
            return 0
        }

        companion object CREATOR : Parcelable.Creator<Detection> {
            override fun createFromParcel(parcel: Parcel): Detection {
                return Detection(parcel)
            }

            override fun newArray(size: Int): Array<Detection?> {
                return arrayOfNulls(size)
            }
        }
    }
    
    // Custom Parcelable implementation for BoundingBox
    data class BoundingBox(
        val x: Float, // center x coordinate
        val y: Float, // center y coordinate
        val width: Float, // width of box
        val height: Float // height of box
    ) : Parcelable {
        constructor(parcel: Parcel) : this(
            parcel.readFloat(),
            parcel.readFloat(),
            parcel.readFloat(),
            parcel.readFloat()
        )

        override fun writeToParcel(parcel: Parcel, flags: Int) {
            parcel.writeFloat(x)
            parcel.writeFloat(y)
            parcel.writeFloat(width)
            parcel.writeFloat(height)
        }

        override fun describeContents(): Int {
            return 0
        }

        companion object CREATOR : Parcelable.Creator<BoundingBox> {
            override fun createFromParcel(parcel: Parcel): BoundingBox {
                return BoundingBox(parcel)
            }

            override fun newArray(size: Int): Array<BoundingBox?> {
                return arrayOfNulls(size)
            }
        }
    }

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
     * Shows a full screen dialog with the expanded region image and detections
     */
    private fun showExpandedImageDialog(regionData: RegionData) {
        val dialog = ExpandedImageDialogFragment.newInstance(
            regionData.displayName,
            regionData.bitmap,
            regionData.detections.toTypedArray(),
            regionData.inferenceTimeMs
        )
        dialog.show(supportFragmentManager, "expanded_image_dialog")
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
            
            // Add tap-to-expand functionality
            holder.regionImage.setOnClickListener {
                showExpandedImageDialog(regionData)
            }
            
            // Also add a hint to inform users about the tap-to-expand feature
            holder.regionImage.contentDescription = "Tap to view ${regionData.displayName} in fullscreen"
            
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
            strokeWidth = 2f  // Thinner for smaller view
        }
        
        private val textPaint = Paint().apply {
            isAntiAlias = true
            style = Paint.Style.FILL
            color = Color.WHITE
            textSize = 24f  // Smaller text for pager view
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
                
                // Update image rect for accurate positioning
                updateImageRect()
                
                // Draw each detection box
                if (detections.isEmpty()) {
                    // Optional: Draw "No detections" text if needed
                } else {
                    // Filter to only include detections that have centers within the image area
                    val visibleDetections = detections.filter { detection ->
                        val centerX = imageRect.left + (detection.boundingBox.x * imageRect.width())
                        val centerY = imageRect.top + (detection.boundingBox.y * imageRect.height())
                        centerX >= imageRect.left && centerX <= imageRect.right &&
                        centerY >= imageRect.top && centerY <= imageRect.bottom
                    }
                    
                    // Draw each detection
                    visibleDetections.forEach { detection ->
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
                // Convert normalized coordinates to image coordinates
                val centerX = imageRect.left + (detection.boundingBox.x * imageRect.width())
                val centerY = imageRect.top + (detection.boundingBox.y * imageRect.height())
                
                // Calculate box size (ensure minimum size for visibility)
                val minBoxSize = imageRect.width() * 0.02f // 2% of image width minimum
                val boxWidth = Math.max(detection.boundingBox.width * imageRect.width(), minBoxSize)
                val boxHeight = Math.max(detection.boundingBox.height * imageRect.height(), minBoxSize)
                
                // Calculate box coordinates
                val left = centerX - (boxWidth / 2)
                val top = centerY - (boxHeight / 2)
                val right = centerX + (boxWidth / 2)
                val bottom = centerY + (boxHeight / 2)
                
                // Clip coordinates to image bounds
                val clippedLeft = left.coerceIn(imageRect.left.toFloat(), imageRect.right.toFloat())
                val clippedTop = top.coerceIn(imageRect.top.toFloat(), imageRect.bottom.toFloat())
                val clippedRight = right.coerceIn(imageRect.left.toFloat(), imageRect.right.toFloat())
                val clippedBottom = bottom.coerceIn(imageRect.top.toFloat(), imageRect.bottom.toFloat())
                
                // Only draw if the box is visible (has width and height)
                if (clippedRight > clippedLeft && clippedBottom > clippedTop) {
                    // Draw a semi-transparent fill (more visible in small view)
                    val fillPaint = Paint().apply {
                        color = boxColor
                        style = Paint.Style.FILL
                        alpha = 80  // More opaque for better visibility in small view
                    }
                    canvas.drawRect(clippedLeft, clippedTop, clippedRight, clippedBottom, fillPaint)
                    
                    // Draw the box outline
                    canvas.drawRect(clippedLeft, clippedTop, clippedRight, clippedBottom, paint)
                    
                    // Draw center dot
                    if (centerX >= imageRect.left && centerX <= imageRect.right &&
                        centerY >= imageRect.top && centerY <= imageRect.bottom) {
                        val centerPaint = Paint().apply {
                            color = boxColor
                            style = Paint.Style.FILL
                            alpha = 255
                        }
                        canvas.drawCircle(centerX, centerY, 3f, centerPaint)  // Smaller dot for pager view
                    }
                    
                    // For small view, may want to skip labels for very small boxes
                    val boxArea = (clippedRight - clippedLeft) * (clippedBottom - clippedTop)
                    val minAreaForLabel = imageRect.width() * imageRect.height() * 0.003f  // 0.3% of image area
                    
                    if (boxArea > minAreaForLabel) {
                        // Prepare text with abbreviated class name and confidence
                        val confidence = (detection.confidence * 100).toInt()
                        val shortName = when {
                            detection.className.contains("comedone") -> "Com"
                            detection.className.contains("pustule") -> "Pus"
                            detection.className.contains("papule") -> "Pap"
                            detection.className.contains("nodule") -> "Nod"
                            else -> "Acne"
                        }
                        val text = "$shortName ${confidence}%"
                        
                        // Smaller text for pager view
                        textPaint.textSize = 20f
                        val textWidth = textPaint.measureText(text)
                        val textHeight = textPaint.textSize + 2f
                        
                        // Position text above the box if possible
                        var labelLeft = clippedLeft
                        var labelTop = clippedTop - textHeight - 2f
                        
                        // Keep label inside image bounds
                        if (labelTop < imageRect.top) {
                            labelTop = clippedTop + 2f
                        }
                        if (labelLeft + textWidth > imageRect.right) {
                            labelLeft = imageRect.right - textWidth - 2f
                        }
                        
                        // Draw text background
                        val textBackground = RectF(
                            labelLeft,
                            labelTop,
                            labelLeft + textWidth + 4f,
                            labelTop + textHeight
                        )
                        canvas.drawRect(textBackground, backgroundPaint)
                        
                        // Draw text
                        canvas.drawText(text, labelLeft + 2f, labelTop + textHeight - 2f, textPaint)
                    }
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
    
    /**
     * DialogFragment to show a fullscreen expanded image with detection boxes
     */
    class ExpandedImageDialogFragment : DialogFragment() {
        private var image: Bitmap? = null
        private var detections: Array<Detection>? = null
        private var title: String? = null
        private var inferenceTimeMs: Long = 0
        
        companion object {
            private const val ARG_TITLE = "title"
            private const val ARG_DETECTIONS = "detections"
            private const val ARG_INFERENCE_TIME = "inference_time"
            
            fun newInstance(title: String, image: Bitmap, detections: Array<Detection>, inferenceTimeMs: Long): ExpandedImageDialogFragment {
                val fragment = ExpandedImageDialogFragment()
                fragment.image = image
                fragment.detections = detections
                fragment.title = title
                fragment.inferenceTimeMs = inferenceTimeMs
                return fragment
            }
        }
        
        override fun onCreateView(
            inflater: LayoutInflater,
            container: ViewGroup?,
            savedInstanceState: Bundle?
        ): View {
            // Use a custom layout for the dialog
            val view = inflater.inflate(R.layout.dialog_expanded_image, container, false)
            
            // Set up the image view
            val imageView = view.findViewById<ImageView>(R.id.expanded_image)
            imageView.setImageBitmap(image)
            
            // Set up the title
            val titleView = view.findViewById<TextView>(R.id.expanded_title)
            titleView.text = title
            
            // Set up inference time
            val timeView = view.findViewById<TextView>(R.id.expanded_inference_time)
            timeView.text = "Inference time: ${inferenceTimeMs}ms"
            
            // Add back button functionality
            val backButton = view.findViewById<ImageButton>(R.id.back_button)
            backButton.setOnClickListener {
                dismiss()
            }
            
            // Set up the overlay for detections
            val container = view.findViewById<FrameLayout>(R.id.expanded_container)
            
            // Create overlay for drawing detection boxes
            val overlay = ExpandedBoxOverlay(requireContext())
            container.addView(overlay, ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            ))
            
            // Set detections after layout is ready and image is measured
            imageView.post {
                // Get more accurate image bounds by checking the image matrix
                val matrixValues = FloatArray(9)
                imageView.imageMatrix.getValues(matrixValues)
                
                // Get the scaling factors and translation from the matrix
                val scaleX = matrixValues[Matrix.MSCALE_X]
                val scaleY = matrixValues[Matrix.MSCALE_Y]
                val transX = matrixValues[Matrix.MTRANS_X]
                val transY = matrixValues[Matrix.MTRANS_Y]
                
                // Calculate the actual display dimensions
                val imageWidth = image?.width?.toFloat() ?: 0f
                val imageHeight = image?.height?.toFloat() ?: 0f
                val displayWidth = imageWidth * scaleX
                val displayHeight = imageHeight * scaleY
                
                // Calculate the actual image boundaries within the container
                val imageRect = Rect(
                    transX.toInt(),
                    transY.toInt(),
                    (transX + displayWidth).toInt(),
                    (transY + displayHeight).toInt()
                )
                
                Log.d("ExpandedDialog", "Image displayed at: $imageRect, original: ${image?.width}x${image?.height}")
                
                overlay.setPreviewSize(container.width, container.height)
                overlay.setImageDisplayRect(imageRect)
                overlay.setDetections(detections?.toList() ?: emptyList())
            }
            
            return view
        }
        
        override fun onStart() {
            super.onStart()
            
            // Make dialog fullscreen
            dialog?.window?.apply {
                setLayout(WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.MATCH_PARENT)
                setGravity(Gravity.CENTER)
            }
        }
        
        /**
         * Custom view for drawing detection boxes in fullscreen mode
         */
        inner class ExpandedBoxOverlay(context: Context) : SurfaceView(context), SurfaceHolder.Callback {
            private val paint = Paint().apply {
                isAntiAlias = true
                style = Paint.Style.STROKE
                strokeWidth = 6f // Slightly thicker lines for visibility
            }
            
            private val textPaint = Paint().apply {
                isAntiAlias = true
                style = Paint.Style.FILL
                color = Color.WHITE
                textSize = 40f // Larger text for better readability
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
            private var imageDisplayRect = Rect()
            
            init {
                setZOrderOnTop(true)
                holder.setFormat(android.graphics.PixelFormat.TRANSPARENT)
                holder.addCallback(this)
                setWillNotDraw(false)
            }
            
            fun setDetections(newDetections: List<Detection>) {
                detections = newDetections
                drawDetections()
            }
            
            fun setPreviewSize(width: Int, height: Int) {
                previewWidth = width
                previewHeight = height
                // Set a default image rect covering the whole area
                imageRect.set(0, 0, width, height)
            }
            
            fun setImageDisplayRect(rect: Rect) {
                imageDisplayRect = rect
                imageRect = rect
                // Force redraw if we have detections
                if (detections.isNotEmpty()) {
                    drawDetections()
                }
            }
            
            private fun drawDetections() {
                if (!holder.surface.isValid) {
                    return
                }
                
                val canvas = holder.lockCanvas() ?: return
                try {
                    // Clear the canvas
                    canvas.drawColor(Color.TRANSPARENT, android.graphics.PorterDuff.Mode.CLEAR)
                    
                    // Draw each detection box
                    if (detections.isEmpty()) {
                        // Show "No detections" text if there are none
                        textPaint.textSize = 60f
                        val text = "No acne detected in this region"
                        val x = (previewWidth - textPaint.measureText(text)) / 2
                        canvas.drawText(text, x, previewHeight / 2f, textPaint)
                    } else {
                        // Draw each detection with improved visibility
                        detections.forEach { detection ->
                            drawDetection(canvas, detection)
                        }
                        
                        // Add a count indicator
                        val countText = "${detections.size} acne lesions detected"
                        textPaint.textSize = 50f
                        val textWidth = textPaint.measureText(countText)
                        val textBackground = Rect(
                            10, 
                            previewHeight - 100,
                            (textWidth + 30).toInt(),
                            previewHeight - 10
                        )
                        canvas.drawRect(textBackground, backgroundPaint)
                        canvas.drawText(countText, 20f, previewHeight - 30f, textPaint)
                    }
                } finally {
                    holder.unlockCanvasAndPost(canvas)
                }
            }
            
            private fun drawDetection(canvas: Canvas, detection: Detection) {
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
                    // Convert normalized coordinates to image display coordinates
                    val centerX = imageDisplayRect.left + (detection.boundingBox.x * imageDisplayRect.width())
                    val centerY = imageDisplayRect.top + (detection.boundingBox.y * imageDisplayRect.height())
                    val boxWidth = detection.boundingBox.width * imageDisplayRect.width()
                    val boxHeight = detection.boundingBox.height * imageDisplayRect.height()
                    
                    // Calculate raw box coordinates
                    val left = centerX - (boxWidth / 2)
                    val top = centerY - (boxHeight / 2)
                    val right = centerX + (boxWidth / 2)
                    val bottom = centerY + (boxHeight / 2)
                    
                    // IMPORTANT FIX: Clip coordinates to image bounds
                    val clippedLeft = left.coerceIn(imageDisplayRect.left.toFloat(), imageDisplayRect.right.toFloat())
                    val clippedTop = top.coerceIn(imageDisplayRect.top.toFloat(), imageDisplayRect.bottom.toFloat())
                    val clippedRight = right.coerceIn(imageDisplayRect.left.toFloat(), imageDisplayRect.right.toFloat())
                    val clippedBottom = bottom.coerceIn(imageDisplayRect.top.toFloat(), imageDisplayRect.bottom.toFloat())
                    
                    // Only draw if the box has positive width and height after clipping
                    if (clippedRight > clippedLeft && clippedBottom > clippedTop) {
                        // Draw bounding box with thicker stroke for better visibility
                        canvas.drawRect(clippedLeft, clippedTop, clippedRight, clippedBottom, paint)
                        
                        // Draw a semi-transparent fill
                        val fillPaint = Paint().apply {
                            color = boxColor
                            style = Paint.Style.FILL
                            alpha = 60
                        }
                        canvas.drawRect(clippedLeft, clippedTop, clippedRight, clippedBottom, fillPaint)
                        
                        // Draw center marker only if center is within image bounds
                        if (centerX >= imageDisplayRect.left && centerX <= imageDisplayRect.right &&
                            centerY >= imageDisplayRect.top && centerY <= imageDisplayRect.bottom) {
                            val centerPaint = Paint().apply {
                                color = boxColor
                                style = Paint.Style.FILL
                                alpha = 255
                            }
                            canvas.drawCircle(
                                centerX,
                                centerY,
                                10f, // Larger dot in fullscreen mode
                                centerPaint
                            )
                        }
                        
                        // Prepare text with class name and confidence
                        val confidence = (detection.confidence * 100).toInt()
                        val className = when {
                            detection.className.contains("comedone") -> "Comedone"
                            detection.className.contains("pustule") -> "Pustule"
                            detection.className.contains("papule") -> "Papule"
                            detection.className.contains("nodule") -> "Nodule"
                            else -> detection.className
                        }
                        val text = "$className ${confidence}%"
                        
                        // Larger text in fullscreen mode
                        textPaint.textSize = 40f
                        val textWidth = textPaint.measureText(text)
                        
                        // Position text above the box
                        var labelLeft = clippedLeft
                        var labelTop = clippedTop - 50
                        
                        // Ensure label stays on screen and within image bounds
                        if (labelTop < imageDisplayRect.top + 10) labelTop = clippedTop + 30
                        if (labelLeft + textWidth + 20 > imageDisplayRect.right) {
                            labelLeft = imageDisplayRect.right - textWidth - 20
                        }
                        
                        // Further constrain to image bounds
                        labelLeft = labelLeft.coerceIn(imageDisplayRect.left.toFloat(), 
                            (imageDisplayRect.right - textWidth - 20).coerceAtLeast(imageDisplayRect.left.toFloat()))
                        
                        // Draw background for text
                        val textBackgroundRect = Rect(
                            labelLeft.toInt(), 
                            labelTop.toInt(),
                            (labelLeft + textWidth + 20).toInt(),
                            (labelTop + 45).toInt()
                        )
                        canvas.drawRect(textBackgroundRect, backgroundPaint)
                        
                        // Draw text
                        canvas.drawText(text, labelLeft + 10, labelTop + 35, textPaint)
                    }
                } catch (e: Exception) {
                    Log.e("ExpandedBoxOverlay", "Error drawing detection: ${e.message}")
                }
            }
            
            override fun surfaceCreated(holder: SurfaceHolder) {
                drawDetections()
            }
            
            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
                if (width > 0 && height > 0) {
                    previewWidth = width
                    previewHeight = height
                    imageRect.set(0, 0, width, height)
                    drawDetections()
                }
            }
            
            override fun surfaceDestroyed(holder: SurfaceHolder) {}
        }
    }
}