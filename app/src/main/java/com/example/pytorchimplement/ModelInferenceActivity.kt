package com.example.pytorchimplement

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.nio.FloatBuffer
import android.content.Intent
import com.example.pytorchimplement.GAGSData.severity
import java.util.HashMap

class ModelInferenceActivity : AppCompatActivity() {
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private lateinit var resultTextView: TextView
    
    // Model config
    private val modelInputWidth = 640
    private val modelInputHeight = 640
    private val modelInputChannels = 3
    private val modelFileName = "yolov9.onnx"
    
    // Model input dimensions
    private var inputWidth = 640
    private var inputHeight = 640
    private var inputChannels = 3
    
    // Store detections for GAGS calculation
    private var detections = listOf<ImageAnalyzer.Detection>()
    
    companion object {
        private const val TAG = "ModelInferenceActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_inference)

        resultTextView = findViewById(R.id.resultsTextView)

        // Load ONNX model
        loadModel()

        // Process images based on the source

        val (imageList, acneCounts) = processImagesFromIntent()

        val gagsCalculator = GAGSCalculator()

        processMultipleImages(imageList, gagsCalculator)

        //singleton data
        val totalGAGSScore = GAGSData.totalGAGSScore
        val severity = GAGSData.severity
        val totalAcneCounts = GAGSData.totalAcneCounts


        // Display results
        displayResults(totalAcneCounts, totalGAGSScore, severity)
    }
    
    private fun loadModel() {
        try {
            // Create ONNX Runtime environment
            ortEnvironment = OrtEnvironment.getEnvironment()
            
            // Load the model from assets
            val modelBytes = assets.open(modelFileName).readBytes()
            
            // Create session options
            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setIntraOpNumThreads(4)
            
            // Create inference session
            ortSession = ortEnvironment?.createSession(modelBytes, sessionOptions)
            
            Log.d(TAG, "ONNX model loaded successfully: $modelFileName")
            
            // Get input information
            ortSession?.let { session ->
                val inputInfo = session.inputInfo
                val inputName = inputInfo.keys.firstOrNull()
                
                if (inputName != null) {
                    val inputNodeInfo = inputInfo[inputName]
                    val inputShape = (inputNodeInfo?.info as TensorInfo?)?.shape
                    
                    if (inputShape != null && inputShape.size >= 4) {
                        // Update model dimensions if available from the ONNX model
                        // ONNX models typically use NCHW format [batch_size, channels, height, width]
                        Log.d(TAG, "Model input shape: ${inputShape.contentToString()}")
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading ONNX model: ${e.message}")
            e.printStackTrace()
        }
    }

    private fun processImagesFromIntent(): Pair<List<Bitmap>, Map<String, Int>> {
        val imageList = mutableListOf<Bitmap>()  // List of images for GAGS scoring
        val totalCounts = mutableMapOf(
            "comedone" to 0,
            "pustule" to 0,
            "papule" to 0,
            "nodule" to 0
        )

        try {
            // Check for different image sources
            when {
                // Option 1: Images from CaptureActivity (region-specific)
                intent.hasExtra("forehead") -> {
                    val regions = listOf("forehead", "nose", "left_cheek", "right_cheek", "chin")

                    regions.forEach { region ->
                        intent.getByteArrayExtra(region)?.let { imageBytes ->
                            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                            imageList.add(bitmap)  // Store for GAGS calculation

                            }
                        }
                    }

                // Option 2: Images from SeverityActivity (multiple images)
                intent.hasExtra("selected_images") -> {
                    val selectedImagesCount = intent.getIntExtra("selected_images_count", 0)

                    for (i in 0 until selectedImagesCount) {
                        intent.getByteArrayExtra("selected_image_$i")?.let { imageBytes ->
                            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                            imageList.add(bitmap)  // Store for GAGS calculation

                            val counts = runInference(bitmap)
                            totalCounts.keys.forEach { key ->
                                totalCounts[key] = totalCounts.getOrDefault(key, 0) + counts.getOrDefault(key, 0)
                            }
                        }
                    }
                }

                // Option 3: Single image from RealTimeActivity
                intent.hasExtra("image_bytes") -> {
                    intent.getByteArrayExtra("image_bytes")?.let { imageBytes ->
                        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                        imageList.add(bitmap)  // Store for GAGS calculation

                        val counts = runInference(bitmap)
                        totalCounts.keys.forEach { key ->
                            totalCounts[key] = totalCounts.getOrDefault(key, 0) + counts.getOrDefault(key, 0)
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing images: ${e.message}")
            e.printStackTrace()
        }

        return Pair(imageList, totalCounts)
    }

    private fun runInference(bitmap: Bitmap): Map<String, Int> {
        val acneCounts = mutableMapOf(
            "comedone" to 0,
            "pustule" to 0,
            "papule" to 0,
            "nodule" to 0
        )

        try {
            val ortEnv = ortEnvironment ?: throw IllegalStateException("ORT environment is null")
            val ortSession = ortSession ?: throw IllegalStateException("ORT session is null")

            // Resize bitmap to expected input size
            val resizedBitmap = if (bitmap.width != modelInputWidth || bitmap.height != modelInputHeight) {
                Bitmap.createScaledBitmap(bitmap, modelInputWidth, modelInputHeight, true)
            } else {
                bitmap
            }

            // Create input tensor
            val inputBuffer = prepareInputBuffer(resizedBitmap)

            // Get input name (usually "images" for YOLOv9 models)
            val inputName = ortSession.inputInfo.keys.firstOrNull() ?: "images"

            // Create input shape (NCHW format: batch_size, channels, height, width)
            val shape = longArrayOf(1, modelInputChannels.toLong(), modelInputHeight.toLong(), modelInputWidth.toLong())

            // Create ONNX tensor from float buffer
            val inputTensor = OnnxTensor.createTensor(ortEnv, inputBuffer, shape)

            // Prepare input map
//            val inputs = mapOf(inputName to inputTensor)
            // Prepare input map
            val inputs = HashMap<String, OnnxTensor>()
            inputs[inputName] = inputTensor

            // Run inference
            Log.d(TAG, "Running ONNX inference")
            val output = ortSession.run(inputs)

            // Process the output to get acne counts
            processOnnxResults(output, acneCounts)

            // Clean up
            inputTensor.close()
            output.close()

        } catch (e: Exception) {
            Log.e(TAG, "Error during inference: ${e.message}")
            e.printStackTrace()
        }

        return acneCounts
    }

    private fun prepareInputBuffer(bitmap: Bitmap): FloatBuffer {
        // Allocate a float buffer for the input tensor (NCHW format)
        val bufferSize = modelInputChannels * modelInputHeight * modelInputWidth
        val floatBuffer = FloatBuffer.allocate(bufferSize)

        // Extract pixel values
        val pixels = IntArray(modelInputWidth * modelInputHeight)
        bitmap.getPixels(pixels, 0, modelInputWidth, 0, 0, modelInputWidth, modelInputHeight)

        // Extract and normalize RGB channels in NCHW format (channels first)
        val r = FloatArray(modelInputHeight * modelInputWidth)
        val g = FloatArray(modelInputHeight * modelInputWidth)
        val b = FloatArray(modelInputHeight * modelInputWidth)

        var pixelIndex = 0
        for (y in 0 until modelInputHeight) {
            for (x in 0 until modelInputWidth) {
                val pixel = pixels[pixelIndex]

                // Normalize to 0-1 range
                r[pixelIndex] = (pixel shr 16 and 0xFF) / 255.0f
                g[pixelIndex] = (pixel shr 8 and 0xFF) / 255.0f
                b[pixelIndex] = (pixel and 0xFF) / 255.0f

                pixelIndex++
            }
        }

        // Add all R values, then all G values, then all B values (NCHW format)
        for (i in 0 until modelInputHeight * modelInputWidth) {
            floatBuffer.put(r[i])
        }

        for (i in 0 until modelInputHeight * modelInputWidth) {
            floatBuffer.put(g[i])
        }

        for (i in 0 until modelInputHeight * modelInputWidth) {
            floatBuffer.put(b[i])
        }

        // Reset position to beginning
        floatBuffer.rewind()

        return floatBuffer
    }

    private fun processOnnxResults(output: OrtSession.Result, acneCounts: MutableMap<String, Int>) {
        val classNames = mapOf(
            0 to "comedone",
            1 to "pustule",
            2 to "papule",
            3 to "nodule"
        )

        val confidenceThreshold = 0.6f  // 🔹 Increased to reduce false positives
        val iouThreshold = 0.7f         // 🔹 Stricter NMS threshold
        val detectionsList = mutableListOf<ImageAnalyzer.Detection>()

        try {
            val outputTensor = output.firstOrNull()?.value as? OnnxTensor ?: return
            val shape = outputTensor.info.shape

            if (shape.size == 3 && shape[0] == 1L) {
                val rawData = outputTensor.floatBuffer
                val numDetections = shape[2].toInt()

                for (i in 0 until numDetections) {
                    val offset = i * 8
                    val confidence = rawData.get(offset + 4)

                    if (confidence < confidenceThreshold) continue  // 🔹 Skip low-confidence detections

                    val classIndex = rawData.get(offset + 5).toInt()
                    Log.d(TAG, "Detected class index: $classIndex, confidence: $confidence")  // 🔹 Debug

                    val className = classNames[classIndex] ?: continue

                    val boundingBox = ImageAnalyzer.BoundingBox(
                        x = rawData.get(offset),
                        y = rawData.get(offset + 1),
                        width = rawData.get(offset + 2),
                        height = rawData.get(offset + 3)
                    )

                    Log.d(TAG, "Bounding Box - X: ${boundingBox.x}, Y: ${boundingBox.y}, W: ${boundingBox.width}, H: ${boundingBox.height}")  // 🔹 Debug

                    detectionsList.add(
                        ImageAnalyzer.Detection(
                            classId = classIndex,
                            className = className,
                            confidence = confidence,
                            boundingBox = boundingBox
                        )
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing ONNX results: ${e.message}")
        }

        // 🔹 Log before NMS
        Log.d(TAG, "Detections before NMS: ${detectionsList.size}")

        // Apply NMS
        val filteredDetections = applyNMS(detectionsList, iouThreshold)

        // 🔹 Log after NMS
        Log.d(TAG, "Detections after NMS: ${filteredDetections.size}")

        // Update acne counts with filtered detections
        for (detection in filteredDetections) {
            acneCounts[detection.className] = acneCounts.getOrDefault(detection.className, 0) + 1
        }

        Log.d(TAG, "Final Acne Counts after NMS: $acneCounts")
    }

//        } catch (e: Exception) {
//            Log.e(TAG, "Error processing ONNX results: ${e.message}")
//        }
//
//        Log.d(TAG, "Final Acne Counts: $acneCounts")
//
//        this.detections = detectionsList
//    }

    private fun applyNMS(detections: List<ImageAnalyzer.Detection>, iouThreshold: Float): List<ImageAnalyzer.Detection> {
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val selectedDetections = mutableListOf<ImageAnalyzer.Detection>()

        for (detection in sortedDetections) {
            val keep = selectedDetections.none { existing ->
                calculateIoU(detection.boundingBox, existing.boundingBox) > iouThreshold
            }
            if (keep) {
                selectedDetections.add(detection)
            }
        }
        return selectedDetections
    }

    private fun calculateIoU(box1: ImageAnalyzer.BoundingBox, box2: ImageAnalyzer.BoundingBox): Float {
        val x1 = maxOf(box1.x, box2.x)
        val y1 = maxOf(box1.y, box2.y)
        val x2 = minOf(box1.x + box1.width, box2.x + box2.width)
        val y2 = minOf(box1.y + box1.height, box2.y + box2.height)

        val intersection = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val area1 = box1.width * box1.height
        val area2 = box2.width * box2.height
        val union = area1 + area2 - intersection

        return if (union > 0) intersection / union else 0f
    }

    private fun processMultipleImages(imageList: List<Bitmap>, gagsCalculator: GAGSCalculator) {
        val faceRegions = listOf(
            GAGSCalculator.FaceRegion.FOREHEAD,
            GAGSCalculator.FaceRegion.NOSE,
            GAGSCalculator.FaceRegion.RIGHT_CHEEK,
            GAGSCalculator.FaceRegion.LEFT_CHEEK,
            GAGSCalculator.FaceRegion.CHIN
        )

        if (imageList.size != faceRegions.size) {
            Log.e(TAG, "Error: Expected ${faceRegions.size} images, but got ${imageList.size}")
            return
        }

        var totalGAGSScore = 0
        val totalAcneCounts = mutableMapOf<String, Int>()

        // ✅ Proper mapping of detected acne types to GAGS lesion types
        val acneTypeMapping = mapOf(
            "NO_LESION" to GAGSCalculator.LesionType.NO_LESION,
            "comedone" to GAGSCalculator.LesionType.COMEDONE,
            "papule" to GAGSCalculator.LesionType.PAPULE,
            "pustule" to GAGSCalculator.LesionType.PUSTULE,
            "nodule" to GAGSCalculator.LesionType.NODULE_CYST  // Fix: Now correctly mapped
        )

        for ((index, image) in imageList.withIndex()) {
            val region = faceRegions[index]
            val acneCounts = runInference(image)  // Detect acne per image

            // Add acne counts to total
            for ((acneType, count) in acneCounts) {
                totalAcneCounts[acneType] = totalAcneCounts.getOrDefault(acneType, 0) + count
            }

            // ✅ Find dominant acne type (most frequent)
            val dominantAcne = acneCounts.maxByOrNull { it.value }?.key ?: "NO_LESION"
            val lesionType = acneTypeMapping[dominantAcne] ?: GAGSCalculator.LesionType.NO_LESION

            // ✅ Calculate GAGS score correctly
            val gagsScore = region.areaFactor * lesionType.score
            totalGAGSScore += gagsScore

            Log.d(TAG, "Region: ${region.name}, Weight: ${region.areaFactor}, Lesion Score: ${lesionType.score}")
            Log.d(TAG, "Region: ${region.name}, Dominant Acne: ${dominantAcne}, Score: $gagsScore")
        }

        // Store in Singleton
        GAGSData.totalGAGSScore = totalGAGSScore
        GAGSData.totalAcneCounts = totalAcneCounts

        severity = gagsCalculator.severityTotalGAGSMultipleImage(GAGSData.totalGAGSScore)

        Log.d(TAG, "Final GAGS Score: $totalGAGSScore, Severity: $severity")
        Log.d(TAG, "Total Acne Counts: $totalAcneCounts")
    }


    private fun displayResults(acneCounts: Map<String, Int>, totalGAGSScore: Int, severity: String) {
        val resultsText = StringBuilder()
        resultsText.append("Acne Severity: $severity ($totalGAGSScore)\n\n")
        resultsText.append("Acne Counts:\n")

        acneCounts.forEach { (type, count) ->
            resultsText.append("${type.capitalize()}: $count\n")
        }

        resultsText.append("\nTotal Acne Count: ${acneCounts.values.sum()}")

        // Update UI
        resultTextView.text = resultsText.toString()

        // Start ResultActivity with the analysis
        val intent = Intent(this, ResultActivity::class.java).apply {
            putExtra("severity", severity)
            putExtra("total_score", totalGAGSScore)
            putExtra("total_count", acneCounts.values.sum())
            putExtra("comedone_count", acneCounts["comedone"] ?: 0)
            putExtra("pustule_count", acneCounts["pustule"] ?: 0)
            putExtra("papule_count", acneCounts["papule"] ?: 0)
            putExtra("nodule_count", acneCounts["nodule"] ?: 0)
            putExtra("timestamp", System.currentTimeMillis())
        }

        startActivity(intent)
        finish()
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            ortSession?.close()
            ortEnvironment?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing ONNX resources: ${e.message}")
        }
    }
}
