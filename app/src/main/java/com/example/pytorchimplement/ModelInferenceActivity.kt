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
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import android.content.Intent

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
        val acneCounts = processImagesFromIntent()
        
        // Display results
        displayResults(acneCounts)
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
    
    private fun processImagesFromIntent(): Map<String, Int> {
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
                    // Facial region images (categorized approach)
                    val regions = listOf("forehead", "nose", "left_cheek", "right_cheek", "chin")
                    
                    regions.forEach { region ->
                        intent.getByteArrayExtra(region)?.let { imageBytes ->
                            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                            val regionCounts = runInference(bitmap)
                            
                            // Add to total counts
                            totalCounts.keys.forEach { key ->
                                totalCounts[key] = totalCounts[key]!! + regionCounts[key]!!
                            }
                        }
                    }
                }
                
                // Option 2: Images from SeverityActivity (multiple images)
                intent.hasExtra("selected_images") -> {
                    // Get byte array list instead of byte array array
                    val selectedImagesCount = intent.getIntExtra("selected_images_count", 0)
                    
                    // Process each image byte array individually
                    for (i in 0 until selectedImagesCount) {
                        val imageBytes = intent.getByteArrayExtra("selected_image_$i")
                        imageBytes?.let {
                            val bitmap = BitmapFactory.decodeByteArray(it, 0, it.size)
                            val counts = runInference(bitmap)
                            
                            totalCounts.keys.forEach { key ->
                                totalCounts[key] = totalCounts[key]!! + counts[key]!!
                            }
                        }
                    }
                }
                
                // Option 3: Single image from RealTimeActivity
                intent.hasExtra("image_bytes") -> {
                    val imageBytes = intent.getByteArrayExtra("image_bytes")
                    imageBytes?.let {
                        val bitmap = BitmapFactory.decodeByteArray(it, 0, it.size)
                        val counts = runInference(bitmap)
                        
                        totalCounts.keys.forEach { key ->
                            totalCounts[key] = totalCounts[key]!! + counts[key]!!
                        }
                    }
                }
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing images: ${e.message}")
            e.printStackTrace()
        }
        
        return totalCounts
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
            val inputs = mapOf(inputName to inputTensor)
            
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
        // Define class names mapping
        val classNames = mapOf(
            0 to "comedone",
            1 to "pustule",
            2 to "papule",
            3 to "nodule"
        )
        
        // Detection confidence threshold
        val confidenceThreshold = 0.2f
        
        // List to store detections
        val detectionsList = mutableListOf<ImageAnalyzer.Detection>()
        
        try {
            // Get output tensor (YOLOv9 output tensor name is typically "output0" or "output")
            val outputTensor = output.first { o -> o.key == "output" }

            if (outputTensor == null) {
                Log.e(TAG, "No output tensor found")
                return
            }
            
            // Process output tensor based on its shape
            when (val tensor = outputTensor) {
                is OnnxTensor -> {
                    val shape = tensor.info.shape
                    Log.d(TAG, "Output tensor shape: ${shape.contentToString()}")
                    
                    // YOLOv9 typical output shape is [1, 84, num_detections] or similar
                    if (shape.size == 3) {
                        val numClasses = 4 // We have 4 acne classes
                        val numDetections = shape[2].toInt()
                        
                        // Get the raw float data from the tensor
                        val rawData = tensor.floatBuffer
                        Log.d(TAG, "Raw data capacity: ${rawData.capacity()}")
                        
                        // Process each detection
                        for (i in 0 until numDetections) {
                            // Format is typically [cx, cy, w, h, conf, class_prob1, class_prob2, ...]
                            
                            // In YOLO outputs, box coordinates are usually the first 4 values
                            val xIndex = i * (5 + numClasses)
                            
                            // Ensure we're not out of bounds
                            if (xIndex + 4 + numClasses >= rawData.capacity()) {
                                continue
                            }
                            
                            // Extract confidence
                            val confidence = rawData.get(xIndex + 4)
                            
                            // Skip low confidence detections
                            if (confidence < confidenceThreshold) {
                                continue
                            }
                            
                            // Find class with highest probability
                            var maxClassProb = 0f
                            var maxClassId = 0
                            for (c in 0 until numClasses) {
                                val classProb = rawData.get(xIndex + 5 + c)
                                if (classProb > maxClassProb) {
                                    maxClassProb = classProb
                                    maxClassId = c
                                }
                            }
                            
                            // Update counts
                            val className = classNames[maxClassId] ?: continue
                            acneCounts[className] = acneCounts[className]!! + 1
                            
                            // Extract bounding box coordinates
                            val centerX = rawData.get(xIndex)
                            val centerY = rawData.get(xIndex + 1)
                            val width = rawData.get(xIndex + 2)
                            val height = rawData.get(xIndex + 3)
                            
                            // Create normalized bounding box (0-1 range)
                            val boundingBox = ImageAnalyzer.BoundingBox(
                                x = centerX / modelInputWidth,
                                y = centerY / modelInputHeight,
                                width = width / modelInputWidth,
                                height = height / modelInputHeight
                            )
                            
                            // Create detection object
                            val detection = ImageAnalyzer.Detection(
                                classId = maxClassId,
                                className = className,
                                confidence = maxClassProb,
                                boundingBox = boundingBox
                            )
                            
                            // Add to detections list
                            detectionsList.add(detection)
                        }
                    }
                }
            }
            
            // Store detections for GAGS calculation
            this.detections = detectionsList
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing ONNX results: ${e.message}")
            e.printStackTrace()
        }
    }

    private fun displayResults(acneCounts: Map<String, Int>) {
        // Calculate severity
        val gagsCalculator = GAGSCalculator()
        
        // If we have detections with bounding boxes, use GAGS
        val normalizedSeverity = if (detections.isNotEmpty()) {
            gagsCalculator.calculateGAGSScore(detections)
        } else {
            // Fallback to simple weight-based calculation
            val weights = mapOf(
                "comedone" to 0.25f,
                "pustule" to 0.5f,
                "papule" to 0.75f,
                "nodule" to 1.0f
            )
            
            val totalCount = acneCounts.values.sum()
            var weightedScore = 0f
            
            acneCounts.forEach { (type, count) ->
                weightedScore += count * (weights[type] ?: 0f)
            }
            
            // Calculate as float first
            val normalizedSeverityFloat = if (totalCount > 0) {
                (weightedScore / totalCount).coerceIn(0f, 1f)
            } else {
                0f
            }
            
            // Convert to integer scale (0-10)
            (normalizedSeverityFloat * 10).toInt()
        }
        
        // Get severity description from GAGS calculator
        val severityLevel = if (detections.isNotEmpty()) {
            gagsCalculator.getSeverityDescription(normalizedSeverity)
        } else {
            when {
                normalizedSeverity < 3 -> "Low"
                normalizedSeverity < 7 -> "Medium"
                else -> "High"
            }
        }
        
        // Prepare display text
        val resultsText = StringBuilder()
        resultsText.append("Acne Severity: $severityLevel ($normalizedSeverity/10)\n\n")
        resultsText.append("Acne Counts:\n")
        
        acneCounts.forEach { (type, count) ->
            resultsText.append("${type.capitalize()}: $count\n")
        }
        
        resultsText.append("\nTotal Acne Count: ${acneCounts.values.sum()}")
        
        // Update the UI
        resultTextView.text = resultsText.toString()
        
        // Start Result activity with the analysis
        val intent = Intent(this, ResultActivity::class.java)
        intent.putExtra("severity", normalizedSeverity)
        intent.putExtra("total_count", acneCounts.values.sum())
        intent.putExtra("comedone_count", acneCounts["comedone"] ?: 0)
        intent.putExtra("pustule_count", acneCounts["pustule"] ?: 0)
        intent.putExtra("papule_count", acneCounts["papule"] ?: 0)
        intent.putExtra("nodule_count", acneCounts["nodule"] ?: 0)
        intent.putExtra("timestamp", System.currentTimeMillis())
        
        // Get image bytes from any of the potential sources
        val imageBytes = when {
            intent.hasExtra("image_bytes") -> intent.getByteArrayExtra("image_bytes")
            intent.hasExtra("forehead") -> intent.getByteArrayExtra("forehead") // Just use forehead as example
            intent.hasExtra("selected_image_0") -> intent.getByteArrayExtra("selected_image_0")
            else -> null
        }
        
        if (imageBytes != null) {
            intent.putExtra("image", imageBytes)
        }
        
        // Add detection information if available
        // This would come from more advanced detection code
        
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
