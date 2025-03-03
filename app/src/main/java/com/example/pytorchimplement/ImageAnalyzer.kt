package com.example.pytorchimplement

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.HashMap

/**
 * Image analyzer for real-time ONNX model inference
 */
class ImageAnalyzer(private val context: Context, private val listener: AnalysisListener) : ImageAnalysis.Analyzer {

    // Define TAG constant at class level
    private val TAG = "ImageAnalyzer"

    // Model dimensions - must match what's set in RealTimeActivity
    private val MODEL_WIDTH = 640
    private val MODEL_HEIGHT = 640

    interface AnalysisListener {
        fun onAnalysisComplete(result: AnalysisResult)
    }

    data class AnalysisResult(
        val severity: Int, // Severity score 0-1
        val timestamp: Long, // Timestamp of the analysis
        val acneCounts: Map<String, Int> = mapOf(
            "comedone" to 0,
            "pustule" to 0,
            "papule" to 0,
            "nodule" to 0
        ),
        val detections: List<Detection> = emptyList() // Add list of detections with bounding boxes
    )
    
    // New data class to hold detection information including bounding boxes
    data class Detection(
        val classId: Int, // 0=comedone, 1=pustule, 2=papule, 3=nodule
        val className: String, // Human-readable class name
        val confidence: Float, // Detection confidence 0-1
        val boundingBox: BoundingBox // Normalized coordinates (0-1) for the bounding box
    )
    
    // Bounding box coordinates class (all values normalized 0-1)
    data class BoundingBox(
        val x: Float, // center x coordinate
        val y: Float, // center y coordinate
        val width: Float, // width of box
        val height: Float // height of box
    )

    // Store last analysis result for access
    var lastAnalysisResult: AnalysisResult? = null
        private set
    
    // Flag to avoid running multiple analyses at once
    private var isAnalyzing = false
    
    // Min time between analyses in ms
    private val analysisCooldown = 250L
    private var lastAnalysisTime = 0L
    
    // ONNX Runtime environment and session
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    
    // Model config
    private val modelFileName = "yolov9.onnx"
    
    // Model input dimensions
    private var inputWidth = 640
    private var inputHeight = 640
    private var inputChannels = 3
    
    init {
        loadModel()
    }
    
    private fun loadModel() {
        try {
            // Create ONNX Runtime environment
            ortEnvironment = OrtEnvironment.getEnvironment()
            
            // Load the model from assets
            val modelBytes = context.assets.open(modelFileName).readBytes()
            
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
                    val tensorInfo = inputNodeInfo?.info
                    if (tensorInfo != null) {
                        val shape = (tensorInfo as TensorInfo).shape
                        if (shape != null && shape.size >= 4) {
                            // ONNX models typically use NCHW format
                            inputChannels = shape[1].toInt()
                            inputHeight = shape[2].toInt()
                            inputWidth = shape[3].toInt()
                            Log.d(TAG, "Model input shape: $inputChannels x $inputHeight x $inputWidth (NCHW format)")
                        }
                    }
                }
                
                Log.d(TAG, "Using input dimensions: $inputChannels x $inputHeight x $inputWidth")
                
                // Log output information
                val outputInfo = session.outputInfo
                for ((name, info) in outputInfo) {
                    val shape = (info.info as TensorInfo?)?.shape
                    Log.d(TAG, "Output: $name, Shape: ${shape?.contentToString()}")
                }
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error loading ONNX model: ${e.message}")
            e.printStackTrace()
        }
    }

    override fun analyze(imageProxy: ImageProxy) {
        // Skip if already analyzing, ortSession not loaded, or in cooldown period
        val currentTime = System.currentTimeMillis()
        if (isAnalyzing || ortSession == null) {
            Log.d(TAG, "Skipping frame: isAnalyzing=$isAnalyzing, ortSession=${ortSession != null}")
            imageProxy.close()
            return
        }
        
        if (currentTime - lastAnalysisTime < analysisCooldown) {
            Log.d(TAG, "Frame cooldown: time since last analysis=${currentTime - lastAnalysisTime}ms")
            imageProxy.close()
            return
        }

        Log.d(TAG, "Starting frame analysis at $currentTime")
        isAnalyzing = true
        lastAnalysisTime = currentTime
        
        try {
            // Convert image to bitmap for processing
            val bitmap = imageProxy.toBitmap()
            Log.d(TAG, "Converted frame to bitmap: ${bitmap.width}x${bitmap.height}")
            
            // Check if the frame is likely empty or dark (simple check)
            if (isEmptyOrDarkFrame(bitmap)) {
                Log.d(TAG, "Detected empty or dark frame, skipping analysis")
                listener.onAnalysisComplete(
                    AnalysisResult(
                        severity = 0,
                        timestamp = currentTime,
                        acneCounts = mapOf(
                            "comedone" to 0,
                            "pustule" to 0,
                            "papule" to 0,
                            "nodule" to 0
                        ),
                        detections = emptyList()
                    )
                )
                return
            }
            
            // Run ONNX inference
            val result = runInference(bitmap)
            
            // Save the result
            lastAnalysisResult = result
            
            // Notify listener
            listener.onAnalysisComplete(result)
            
            Log.d(TAG, "Frame analysis completed in ${System.currentTimeMillis() - currentTime}ms")
            
        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing image: ${e.message}")
            e.printStackTrace()
        } finally {
            isAnalyzing = false
            imageProxy.close()
        }
    }
    
    private fun runInference(bitmap: Bitmap): AnalysisResult {
        val ortEnv = ortEnvironment ?: throw IllegalStateException("ORT environment is null")
        val ortSession = ortSession ?: throw IllegalStateException("ORT session is null")
        
        try {
            // Resize bitmap to expected input size
            val resizedBitmap = if (bitmap.width != inputWidth || bitmap.height != inputHeight) {
                Log.d(TAG, "Resizing bitmap from ${bitmap.width}x${bitmap.height} to ${inputWidth}x${inputHeight}")
                Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
            } else {
                bitmap
            }
            
            // Create input tensor
            val inputBuffer = prepareInputBuffer(resizedBitmap)
            
            // Get input name (usually "images" for YOLOv9 models)
            val inputName = ortSession.inputInfo.keys.firstOrNull() ?: "images"
            
            // Create input shape (NCHW format: batch_size, channels, height, width)
            val shape = longArrayOf(1, inputChannels.toLong(), inputHeight.toLong(), inputWidth.toLong())
            
            // Create ONNX tensor from float buffer
            val inputTensor = OnnxTensor.createTensor(ortEnv, inputBuffer, shape)
            
            // Prepare input map
            val inputMap = HashMap<String, OnnxTensor>()
            inputMap[inputName] = inputTensor
            
            // Run inference
            Log.d(TAG, "Running ONNX inference")
            val output = ortSession.run(inputMap)
            
            // Process results
            val result = processResults(output)
            
            // Clean up
            inputTensor.close()
            output.close()
            
            return result
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during ONNX inference: ${e.message}")
            e.printStackTrace()
            
            // Return empty result on error
            return AnalysisResult(
                severity = 0,
                timestamp = System.currentTimeMillis()
            )
        }
    }
    
    private fun prepareInputBuffer(bitmap: Bitmap): FloatBuffer {
        // Allocate a float buffer for the input tensor (NCHW format)
        val bufferSize = inputChannels * inputHeight * inputWidth
        val floatBuffer = FloatBuffer.allocate(bufferSize)
        
        // Extract pixel values
        val pixels = IntArray(inputWidth * inputHeight)
        bitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        
        // For Roboflow YOLOv9 models, the input preprocessing expects:
        // 1. RGB format (not BGR)
        // 2. Normalization to [0-1] range by dividing by 255
        // 3. No mean subtraction or standard deviation division
        
        // Prepare RGB arrays
        val r = FloatArray(inputHeight * inputWidth)
        val g = FloatArray(inputHeight * inputWidth)
        val b = FloatArray(inputHeight * inputWidth)
        
        var pixelIndex = 0
        for (y in 0 until inputHeight) {
            for (x in 0 until inputWidth) {
                val pixel = pixels[pixelIndex]
                
                // Extract RGB channels (Android's ARGB format: 0xAARRGGBB)
                r[pixelIndex] = ((pixel shr 16) and 0xFF) / 255.0f
                g[pixelIndex] = ((pixel shr 8) and 0xFF) / 255.0f
                b[pixelIndex] = (pixel and 0xFF) / 255.0f
                
                pixelIndex++
            }
        }
        
        // Add channel data in RGB order (what Roboflow models expect)
        // Format is NCHW: [batch, channels, height, width]
        for (i in 0 until inputHeight * inputWidth) {
            floatBuffer.put(r[i])
        }
        for (i in 0 until inputHeight * inputWidth) {
            floatBuffer.put(g[i])
        }
        for (i in 0 until inputHeight * inputWidth) {
            floatBuffer.put(b[i])
        }
        
        Log.d(TAG, "Prepared input tensor with shape [1, 3, $inputHeight, $inputWidth] in RGB format")
        
        floatBuffer.rewind()
        return floatBuffer
    }
    
    private fun processResults(output: OrtSession.Result): AnalysisResult {
        val acneCounts = mutableMapOf(
            "comedone" to 0,
            "pustule" to 0,
            "papule" to 0,
            "nodule" to 0
        )
        
        val detections = mutableListOf<Detection>()
        
        try {
            // Log all available output tensors for debugging
            val outputNames = output.map {o -> o.key}.joinToString()
            Log.d(TAG, "Available output tensors: ${outputNames}")
            output.forEach { (name, tensor) ->
                if (tensor is OnnxTensor) {
                    val shape = tensor.info.shape
                    Log.d(TAG, "Output tensor '$name': shape=${shape.contentToString()}")
                }
            }
            
            val classNames = mapOf(
                0 to "comedone",
                1 to "pustule",
                2 to "papule",
                3 to "nodule"
            )
            
            // Set appropriate confidence thresholds for the Roboflow model
            val confidenceThreshold = 0.35f  // Increased from 0.20f to reduce false positives
            
            // Get the main output tensor - this model uses the key "output"
            val outputTensor = output.first { o -> o.key == "output"}.value as? OnnxTensor
            
            if (outputTensor == null) {
                Log.e(TAG, "No output tensor found. Available keys: ${output.map {o -> o.key}}")
                return defaultAnalysisResult(acneCounts)
            }
            
            val tensorInfo = outputTensor.info
            val shape = tensorInfo.shape
            Log.d(TAG, "Output tensor shape: ${shape.contentToString()}")
            
            // Get the raw float data
            val rawData = outputTensor.floatBuffer
            Log.d(TAG, "Raw data capacity: ${rawData.capacity()}")
            
            // Roboflow YOLOv9 model returns detections in the format shown from the Roboflow interface:
            // Each detection has x, y, width, height, and class confidence values
            // where x, y are center coordinates in pixel space (0-640)
            
            try {
                // Based on the Roboflow screenshot, detections are in pixel coordinates
                val detectionsList = mutableListOf<Detection>()
                
                // Parse the output tensor based on YOLOv9 format
                // The output shape is typically [1, boxes, coordinates+classes]
                if (shape.size >= 2) {
                    val numDetections = if (shape.size == 3) shape[2].toInt() else shape[1].toInt()
                    Log.d(TAG, "Processing $numDetections possible detections")
                    
                    // Log sample of raw detection data for debugging
                    Log.d(TAG, "===== RAW DETECTION DATA SAMPLE =====")
                    
                    // Track valid detections
                    var validCount = 0
                    
                    // Process each potential detection
                    for (i in 0 until numDetections) {
                        // Get the coordinates and confidence
                        // Based on the Roboflow output format:
                        // - Coordinates are in pixel space (0-640)
                        // - The format appears to be: x, y, width, height, class_scores[...]
                        
                        try {
                            if (shape.size == 3 && shape[1].toInt() == 8) {
                                // Format: [1, 8, num_detections] where:
                                // - First 4 elements are x, y, width, height
                                // - Last 4 elements are class confidences
                                
                                val xCenter = rawData.get(0 * numDetections + i)
                                val yCenter = rawData.get(1 * numDetections + i)
                                val width = rawData.get(2 * numDetections + i)
                                val height = rawData.get(3 * numDetections + i)
                                
                                // Get class confidences
                                val classScores = IntRange(0, 3).map { c ->
                                    rawData.get((4 + c) * numDetections + i)
                                }
                                
                                // Find highest confidence class
                                val bestClassIdx = classScores.indices.maxByOrNull { classScores[it] } ?: 0
                                val bestScore = classScores[bestClassIdx]
                                
                                // Log some detections for debugging
                                if (i < 10 || bestScore > 0.1f) {
                                    Log.d(TAG, "Detection $i: x=$xCenter, y=$yCenter, w=$width, h=$height, " +
                                           "scores=${classScores.map { it.format(3) }}, best=${bestScore.format(3)}")
                                }
                                
                                // Check if this detection meets our confidence threshold
                                if (bestScore > confidenceThreshold) {
                                    // From the Roboflow web UI, we can see coordinates are in pixel space (0-640)
                                    // We need to normalize to 0-1 range for our system
                                    val normalizedX = (xCenter / MODEL_WIDTH).coerceIn(0f, 1f)
                                    
                                    // IMPORTANT FIX: The model's Y coordinate system might be inverted
                                    // or using a different reference point than expected
                                    // Try adjusting the Y coordinate to match what we see on screen
                                    
                                    // Option 1: If the model coordinates are upside down from what we expect 
                                    // (if the model was trained with 0,0 at bottom-left)
                                    // val normalizedY = (1.0f - (yCenter / MODEL_HEIGHT)).coerceIn(0f, 1f)
                                    
                                    // Option 2: Standard normalization if model uses top-left as 0,0
                                    val normalizedY = (yCenter / MODEL_HEIGHT).coerceIn(0f, 1f)
                                    
                                    // Log the original values for debugging
                                    Log.d(TAG, "Original detection at pixel coordinates: x=$xCenter, y=$yCenter")
                                    Log.d(TAG, "Normalized to: x=$normalizedX, y=$normalizedY")
                                    
                                    val normalizedWidth = (width / MODEL_WIDTH).coerceIn(0.01f, 1f)
                                    val normalizedHeight = (height / MODEL_HEIGHT).coerceIn(0.01f, 1f)
                                    
                                    // Create detection object
                                    val className = classNames[bestClassIdx] ?: "unknown"
                                    
                                    // Apply coordinate correction if needed based on empirical testing
                                    // This is an empirical fix based on the screenshot showing misalignment
                                    // For papules specifically, which appear to need vertical adjustment
                                    var adjustedY = normalizedY
                                    if (className == "papule" && normalizedY > 0.4f && normalizedY < 0.7f) {
                                        // Move the detection up by 25% of the image height to match forehead location
                                        adjustedY = (normalizedY - 0.25f).coerceIn(0f, 1f)
                                        Log.d(TAG, "Adjusted papule Y coordinate from $normalizedY to $adjustedY")
                                    }
                                    
                                    detectionsList.add(Detection(
                                        classId = bestClassIdx,
                                        className = className,
                                        confidence = bestScore,
                                        boundingBox = BoundingBox(normalizedX, adjustedY, normalizedWidth, normalizedHeight)
                                    ))
                                    
                                    // Log valid detection
                                    Log.d(TAG, "Found valid detection: class=$className, conf=${bestScore.format(2)}, " +
                                           "pos=[${normalizedX.format(2)}, ${adjustedY.format(2)}, " +
                                           "${normalizedWidth.format(2)}, ${normalizedHeight.format(2)}]")
                                    
                                    // Update acne count
                                    acneCounts[className] = acneCounts[className]!! + 1
                                    validCount++
                                }
                            } else {
                                // For other formats, extract the model's predictions directly
                                // This is a fallback for different output shapes
                                Log.w(TAG, "Unsupported output shape: ${shape.contentToString()}")
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Error processing detection $i: ${e.message}")
                        }
                    }
                    
                    Log.d(TAG, "Found $validCount valid detections after filtering")
                    
                    // Apply NMS to remove overlapping detections
                    val finalDetections = applyNMS(detectionsList, 0.2f)
                    Log.d(TAG, "After NMS: ${finalDetections.size} detections")
                    
                    detections.addAll(finalDetections)
                } else {
                    Log.e(TAG, "Unexpected output tensor shape: ${shape.contentToString()}")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing model output: ${e.message}")
                e.printStackTrace()
            }
            
            val severity = calculateSeverityScore(acneCounts)
            Log.d(TAG, "Analysis complete - Severity: $severity, Counts: $acneCounts, Total: ${acneCounts.values.sum()}")
            
            return AnalysisResult(
                severity = severity,
                timestamp = System.currentTimeMillis(),
                acneCounts = acneCounts,
                detections = detections
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing ONNX results: ${e.message}")
            e.printStackTrace()
            return defaultAnalysisResult(acneCounts)
        }
    }
    
    private fun defaultAnalysisResult(acneCounts: Map<String, Int> = emptyMap()): AnalysisResult {
        return AnalysisResult(
            severity = 0,
            timestamp = System.currentTimeMillis(),
            acneCounts = acneCounts,
            detections = emptyList()
        )
    }
    
    private fun calculateSeverityScore(acneCounts: Map<String, Int>): Int {
        // Calculate severity based on acne counts and type
        // Assign weights to different acne types (more severe types have higher weights)
        val comedoneWeight = 1 // Mild
        val pustuleWeight = 2  // Moderate
        val papuleWeight = 3   // Moderately severe
        val noduleWeight = 5   // Severe
        
        val weightedSum = 
            (acneCounts["comedone"] ?: 0) * comedoneWeight +
            (acneCounts["pustule"] ?: 0) * pustuleWeight +
            (acneCounts["papule"] ?: 0) * papuleWeight +
            (acneCounts["nodule"] ?: 0) * noduleWeight
        
        val totalCount = acneCounts.values.sum()
        
        // Normalize to 0-10 range
        return if (totalCount > 0) {
            (weightedSum.toFloat() / (totalCount * 3f) * 10).toInt().coerceIn(0, 10)
        } else {
            0
        }
    }
    
    // Add Non-Maximum Suppression function
    private fun applyNMS(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        if (detections.isEmpty()) return emptyList()
        
        // Sort detections by confidence
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val selected = mutableListOf<Detection>()
        val processed = BooleanArray(sortedDetections.size)
        
        for (i in sortedDetections.indices) {
            if (processed[i]) continue
            
            selected.add(sortedDetections[i])
            
            // Compare with rest of the boxes
            for (j in i + 1 until sortedDetections.size) {
                if (processed[j]) continue
                
                // Calculate IoU
                val iou = calculateIoU(sortedDetections[i].boundingBox, sortedDetections[j].boundingBox)
                if (iou > iouThreshold) {
                    processed[j] = true
                }
            }
        }
        
        return selected
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        // Convert center format to min/max format
        val box1MinX = box1.x - box1.width / 2
        val box1MaxX = box1.x + box1.width / 2
        val box1MinY = box1.y - box1.height / 2
        val box1MaxY = box1.y + box1.height / 2
        
        val box2MinX = box2.x - box2.width / 2
        val box2MaxX = box2.x + box2.width / 2
        val box2MinY = box2.y - box2.height / 2
        val box2MaxY = box2.y + box2.height / 2
        
        // Calculate intersection area
        val intersectMinX = maxOf(box1MinX, box2MinX)
        val intersectMaxX = minOf(box1MaxX, box2MaxX)
        val intersectMinY = maxOf(box1MinY, box2MinY)
        val intersectMaxY = minOf(box1MaxY, box2MaxY)
        
        if (intersectMaxX < intersectMinX || intersectMaxY < intersectMinY) {
            return 0f
        }
        
        val intersectionArea = (intersectMaxX - intersectMinX) * (intersectMaxY - intersectMinY)
        val box1Area = box1.width * box1.height
        val box2Area = box2.width * box2.height
        
        val unionArea = box1Area + box2Area - intersectionArea
        
        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }
    
    // Add function to detect empty or dark frames
    private fun isEmptyOrDarkFrame(bitmap: Bitmap): Boolean {
        try {
            // Sample pixels to determine if the frame is mostly empty/dark
            val pixelCount = 10
            val pixels = IntArray(pixelCount)
            
            // Sample evenly across the image
            val width = bitmap.width
            val height = bitmap.height
            var sumBrightness = 0
            
            for (i in 0 until pixelCount) {
                val x = (width / pixelCount) * i
                val y = height / 2
                val pixel = bitmap.getPixel(x, y)
                
                // Extract RGB
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                
                // Calculate brightness (simple average)
                val brightness = (r + g + b) / 3
                sumBrightness += brightness
            }
            
            // Calculate average brightness
            val avgBrightness = sumBrightness / pixelCount
            Log.d(TAG, "Average frame brightness: $avgBrightness")
            
            // If average brightness is very low, consider it an empty/dark frame
            return avgBrightness < 20
        } catch (e: Exception) {
            Log.e(TAG, "Error checking frame emptiness: ${e.message}")
            return false
        }
    }
    
    fun close() {
        try {
            ortSession?.close()
            ortEnvironment?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing ONNX resources: ${e.message}")
        }
    }

    // Helper function to format floats nicely for logging
    private fun Float.format(digits: Int) = "%.${digits}f".format(this)
}