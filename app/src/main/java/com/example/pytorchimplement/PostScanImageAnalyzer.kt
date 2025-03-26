package com.example.pytorchimplement

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
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
 * Image analyzer for post-scan ONNX model inference on captured facial region images
 */
class PostScanImageAnalyzer(private val context: Context) {

    // Define TAG constant at class level
    private val TAG = "PostScanImageAnalyzer"

    // Model dimensions
    private val MODEL_WIDTH = 640
    private val MODEL_HEIGHT = 640
    private val MODEL_CHANNELS = 3

    // Model Name - Change this to the name of the model you are using
    private val MODEL_NAME = "v9s-10e.onnx"

    // ONNX Runtime environment and session
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    // Facial regions
    enum class FacialRegion(val id: String, val displayName: String, val areaFactor: Int) {
        FOREHEAD("forehead", "Forehead", 2),
        NOSE("nose", "Nose", 1),
        RIGHT_CHEEK("right_cheek", "Right Cheek", 2),
        LEFT_CHEEK("left_cheek", "Left Cheek", 2),
        CHIN("chin", "Chin", 1)
    }

    // Data class to hold detection information including bounding boxes
    data class Detection(
        val classId: Int, // 0=comedone, 1=pustule, 2=papule, 3=nodule
        val className: String, // Human-readable class name
        val confidence: Float, // Detection confidence 0-1
        val boundingBox: BoundingBox, // Normalized coordinates (0-1) for the bounding box
        val region: FacialRegion // The facial region this detection belongs to
    )
    
    // Bounding box coordinates class (all values normalized 0-1)
    data class BoundingBox(
        val x: Float, // center x coordinate
        val y: Float, // center y coordinate
        val width: Float, // width of box
        val height: Float // height of box
    )

    // Analysis result for a single region
    data class RegionAnalysisResult(
        val region: FacialRegion,
        val acneCounts: Map<String, Int>,
        val detections: List<Detection>,
        val dominantAcneType: String,
        val regionScore: Int,
        val inferenceTimeMs: Long
    )

    // Overall analysis result
    data class AnalysisResult(
        val timestamp: Long,
        val regionResults: Map<FacialRegion, RegionAnalysisResult>,
        val totalAcneCounts: Map<String, Int>,
        val totalGAGSScore: Int,
        val severity: String,
        val totalInferenceTimeMs: Long
    )

    init {
        loadModel()
    }
    
    private fun loadModel() {
        try {
            // Create ONNX Runtime environment
            ortEnvironment = OrtEnvironment.getEnvironment()
            
            // Load the model from assets
            val modelBytes = context.assets.open(MODEL_NAME).readBytes()
            
            // Create session options
            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setIntraOpNumThreads(4)
            
            // Create inference session
            ortSession = ortEnvironment?.createSession(modelBytes, sessionOptions)
            
            Log.d(TAG, "ONNX model loaded successfully: $MODEL_NAME")
            
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
                            Log.d(TAG, "Model input shape: ${shape.contentToString()}")
                        }
                    }
                }
                
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

    /**
     * Analyze all facial region images and calculate GAGS score
     * @param images Map of facial region ID to bitmap image
     * @return Overall analysis result with GAGS score and detections
     */
    fun analyzeAllRegions(images: Map<String, Bitmap>): AnalysisResult {
        val regionResults = mutableMapOf<FacialRegion, RegionAnalysisResult>()
        val totalAcneCounts = mutableMapOf(
            "comedone" to 0,
            "pustule" to 0,
            "papule" to 0,
            "nodule" to 0
        )
        var totalGAGSScore = 0
        var totalInferenceTimeMs = 0L

        // Process each region
        FacialRegion.values().forEach { region ->
            val bitmap = images[region.id]
            if (bitmap != null) {
                val result = analyzeRegion(bitmap, region)
                regionResults[region] = result

                // Add to total counts
                result.acneCounts.forEach { (acneType, count) ->
                    totalAcneCounts[acneType] = totalAcneCounts.getOrDefault(acneType, 0) + count
                }

                // Add to total GAGS score
                totalGAGSScore += result.regionScore

                // Add to total inference time
                totalInferenceTimeMs += result.inferenceTimeMs

                Log.d(TAG, "Region ${region.displayName}: Score=${result.regionScore}, " +
                        "Dominant=${result.dominantAcneType}, Counts=${result.acneCounts}, " +
                        "Inference time=${result.inferenceTimeMs}ms")
            } else {
                Log.e(TAG, "Missing image for region: ${region.displayName}")
            }
        }

        // Calculate severity based on total GAGS score
        val severity = calculateSeverityFromGAGS(totalGAGSScore)

        Log.d(TAG, "Total GAGS Score: $totalGAGSScore, Severity: $severity")
        Log.d(TAG, "Total Acne Counts: $totalAcneCounts")
        Log.d(TAG, "Total inference time: ${totalInferenceTimeMs}ms")

        return AnalysisResult(
            timestamp = System.currentTimeMillis(),
            regionResults = regionResults,
            totalAcneCounts = totalAcneCounts,
            totalGAGSScore = totalGAGSScore,
            severity = severity,
            totalInferenceTimeMs = totalInferenceTimeMs
        )
    }

    /**
     * Analyze a single facial region image
     * @param bitmap The image to analyze
     * @param region The facial region this image represents
     * @return Analysis result for this region
     */
    private fun analyzeRegion(bitmap: Bitmap, region: FacialRegion): RegionAnalysisResult {
        val acneCounts = mutableMapOf(
            "comedone" to 0,
            "pustule" to 0,
            "papule" to 0,
            "nodule" to 0
        )

        val severityMap = mapOf(
            "comedone" to 1,
            "papule" to 2,
            "pustule" to 3,
            "nodule" to 4
        )

        // Get detections and inference time
        val (detections, inferenceTimeMs) = runInference(bitmap, region)
        
        // Count detections by type
        detections.forEach { detection ->
            acneCounts[detection.className] = acneCounts.getOrDefault(detection.className, 0) + 1
        }

        // Find most severe acne type (highest severity score)
        val dominantAcne = acneCounts.entries
            .filter { it.value > 0 }
            .maxByOrNull { severityMap[it.key] ?: 0 }?.key ?: "NO_LESION"


        // Calculate region score based on GAGS methodology
        val lesionScore = when (dominantAcne) {
            "comedone" -> 1
            "papule" -> 2
            "pustule" -> 3
            "nodule" -> 4
            else -> 0
        }
        
        val regionScore = region.areaFactor * lesionScore
        
        return RegionAnalysisResult(
            region = region,
            acneCounts = acneCounts,
            detections = detections,
            dominantAcneType = dominantAcne,
            regionScore = regionScore,
            inferenceTimeMs = inferenceTimeMs
        )
    }
    
    /**
     * Run ONNX inference on a single image
     * @param bitmap The image to analyze
     * @param region The facial region this image represents
     * @return List of detections found in the image
     */
    private fun runInference(bitmap: Bitmap, region: FacialRegion): Pair<List<Detection>, Long> {
        val detections = mutableListOf<Detection>()
        var inferenceTimeMs = 0L
        
        try {
            val ortEnv = ortEnvironment ?: throw IllegalStateException("ORT environment is null")
            val ortSession = ortSession ?: throw IllegalStateException("ORT session is null")
            
            // Resize bitmap to expected input size
            val resizedBitmap = if (bitmap.width != MODEL_WIDTH || bitmap.height != MODEL_HEIGHT) {
                Bitmap.createScaledBitmap(bitmap, MODEL_WIDTH, MODEL_HEIGHT, true)
            } else {
                bitmap
            }
            
            // Create input tensor
            val inputBuffer = prepareInputBuffer(resizedBitmap)
            
            // Get input name (usually "images" for YOLOv9 models)
            val inputName = ortSession.inputInfo.keys.firstOrNull() ?: "images"
            
            // Create input shape (NCHW format: batch_size, channels, height, width)
            val shape = longArrayOf(1, MODEL_CHANNELS.toLong(), MODEL_HEIGHT.toLong(), MODEL_WIDTH.toLong())
            
            // Create ONNX tensor from float buffer
            val inputTensor = OnnxTensor.createTensor(ortEnv, inputBuffer, shape)
            
            // Prepare input map
            val inputs = HashMap<String, OnnxTensor>()
            inputs[inputName] = inputTensor
            
            // Run inference and time it
            Log.d(TAG, "Running ONNX inference on ${region.displayName}")
            val startTime = System.currentTimeMillis()
            val output = ortSession.run(inputs)
            inferenceTimeMs = System.currentTimeMillis() - startTime
            Log.d(TAG, "Inference completed in ${inferenceTimeMs}ms")
            
            // Process results
            val classNames = mapOf(
                0 to "comedone",
                3 to "pustule",
                2 to "papule",
                1 to "nodule"
            )
            
            val confidenceThreshold = 0.1f
            
            try {
                // Get the main output tensor - this model uses the key "output"
                val outputTensor = output.first { o -> o.key == "output"}.value as? OnnxTensor
                
                if (outputTensor != null) {
                    val tensorInfo = outputTensor.info
                    val shape = tensorInfo.shape
                    Log.d(TAG, "Output tensor shape: ${shape.contentToString()}")
                    
                    // Get the raw float data
                    val rawData = outputTensor.floatBuffer
                    
                    // Parse the output tensor based on YOLOv9 format
                    if (shape.size >= 2) {
                        val numDetections = if (shape.size == 3) shape[2].toInt() else shape[1].toInt()
                        Log.d(TAG, "Processing $numDetections possible detections")
                        
                        // Process each potential detection
                        for (i in 0 until numDetections) {
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
                                    
                                    // Check if this detection meets our confidence threshold
                                    if (bestScore > confidenceThreshold) {
                                        // Normalize coordinates to 0-1 range
                                        val normalizedX = xCenter / MODEL_WIDTH
                                        val normalizedY = yCenter / MODEL_HEIGHT
                                        val normalizedWidth = width / MODEL_WIDTH
                                        val normalizedHeight = height / MODEL_HEIGHT
                                        
                                        // Get the class name for this detection
                                        val className = classNames[bestClassIdx] ?: "unknown"
                                        
                                        // Add detection with normalized coordinates
                                        detections.add(Detection(
                                            classId = bestClassIdx,
                                            className = className,
                                            confidence = bestScore,
                                            boundingBox = BoundingBox(
                                                x = normalizedX,
                                                y = normalizedY,
                                                width = normalizedWidth,
                                                height = normalizedHeight
                                            ),
                                            region = region
                                        ))
                                    }
                                }
                            } catch (e: Exception) {
                                Log.e(TAG, "Error processing detection $i: ${e.message}")
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing model output: ${e.message}")
                e.printStackTrace()
            }
            
            // Apply NMS to remove overlapping detections
            val finalDetections = applyNMS(detections, 0.4f)
            Log.d(TAG, "Found ${finalDetections.size} detections in ${region.displayName} after NMS")
            
            // Clean up
            inputTensor.close()
            output.close()
            
            return Pair(finalDetections, inferenceTimeMs)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during inference: ${e.message}")
            e.printStackTrace()
            return Pair(emptyList(), inferenceTimeMs)
        }
    }
    
    /**
     * Prepare input buffer for ONNX model
     */
    private fun prepareInputBuffer(bitmap: Bitmap): FloatBuffer {
        // Allocate a float buffer for the input tensor (NCHW format)
        val bufferSize = MODEL_CHANNELS * MODEL_HEIGHT * MODEL_WIDTH
        val floatBuffer = FloatBuffer.allocate(bufferSize)
        
        // Extract pixel values
        val pixels = IntArray(MODEL_WIDTH * MODEL_HEIGHT)
        bitmap.getPixels(pixels, 0, MODEL_WIDTH, 0, 0, MODEL_WIDTH, MODEL_HEIGHT)
        
        // Prepare RGB arrays
        val r = FloatArray(MODEL_HEIGHT * MODEL_WIDTH)
        val g = FloatArray(MODEL_HEIGHT * MODEL_WIDTH)
        val b = FloatArray(MODEL_HEIGHT * MODEL_WIDTH)
        
        var pixelIndex = 0
        for (y in 0 until MODEL_HEIGHT) {
            for (x in 0 until MODEL_WIDTH) {
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
        for (i in 0 until MODEL_HEIGHT * MODEL_WIDTH) {
            floatBuffer.put(r[i])
        }
        for (i in 0 until MODEL_HEIGHT * MODEL_WIDTH) {
            floatBuffer.put(g[i])
        }
        for (i in 0 until MODEL_HEIGHT * MODEL_WIDTH) {
            floatBuffer.put(b[i])
        }
        
        floatBuffer.rewind()
        return floatBuffer
    }
    
    /**
     * Apply Non-Maximum Suppression to remove overlapping detections
     */
    private fun applyNMS(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        if (detections.isEmpty()) return emptyList()
        
        // Group detections by class
        val detectionsByClass = detections.groupBy { it.className }
        val selected = mutableListOf<Detection>()
        
        // Process each class separately
        detectionsByClass.forEach { (className, classDetections) ->
            // Sort detections by confidence
            val sortedDetections = classDetections.sortedByDescending { it.confidence }
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
        }
        
        return selected
    }
    
    /**
     * Calculate Intersection over Union for two bounding boxes
     */
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
    
    /**
     * Calculate severity level from GAGS score
     */
    private fun calculateSeverityFromGAGS(gagsScore: Int): String {
        return when {
            gagsScore == 0 -> "No Acne"
            gagsScore <= 18 -> "Mild"
            gagsScore <= 30 -> "Moderate"
            gagsScore <= 36 -> "Severe"
            else -> "Very Severe"
        }
    }
    
    /**
     * Close resources when done
     */
    fun close() {
        try {
            ortSession?.close()
            ortEnvironment?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing ONNX resources: ${e.message}")
        }
    }
}