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

    // Camera preview dimensions - will be updated dynamically
    private var previewWidth = 0
    private var previewHeight = 0

    // Coordinate system calibration options
    private var flipYCoordinates = false
    private var yOffsetCorrection = 0.0f 
    private var xOffsetCorrection = 0.0f
    private var boundingBoxScaleFactor = 1.0f

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

    // Class to hold all transformation parameters
    data class TransformationParams(
        val originalWidth: Int,
        val originalHeight: Int,
        val rotationDegrees: Int,
        val scaleFactor: Float,
        val paddingX: Float,
        val paddingY: Float,
        val modelWidth: Int,
        val modelHeight: Int,
        val previewWidth: Int,
        val previewHeight: Int
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

    // Update preview dimensions to enable accurate coordinate transformation
    fun updatePreviewDimensions(width: Int, height: Int, deviceOrientation: Int = 0) {
        this.previewWidth = width
        this.previewHeight = height
        
        // Also store device orientation if provided
        Log.d(TAG, "Updated preview dimensions: $previewWidth x $previewHeight, device orientation: $deviceOrientation°")
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

        // If we don't have preview dimensions yet, try to get them from the image proxy
        if (previewWidth == 0 || previewHeight == 0) {
            previewWidth = imageProxy.width
            previewHeight = imageProxy.height
            Log.d(TAG, "Setting preview dimensions from image proxy: $previewWidth x $previewHeight")
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
            
            // Account for potential camera rotation/orientation
            // This is important as the camera preview may be in different orientation than what the model expects
            // Different devices and configurations may require different adjustments
            val cameraOrientationAdjustment = determineCameraOrientation(imageProxy)
            Log.d(TAG, "Camera orientation adjustment: $cameraOrientationAdjustment")
            
            // Run ONNX inference
            val result = runInference(bitmap, cameraOrientationAdjustment)
            
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
    
    private fun runInference(bitmap: Bitmap, cameraOrientation: Int = 0): AnalysisResult {
        val ortEnv = ortEnvironment ?: throw IllegalStateException("ORT environment is null")
        val ortSession = ortSession ?: throw IllegalStateException("ORT session is null")
        
        try {
            // Track all transformations for accurate coordinate mapping
            var sourceWidth = bitmap.width
            var sourceHeight = bitmap.height
            
            // Record original dimensions for mapping coordinates back
            val originalWidth = bitmap.width
            val originalHeight = bitmap.height
            Log.d(TAG, "Original image dimensions: $originalWidth x $originalHeight")
            
            // Apply rotation to bitmap if needed based on camera orientation
            var rotatedBitmap = bitmap
            var rotationApplied = false
            
            if (cameraOrientation > 0) {
                Log.d(TAG, "Rotating input image by $cameraOrientation degrees to match model expectations")
                val matrix = android.graphics.Matrix()
                matrix.postRotate(cameraOrientation.toFloat())
                rotatedBitmap = Bitmap.createBitmap(
                    bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
                )
                sourceWidth = rotatedBitmap.width
                sourceHeight = rotatedBitmap.height
                rotationApplied = true
                Log.d(TAG, "Image rotated from ${bitmap.width}x${bitmap.height} to ${sourceWidth}x${sourceHeight}")
            }
            
            // Calculate scale factors and padding for letterboxing/pillarboxing
            // This is critical for accurate coordinate translation
            val scaleFactorWidth = MODEL_WIDTH.toFloat() / sourceWidth
            val scaleFactorHeight = MODEL_HEIGHT.toFloat() / sourceHeight
            
            // Choose the scaling method based on aspect ratio
            val scaleFactor = minOf(scaleFactorWidth, scaleFactorHeight)
            
            // Calculate padding to center the image
            val paddingX = if (scaleFactorWidth > scaleFactorHeight) {
                (MODEL_WIDTH - (sourceWidth * scaleFactor)) / 2
            } else {
                0f
            }
            
            val paddingY = if (scaleFactorHeight > scaleFactorWidth) {
                (MODEL_HEIGHT - (sourceHeight * scaleFactor)) / 2
            } else {
                0f
            }
            
            Log.d(TAG, "Letterboxing: scale=$scaleFactor, paddingX=$paddingX, paddingY=$paddingY")
            
            // Store transformation parameters for later use in coordinate mapping
            val transformParams = TransformationParams(
                originalWidth = originalWidth,
                originalHeight = originalHeight,
                rotationDegrees = cameraOrientation,
                scaleFactor = scaleFactor,
                paddingX = paddingX,
                paddingY = paddingY,
                modelWidth = MODEL_WIDTH,
                modelHeight = MODEL_HEIGHT,
                previewWidth = previewWidth,
                previewHeight = previewHeight
            )
            
            // Resize bitmap to expected input size
            val resizedBitmap = if (rotatedBitmap.width != inputWidth || rotatedBitmap.height != inputHeight) {
                Log.d(TAG, "Resizing bitmap from ${rotatedBitmap.width}x${rotatedBitmap.height} to ${inputWidth}x${inputHeight}")
                Bitmap.createScaledBitmap(rotatedBitmap, inputWidth, inputHeight, true)
            } else {
                rotatedBitmap
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
            val result = processResults(output, transformParams)
            
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
    
    private fun processResults(output: OrtSession.Result, transformParams: TransformationParams): AnalysisResult {
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
                                    // Get raw model pixel coordinates
                                    val modelX = xCenter
                                    val modelY = yCenter
                                    val modelWidth = width
                                    val modelHeight = height
                                    
                                    // Log the raw model coordinates for debugging
                                    Log.d(TAG, "Raw model detection: x=$modelX, y=$modelY, w=$modelWidth, h=$modelHeight")
                                    
                                    // Directly map from model space to preview space using our transformation function
                                    val mappedBox = mapModelToPreviewCoordinates(
                                        modelX,
                                        modelY,
                                        modelWidth,
                                        modelHeight,
                                        transformParams
                                    )
                                    
                                    // Get the class name for this detection
                                    val className = classNames[bestClassIdx] ?: "unknown"
                                    
                                    // Log the mapped coordinates
                                    Log.d(TAG, "Mapped to preview coordinates: class=$className, x=${mappedBox.x}, y=${mappedBox.y}, w=${mappedBox.width}, h=${mappedBox.height}")
                                    
                                    // Apply class-specific size adjustments for better visual representation
                                    val adjustedBox = adjustBoundingBoxByClass(mappedBox, className)
                                    
                                    // Add detection with properly mapped coordinates
                                    detectionsList.add(Detection(
                                        classId = bestClassIdx,
                                        className = className,
                                        confidence = bestScore,
                                        boundingBox = adjustedBox
                                    ))
                                    
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
        
        // Group detections by class
        val detectionsByClass = detections.groupBy { it.className }
        val selected = mutableListOf<Detection>()
        
        // Process each class separately with class-specific IoU thresholds
        detectionsByClass.forEach { (className, classDetections) ->
            // Use different IoU thresholds for different acne types
            val classIouThreshold = when (className) {
                "comedone" -> 0.3f  // Comedones are smaller, use higher threshold to preserve more of them
                "pustule" -> 0.25f  // Medium threshold for pustules
                "papule" -> 0.2f    // Standard threshold for papules
                "nodule" -> 0.4f    // Nodules are larger, use higher threshold to avoid duplicates
                else -> iouThreshold // Default fallback
            }
            
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
                    
                    // Log IoU calculations for debugging
                    if (sortedDetections[i].confidence > 0.5f && sortedDetections[j].confidence > 0.5f) {
                        Log.d(TAG, "NMS: IoU between detection $i (conf=${sortedDetections[i].confidence.format(2)}) " +
                                "and $j (conf=${sortedDetections[j].confidence.format(2)}) is ${iou.format(2)}")
                    }
                    
                    if (iou > classIouThreshold) {
                        processed[j] = true
                    }
                }
            }
        }
        
        Log.d(TAG, "NMS: Reduced from ${detections.size} to ${selected.size} detections")
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

    // Add helper method to determine camera orientation relative to the device
    private fun determineCameraOrientation(imageProxy: ImageProxy): Int {
        // Get the rotation value from the image proxy
        val rotation = imageProxy.imageInfo.rotationDegrees
        
        // Log both the image properties and device orientation
        Log.d(TAG, "Camera image rotation: $rotation degrees")
        Log.d(TAG, "Image dimensions: ${imageProxy.width}x${imageProxy.height}")
        
        // If the preview dimensions don't match the image proxy dimensions,
        // that's a clue that there may be a rotation mismatch
        if (previewWidth > 0 && previewHeight > 0) {
            val previewIsLandscape = previewWidth > previewHeight
            val imageIsLandscape = imageProxy.width > imageProxy.height
            
            if (previewIsLandscape != imageIsLandscape) {
                Log.d(TAG, "Preview orientation (${if (previewIsLandscape) "landscape" else "portrait"}) " +
                      "doesn't match image orientation (${if (imageIsLandscape) "landscape" else "portrait"})")
            }
        }
        
        // Return the rotation for coordinate mapping
        return rotation
    }

    // Function to map model coordinates to preview coordinates
    private fun mapModelToPreviewCoordinates(
        x: Float, 
        y: Float, 
        width: Float, 
        height: Float, 
        params: TransformationParams
    ): BoundingBox {
        // First convert from model pixel space to the padded image space
        // x, y, width, height are in model pixel coordinates (0-640)
        
        // 1. Remove padding applied during letterboxing/pillarboxing
        val unpaddedX = (x - params.paddingX) / params.scaleFactor
        val unpaddedY = (y - params.paddingY) / params.scaleFactor
        val unpaddedWidth = width / params.scaleFactor
        val unpaddedHeight = height / params.scaleFactor
        
        // 2. Account for rotation if needed
        var finalX = unpaddedX
        var finalY = unpaddedY
        
        if (params.rotationDegrees > 0) {
            // Coordinate transformation for rotation
            // For 90-degree rotation
            if (params.rotationDegrees == 90) {
                finalX = params.originalHeight - unpaddedY
                finalY = unpaddedX
                Log.d(TAG, "Applied 90° rotation: ($unpaddedX,$unpaddedY) -> ($finalX,$finalY)")
            } 
            // For 180-degree rotation
            else if (params.rotationDegrees == 180) {
                finalX = params.originalWidth - unpaddedX
                finalY = params.originalHeight - unpaddedY
                Log.d(TAG, "Applied 180° rotation: ($unpaddedX,$unpaddedY) -> ($finalX,$finalY)")
            }
            // For 270-degree rotation
            else if (params.rotationDegrees == 270) {
                finalX = unpaddedY
                finalY = params.originalWidth - unpaddedX
                Log.d(TAG, "Applied 270° rotation: ($unpaddedX,$unpaddedY) -> ($finalX,$finalY)")
            }
        }
        
        // 3. Map from original image coordinates to preview coordinates
        // Normalize to 0-1 range relative to the original image
        val normalizedX = finalX / params.originalWidth
        val normalizedY = finalY / params.originalHeight
        val normalizedWidth = unpaddedWidth / params.originalWidth
        val normalizedHeight = unpaddedHeight / params.originalHeight
        
        // Log transformation details for debugging
        Log.d(TAG, "Coordinate mapping: model($x,$y,$width,$height) -> unpadded($unpaddedX,$unpaddedY,$unpaddedWidth,$unpaddedHeight) -> final($finalX,$finalY) -> normalized($normalizedX,$normalizedY,$normalizedWidth,$normalizedHeight)")
        Log.d(TAG, "Transform params: rotation=${params.rotationDegrees}°, scale=${params.scaleFactor}, padding=[${params.paddingX},${params.paddingY}], preview=${params.previewWidth}x${params.previewHeight}")
        
        return BoundingBox(normalizedX, normalizedY, normalizedWidth, normalizedHeight)
    }

    // Apply class-specific adjustments to bounding box sizes without changing their positions
    private fun adjustBoundingBoxByClass(box: BoundingBox, className: String): BoundingBox {
        // Size adjustment factors for each class
        val sizeAdjustment = when (className) {
            "comedone" -> 0.9f  // Smaller - comedones are small and precise
            "pustule" -> 1.1f   // Slightly larger for pustules
            "papule" -> 1.2f    // Larger for papules
            "nodule" -> 1.3f    // Much larger for nodules
            else -> 1.0f
        }
        
        // Apply size adjustment without changing center position
        val newWidth = (box.width * sizeAdjustment).coerceIn(0.01f, 1.0f)
        val newHeight = (box.height * sizeAdjustment).coerceIn(0.01f, 1.0f)
        
        // Make sure the box stays within bounds
        var newX = box.x
        var newY = box.y
        
        // Adjust X if needed
        if (newX - newWidth/2 < 0) {
            newX = newWidth/2
        } else if (newX + newWidth/2 > 1) {
            newX = 1 - newWidth/2
        }
        
        // Adjust Y if needed
        if (newY - newHeight/2 < 0) {
            newY = newHeight/2
        } else if (newY + newHeight/2 > 1) {
            newY = 1 - newHeight/2
        }
        
        Log.d(TAG, "Adjusted $className box: ${box.width.format(2)}x${box.height.format(2)} -> ${newWidth.format(2)}x${newHeight.format(2)}")
        
        return BoundingBox(
            x = newX,
            y = newY,
            width = newWidth,
            height = newHeight
        )
    }
}