package com.example.pytorchimplement

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Image analyzer for real-time TFLite model inference
 */
class ImageAnalyzer(private val context: Context, private val listener: AnalysisListener) : ImageAnalysis.Analyzer {

    // Define TAG constant at class level
    private val TAG = "ImageAnalyzer"

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
    
    // Min time between analyses in ms (reduced from 500ms to 250ms for more frequent updates)
    private val analysisCooldown = 250L
    private var lastAnalysisTime = 0L
    
    // TFLite interpreter
    private var interpreter: Interpreter? = null
    
    // Model config
    private val modelFileName = "model_fp16.tflite" // Update with your actual model filename
    
    // Image processor for preprocessing
    private lateinit var imageProcessor: ImageProcessor
    
    init {
        loadModel()
        setupImageProcessor()
    }
    
    private fun setupImageProcessor() {
        try {
            // Get input dimensions dynamically from the loaded model
            val inputShape = interpreter?.getInputTensor(0)?.shape() ?: intArrayOf(1, 224, 224, 3)
            
            // Log the raw input shape first
            Log.d(TAG, "Raw model input shape: ${inputShape.contentToString()}")
            
            // IMPORTANT: This is a PyTorch model with NCHW format (batch, channels, height, width)
            // The shape is [1, 3, 640, 640] where:
            // - inputShape[0] = batch size (1)
            // - inputShape[1] = channels (3)
            // - inputShape[2] = height (640)
            // - inputShape[3] = width (640)
            val inputBatchSize = inputShape[0]
            val inputChannels = inputShape[1] // Channels is in position 1 for PyTorch models
            val inputHeight = inputShape[2]   // Height is in position 2
            val inputWidth = inputShape[3]    // Width is in position 3
            
            // Based on the error, this model expects 640x640 input (4,915,200 bytes = 1x3x640x640x4)
            Log.d(TAG, "Corrected model input dimensions (channels-first format): C=$inputChannels, H=$inputHeight, W=$inputWidth")
            
            // Calculate total expected bytes
            val totalBytes = inputBatchSize * inputChannels * inputHeight * inputWidth * 4 // batch * channels * height * width * bytes_per_float
            Log.d(TAG, "Expected input buffer size: $totalBytes bytes")
            
            // Setup image processor with the correct dimensions
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f)) // Normalize to [0,1]
                .build()
                
        } catch (e: Exception) {
            Log.e(TAG, "Error setting up image processor: ${e.message}")
            e.printStackTrace()
            
            // For this model, based on the error message, we need 640x640 input
            // 4,915,200 ÷ 4 (bytes per float) ÷ 3 (RGB channels) = 409,600 pixels
            // √409,600 = 640, so we need a 640x640 input
            Log.d(TAG, "Using fixed dimensions for this model: 640x640")
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(640, 640, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .build()
        }
    }
    
    private fun loadModel() {
        try {
            // First attempt to load using FileUtil for regular assets
            try {
                val modelFile = FileUtil.loadMappedFile(context, modelFileName)
                val options = Interpreter.Options()
                options.setNumThreads(4)
                interpreter = Interpreter(modelFile, options)
                Log.d(TAG, "Model loaded successfully using FileUtil")
            } catch (e: Exception) {
                // If that fails, try direct mapping for better performance
                Log.d(TAG, "Trying alternate model loading method: ${e.message}")
                val fileDescriptor = context.assets.openFd(modelFileName)
                val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
                val fileChannel = inputStream.channel
                val startOffset = fileDescriptor.startOffset
                val declaredLength = fileDescriptor.declaredLength
                val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

                val options = Interpreter.Options()
                options.setNumThreads(4)
                interpreter = Interpreter(modelBuffer, options)
                Log.d(TAG, "Model loaded successfully using direct mapping")
            }
            
            // Log model input/output details for debugging
            logModelDetails()
            
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model: ${e.message}")
            e.printStackTrace()
        }
    }
    
    private fun logModelDetails() {
        interpreter?.let { interp ->
            try {
                // Log input tensor details
                val inputTensor = interp.getInputTensor(0)
                val inputShape = inputTensor.shape()
                Log.d(TAG, "Model input shape: ${inputShape.contentToString()}")
                Log.d(TAG, "Model input data type: ${inputTensor.dataType()}")
                
                // Log output tensor details
                val numOutputs = interp.getOutputTensorCount()
                Log.d(TAG, "Number of output tensors: $numOutputs")
                
                for (i in 0 until numOutputs) {
                    val outputTensor = interp.getOutputTensor(i)
                    Log.d(TAG, "Output tensor $i shape: ${outputTensor.shape().contentToString()}")
                    Log.d(TAG, "Output tensor $i data type: ${outputTensor.dataType()}")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error logging model details: ${e.message}")
            }
        }
    }

    override fun analyze(imageProxy: ImageProxy) {
        // Skip if already analyzing, interpreter not loaded, or in cooldown period
        val currentTime = System.currentTimeMillis()
        if (isAnalyzing || interpreter == null) {
            Log.d(TAG, "Skipping frame: isAnalyzing=$isAnalyzing, interpreter=${interpreter != null}")
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
            
            // Run TFLite inference
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
        val interpreter = this.interpreter ?: throw IllegalStateException("TFLite interpreter is null")
        
        try {
            // Get model input details dynamically
            val inputTensor = interpreter.getInputTensor(0)
            val inputShape = inputTensor.shape()
            
            Log.d(TAG, "Model expects input shape: ${inputShape.contentToString()}")
            Log.d(TAG, "Original bitmap dimensions: ${bitmap.width}x${bitmap.height}")
            
            // Calculate required buffer size
            val requiredBytes = inputShape.fold(1) { acc, dim -> acc * dim } * 4 // 4 bytes per float
            Log.d(TAG, "Required buffer size: $requiredBytes bytes")
            
            // IMPORTANT: Correctly interpret the input dimensions for PyTorch model (NCHW format)
            // For PyTorch models, the format is [batch, channels, height, width]
            val inputBatchSize = inputShape[0]
            val inputChannels = inputShape[1] // Channels is position 1 in PyTorch models
            val inputHeight = inputShape[2]   // Height is position 2
            val inputWidth = inputShape[3]    // Width is position 3
            
            Log.d(TAG, "Model expects dimensions: $inputBatchSize x $inputChannels x $inputHeight x $inputWidth (NCHW format)")
            
            // Double-check that our dimensions make sense for a PyTorch model
            if (inputChannels != 3 || inputHeight < 10 || inputWidth < 10) {
                // Something's wrong - the dimensions don't look right
                Log.e(TAG, "Invalid dimensions for PyTorch model: ${inputChannels}x${inputHeight}x${inputWidth}. Using fixed 640x640 instead.")
                return runInferenceWithByteBuffer(bitmap, 640, 640, 3)
            }
            
            // If the bitmap is not the correct size, resize it manually as a fallback
            val resizedBitmap = if (bitmap.width != inputWidth || bitmap.height != inputHeight) {
                Log.d(TAG, "Manually resizing bitmap to ${inputWidth}x${inputHeight}")
                Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
            } else {
                bitmap
            }
            
            // For PyTorch models, we need to manually convert the bitmap to a properly formatted ByteBuffer
            // because TensorFlow's ImageProcessor expects NHWC format but our model needs NCHW
            Log.d(TAG, "Using direct ByteBuffer method for PyTorch model with NCHW format")
            return runInferenceWithByteBuffer(resizedBitmap, inputWidth, inputHeight, inputChannels)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during inference: ${e.message}")
            e.printStackTrace()
            
            // Return empty result on error
            return AnalysisResult(
                severity = 0,
                timestamp = System.currentTimeMillis()
            )
        }
    }
    
    // Fallback method using direct ByteBuffer for PyTorch models with NCHW format
    private fun runInferenceWithByteBuffer(bitmap: Bitmap, width: Int, height: Int, channels: Int): AnalysisResult {
        try {
            Log.d(TAG, "Using direct ByteBuffer method for PyTorch model with dimensions: ${width}x${height}x${channels}")
            val interpreter = this.interpreter ?: throw IllegalStateException("TFLite interpreter is null")
            
            // Ensure bitmap is the right size
            val scaledBitmap = if (bitmap.width != width || bitmap.height != height) {
                Log.d(TAG, "Resizing bitmap for PyTorch model from ${bitmap.width}x${bitmap.height} to ${width}x${height}")
                Bitmap.createScaledBitmap(bitmap, width, height, true)
            } else {
                bitmap
            }
            
            // Create a direct ByteBuffer with the correct size (4 bytes per float)
            val bufferSize = 1 * channels * height * width * 4 // batch=1, channels, height, width, float32=4 bytes
            Log.d(TAG, "Allocating direct ByteBuffer of size: $bufferSize bytes for PyTorch model")
            
            val byteBuffer = ByteBuffer.allocateDirect(bufferSize)
            byteBuffer.order(ByteOrder.nativeOrder())
            
            // Fill buffer with bitmap pixel data
            val pixels = IntArray(width * height)
            scaledBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
            
            // Log some sample pixels for debugging
            if (pixels.isNotEmpty()) {
                Log.d(TAG, "First 5 pixels: ${pixels.take(5).joinToString()}")
            }
            
            // IMPORTANT: For PyTorch models, we need to use NCHW format (channels first)
            // We'll organize the data as [batch][channel][height][width]
            // Extract all pixel values first
            val r = FloatArray(height * width)
            val g = FloatArray(height * width)
            val b = FloatArray(height * width)
            
            var pixelIndex = 0
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val pixel = pixels[pixelIndex]
                    
                    // Extract and normalize RGB values to 0-1
                    r[pixelIndex] = (pixel shr 16 and 0xFF) / 255.0f
                    g[pixelIndex] = (pixel shr 8 and 0xFF) / 255.0f
                    b[pixelIndex] = (pixel and 0xFF) / 255.0f
                    
                    pixelIndex++
                }
            }
            
            // Now add them to the buffer in channel-first order (R values, then G values, then B values)
            // First all R values
            for (i in 0 until height * width) {
                byteBuffer.putFloat(r[i])
            }
            
            // Then all G values
            for (i in 0 until height * width) {
                byteBuffer.putFloat(g[i])
            }
            
            // Then all B values
            for (i in 0 until height * width) {
                byteBuffer.putFloat(b[i])
            }
            
            // Reset position to start
            byteBuffer.rewind()
            
            // Log buffer size confirmation
            Log.d(TAG, "PyTorch-format ByteBuffer prepared with capacity: ${byteBuffer.capacity()} bytes, limit: ${byteBuffer.limit()}")
            
            // Create outputs
            val outputsCount = interpreter.getOutputTensorCount()
            val outputMap = HashMap<Int, Any>()
            
            // Create properly sized output tensors
            for (i in 0 until outputsCount) {
                val outputTensor = interpreter.getOutputTensor(i)
                val outputShape = outputTensor.shape()
                Log.d(TAG, "Output tensor $i shape: ${outputShape.contentToString()}")
                val outputBuffer = createOutputBuffer(outputShape)
                outputMap[i] = outputBuffer
            }
            
            // Run inference
            Log.d(TAG, "Running inference with PyTorch-format ByteBuffer input")
            interpreter.runForMultipleInputsOutputs(arrayOf(byteBuffer), outputMap)
            
            Log.d(TAG, "PyTorch-format inference completed successfully")
            return processResults(outputMap)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during PyTorch inference: ${e.message}")
            e.printStackTrace()
            
            return AnalysisResult(
                severity = 0,
                timestamp = System.currentTimeMillis()
            )
        }
    }
    
    private fun createOutputBuffer(shape: IntArray): Any {
        return when (shape.size) {
            1 -> FloatArray(shape[0])
            2 -> Array(shape[0]) { FloatArray(shape[1]) }
            3 -> Array(shape[0]) { Array(shape[1]) { FloatArray(shape[2]) } }
            4 -> Array(shape[0]) {
                Array(shape[1]) { Array(shape[2]) { FloatArray(shape[3]) } }
            }
            else -> {
                Log.w(
                    TAG,
                    "Unsupported output tensor dimension: ${shape.size}, defaulting to ByteBuffer"
                )
                val bufferSize = shape.fold(1) { acc, dim -> acc * dim } * 4 // 4 bytes per float
                ByteBuffer.allocateDirect(bufferSize).apply { order(ByteOrder.nativeOrder()) }
            }
        }
    }
    
    private fun processResults(outputMap: Map<Int, Any>): AnalysisResult {
        // Log output for debugging
        outputMap.forEach { (index, output) ->
            Log.d(TAG, "Output $index: ${describeOutputArray(output)}")
        }
        
        // Initialize acne counts
        val acneCounts = mutableMapOf(
            "comedone" to 0,
            "pustule" to 0,
            "papule" to 0,
            "nodule" to 0
        )
        
        // Create list to hold detection results with bounding boxes
        val detections = mutableListOf<Detection>()
        
        try {
            // YOLOv9 detection output should be in tensor 5 with shape [1, 8, 8400]
            // where 8400 is the number of possible detections and 8 is:
            // - 4 values for box coordinates (x, y, w, h)
            // - 4 values for class scores (our 4 acne types)

            // Define class names
            val classNames = mapOf(
                0 to "comedone",
                1 to "pustule",
                2 to "papule",
                3 to "nodule"
            )
            
            // Constants for processing - LOWER THRESHOLD to catch more potential detections
            val confidenceThreshold = 0.1f  // Reduced from 0.25f to 0.1f to detect more boxes
            
            // First, check if we have the detection output tensor (index 5)
            val detectionTensor = outputMap[5]
            val detectionTensor2 = outputMap[6]
            
            if (detectionTensor == null && detectionTensor2 == null) {
                Log.e(TAG, "No detection tensors found in output")
                return AnalysisResult(
                    severity = 0,
                    timestamp = System.currentTimeMillis(),
                    acneCounts = acneCounts,
                    detections = detections
                )
            }
            
            // Process the primary detection tensor (index 5)
            if (detectionTensor != null) {
                Log.d(TAG, "Processing YOLOv9 primary detection output tensor (index 5)")
                
                when (detectionTensor) {
                    is Array<*> -> {
                        processYoloArrayOutput(detectionTensor, classNames, confidenceThreshold, acneCounts, detections)
                    }
                    is ByteBuffer -> {
                        processYoloByteBufferOutput(detectionTensor, classNames, confidenceThreshold, acneCounts, detections)
                    }
                    else -> {
                        Log.e(TAG, "Unsupported detection tensor type: ${detectionTensor.javaClass.simpleName}")
                    }
                }
            }
            
            // Process the secondary detection tensor (index 6) if needed
            if (detectionTensor2 != null && detections.isEmpty()) {
                Log.d(TAG, "Processing YOLOv9 secondary detection output tensor (index 6)")
                
                when (detectionTensor2) {
                    is Array<*> -> {
                        processYoloArrayOutput(detectionTensor2, classNames, confidenceThreshold, acneCounts, detections)
                    }
                    is ByteBuffer -> {
                        processYoloByteBufferOutput(detectionTensor2, classNames, confidenceThreshold, acneCounts, detections)
                    }
                    else -> {
                        Log.e(TAG, "Unsupported detection tensor type: ${detectionTensor2.javaClass.simpleName}")
                    }
                }
            }
            
            // Calculate total count
            val totalAcneCount = acneCounts.values.sum()
            
            // If we found detections but counts are zero, update counts
            if (detections.isNotEmpty() && totalAcneCount == 0) {
                // Update acne counts based on detections
                detections.forEach { detection ->
                    val className = classNames[detection.classId] ?: return@forEach
                    acneCounts[className] = acneCounts[className]!! + 1
                }
            }
            
            // Calculate severity score based on acne counts and types
            val totalSeverityScore = calculateSeverityScore(acneCounts)
            
            // Normalize severity to 0-1 range
            val severity = if (totalAcneCount > 0) {
                (totalSeverityScore / totalAcneCount).coerceIn(0f, 1f).toInt()
            } else {
                0
            }
            
            Log.d(TAG, "Analysis complete - Severity: $severity, Counts: $acneCounts, Total: ${acneCounts.values.sum()}, Detections: ${detections.size}")
            
            return AnalysisResult(
                severity = severity,
                timestamp = System.currentTimeMillis(),
                acneCounts = acneCounts,
                detections = detections
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing results: ${e.message}")
            e.printStackTrace()
            
            // Return minimal result on error
            return AnalysisResult(
                severity = 0,
                timestamp = System.currentTimeMillis(),
                acneCounts = acneCounts,
                detections = detections
            )
        }
    }
    
    // Process YOLOv9 detection output tensor as Array
    fun processYoloArrayOutput(
        output: Array<*>,
        classNames: Map<Int, String>,
        confidenceThreshold: Float,
        acneCounts: MutableMap<String, Int>,
        detections: MutableList<Detection>
    ) {
        try {
            Log.d(TAG, "Processing YOLOv9 Array output")
            
            // For YOLOv9, the array shape should be [1][8][8400]
            if (output.size != 1 || output[0] !is Array<*>) {
                Log.e(TAG, "Invalid YOLOv9 output array format")
                return
            }
            
            val batch = output[0] as Array<*>
            
            // Each row represents one component (x, y, width, height, class scores...)
            if (batch.size < 8) {
                Log.e(TAG, "YOLOv9 output has insufficient components: ${batch.size}")
                return
            }
            
            // Get the first row (x coordinates) to determine number of detections
            val xCoords = batch[0] as? Array<*> ?: return
            val numDetections = xCoords.size
            
            Log.d(TAG, "Found $numDetections potential detections in array output")
            
            // Add debug logging for first few detections regardless of confidence
            for (i in 0 until minOf(5, numDetections)) {
                try {
                    // Extract coordinates
                    val x = (batch[0] as? Array<*>)?.get(i) as? Float ?: continue
                    val y = (batch[1] as? Array<*>)?.get(i) as? Float ?: continue
                    val width = (batch[2] as? Array<*>)?.get(i) as? Float ?: continue
                    val height = (batch[3] as? Array<*>)?.get(i) as? Float ?: continue
                    
                    // Log class probabilities for debugging
                    val classProbs = (0 until 4).map { c ->
                        val prob = (batch[4 + c] as? Array<*>)?.get(i) as? Float ?: 0f
                        "$c:$prob"
                    }.joinToString(", ")
                    
                    Log.d(TAG, "DEBUG Raw Detection #$i: Box=[x=$x, y=$y, w=$width, h=$height], Classes=[$classProbs]")
                } catch (e: Exception) {
                    Log.e(TAG, "Error logging raw detection $i: ${e.message}")
                }
            }
            
            // Extract all box coordinates and class scores
            val processedDetections = mutableListOf<Pair<Int, Detection>>()
            
            for (i in 0 until numDetections) {
                try {
                    // Extract coordinates
                    val x = (batch[0] as? Array<*>)?.get(i) as? Float ?: continue
                    val y = (batch[1] as? Array<*>)?.get(i) as? Float ?: continue
                    val width = (batch[2] as? Array<*>)?.get(i) as? Float ?: continue
                    val height = (batch[3] as? Array<*>)?.get(i) as? Float ?: continue
                    
                    // Find best class score
                    var maxClassProb = 0f
                    var bestClassId = -1
                    
                    for (c in 0 until 4) { // 4 acne classes
                        val classProb = (batch[4 + c] as? Array<*>)?.get(i) as? Float ?: 0f
                        if (classProb > maxClassProb) {
                            maxClassProb = classProb
                            bestClassId = c
                        }
                    }
                    
                    // If confidence is above threshold, add to detections
                    if (maxClassProb > confidenceThreshold && bestClassId >= 0) {
                        // Create detection object
                        val className = classNames[bestClassId] ?: "unknown"
                        val detection = Detection(
                            classId = bestClassId,
                            className = className,
                            confidence = maxClassProb,
                            boundingBox = BoundingBox(
                                x = x,
                                y = y,
                                width = width,
                                height = height
                            )
                        )
                        
                        processedDetections.add(Pair(i, detection))
                        
                        if (processedDetections.size < 10) {
                            Log.d(TAG, "YOLOv9 Array Detection #$i: Class=$bestClassId ($className), " +
                                  "Confidence=${maxClassProb}, Box=[x=$x, y=$y, w=$width, h=$height]")
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error processing array detection $i: ${e.message}")
                }
                
                // Early exit if we found enough detections
                if (processedDetections.size >= 100) {
                    Log.d(TAG, "Reached maximum number of detections (100), stopping early")
                    break
                }
            }
            
            // Apply non-maximum suppression
            val selectedDetections = nonMaxSuppression(processedDetections.map { it.second }, 0.5f)
            
            // Update detections list and acne counts
            selectedDetections.forEach { detection ->
                detections.add(detection)
                val className = detection.className
                acneCounts[className] = acneCounts[className]!! + 1
            }
            
            Log.d(TAG, "YOLOv9 Array output: Found ${processedDetections.size} raw detections, " +
                  "selected ${selectedDetections.size} after NMS")
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing YOLOv9 Array output: ${e.message}")
            e.printStackTrace()
        }
    }
    
    // Process YOLOv9 detection output tensor as ByteBuffer
    fun processYoloByteBufferOutput(
        buffer: ByteBuffer,
        classNames: Map<Int, String>,
        confidenceThreshold: Float,
        acneCounts: MutableMap<String, Int>,
        detections: MutableList<Detection>
    ) {
        try {
            // Reset buffer position
            buffer.rewind()
            
            // For YOLOv9, output tensor is [1, 8, 8400]
            // Each detection has 8 values (4 box coordinates + 4 class scores)
            val numDetections = 8400
            val numClasses = 4
            val numValues = 8 // 4 box + 4 class
            
            Log.d(TAG, "Processing YOLOv9 ByteBuffer output with $numDetections possible detections")
            
            // Create a temporary buffer to read all values at once for better debugging
            val allValues = FloatArray(20) // Read more values for better debugging
            val bufferCopy = buffer.duplicate()
            bufferCopy.rewind()
            for (i in 0 until minOf(20, bufferCopy.capacity() / 4)) {
                if (bufferCopy.remaining() >= 4) {
                    allValues[i] = bufferCopy.float
                }
            }
            
            Log.d(TAG, "First 20 raw values from buffer: ${allValues.joinToString()}")
            
            // Reset buffer position
            buffer.rewind()
            
            // For each possible detection
            val processedDetections = mutableListOf<Pair<Int, Detection>>()
            
            for (i in 0 until numDetections) {
                try {
                    // Read all values for this detection
                    // YOLOv9 stores values in a transposed format where all x coordinates come first, then all y, etc.
                    
                    // Read box coordinates
                    // We need to access the values in the right order
                    // Read x coordinate for this detection
                    buffer.position(i * 4) // i*4 because each float is 4 bytes
                    val x = buffer.float
                    
                    // Read y coordinate
                    buffer.position((numDetections + i) * 4)
                    val y = buffer.float
                    
                    // Read width
                    buffer.position((2 * numDetections + i) * 4)
                    val width = buffer.float
                    
                    // Read height
                    buffer.position((3 * numDetections + i) * 4)
                    val height = buffer.float
                    
                    // Find the best class for this detection
                    var maxClassProb = 0f
                    var bestClassId = -1
                    
                    for (c in 0 until numClasses) {
                        buffer.position(((4 + c) * numDetections + i) * 4)
                        val classProb = buffer.float
                        
                        if (classProb > maxClassProb) {
                            maxClassProb = classProb
                            bestClassId = c
                        }
                    }
                    
                    // If confidence is above threshold, add to detections
                    if (maxClassProb > confidenceThreshold && bestClassId >= 0) {
                        // Create a detection object
                        val className = classNames[bestClassId] ?: "unknown"
                        val detection = Detection(
                            classId = bestClassId,
                            className = className,
                            confidence = maxClassProb,
                            boundingBox = BoundingBox(
                                x = x,
                                y = y,
                                width = width,
                                height = height
                            )
                        )
                        
                        processedDetections.add(Pair(i, detection))
                        
                        if (processedDetections.size < 10) {
                            Log.d(TAG, "YOLOv9 Detection #$i: Class=$bestClassId ($className), " +
                                  "Confidence=${maxClassProb}, Box=[x=$x, y=$y, w=$width, h=$height]")
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error processing detection $i: ${e.message}")
                }
                
                // Early exit if we found enough detections
                if (processedDetections.size >= 100) {
                    Log.d(TAG, "Reached maximum number of detections (100), stopping early")
                    break
                }
            }
            
            // Apply non-maximum suppression
            val selectedDetections = nonMaxSuppression(processedDetections.map { it.second }, 0.5f)
            
            // Update detections list and acne counts
            selectedDetections.forEach { detection ->
                detections.add(detection)
                val className = detection.className
                acneCounts[className] = acneCounts[className]!! + 1
            }
            
            Log.d(TAG, "YOLOv9 ByteBuffer output: Found ${processedDetections.size} raw detections, " +
                  "selected ${selectedDetections.size} after NMS")
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing YOLOv9 ByteBuffer output: ${e.message}")
            e.printStackTrace()
        }
    }
    
    // Calculate Intersection over Union (IoU) for two bounding boxes
    fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        // Convert from center format (x, y, w, h) to corner format (x1, y1, x2, y2)
        val box1X1 = box1.x - box1.width / 2
        val box1Y1 = box1.y - box1.height / 2
        val box1X2 = box1.x + box1.width / 2
        val box1Y2 = box1.y + box1.height / 2
        
        val box2X1 = box2.x - box2.width / 2
        val box2Y1 = box2.y - box2.height / 2
        val box2X2 = box2.x + box2.width / 2
        val box2Y2 = box2.y + box2.height / 2
        
        // Calculate intersection area
        val xOverlap = maxOf(0f, minOf(box1X2, box2X2) - maxOf(box1X1, box2X1))
        val yOverlap = maxOf(0f, minOf(box1Y2, box2Y2) - maxOf(box1Y1, box2Y1))
        val intersectionArea = xOverlap * yOverlap
        
        // Calculate union area
        val box1Area = (box1X2 - box1X1) * (box1Y2 - box1Y1)
        val box2Area = (box2X2 - box2X1) * (box2Y2 - box2Y1)
        val unionArea = box1Area + box2Area - intersectionArea
        
        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }
    
    // Apply Non-Maximum Suppression to remove overlapping detections
    fun nonMaxSuppression(
        detections: List<Detection>,
        iouThreshold: Float
    ): List<Detection> {
        if (detections.isEmpty()) return emptyList()
        
        // Sort detections by confidence (descending)
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val selectedDetections = mutableListOf<Detection>()
        val remainingDetections = sortedDetections.toMutableList()
        
        // Process detections
        while (remainingDetections.isNotEmpty()) {
            // Select the detection with highest confidence
            val bestDetection = remainingDetections.removeAt(0)
            selectedDetections.add(bestDetection)
            
            // Remove detections of the same class that overlap significantly
            var i = 0
            while (i < remainingDetections.size) {
                val detection = remainingDetections[i]
                
                // Only compare with detections of the same class
                if (detection.classId == bestDetection.classId) {
                    val iou = calculateIoU(bestDetection.boundingBox, detection.boundingBox)
                    
                    if (iou > iouThreshold) {
                        // Remove this detection
                        remainingDetections.removeAt(i)
                    } else {
                        i++
                    }
                } else {
                    i++
                }
            }
        }
        
        return selectedDetections
    }
    
    // Calculate severity score based on acne counts and types
    fun calculateSeverityScore(acneCounts: Map<String, Int>): Float {
        var score = 0f
        
        // Different acne types contribute differently to severity
        score += acneCounts["comedone"]!! * 0.25f // Least severe
        score += acneCounts["pustule"]!! * 0.5f   // Moderately severe
        score += acneCounts["papule"]!! * 0.75f   // More severe
        score += acneCounts["nodule"]!! * 1.0f    // Most severe
        
        return score
    }
    
    // Helper functions to process different output types
    
    // These methods are no longer used with our new grid-based approach
    private fun processFloatArrayOutput(output: FloatArray, acneCounts: MutableMap<String, Int>): List<Pair<Float, Float>>? {
        return null // Implement based on your model's output format
    }
    
    private fun processArrayOutput(output: Array<*>, acneCounts: MutableMap<String, Int>): List<Pair<Float, Float>>? {
        return null // Implement based on your model's output format
    }
    
    private fun processBufferOutput(buffer: ByteBuffer, acneCounts: MutableMap<String, Int>): List<Pair<Float, Float>> {
        // Reset buffer position
        buffer.rewind()
        
        val detections = mutableListOf<Pair<Float, Float>>()
        val numDetections = 100 // Adjust based on your model
        val confidenceThreshold = 0.5f
        
        try {
            for (i in 0 until numDetections) {
                // Skip bounding box coordinates (x, y, w, h) if they exist
                buffer.position(buffer.position() + 16) // 4 floats * 4 bytes
                
                // Read confidence
                val confidence = buffer.float
                
                if (confidence > confidenceThreshold) {
                    // Read class ID
                    val classId = buffer.float
                    
                    // Add to detections
                    detections.add(Pair(classId, confidence))
                } else {
                    // Skip the rest of this detection
                    buffer.position(buffer.position() + 4) // 1 float * 4 bytes for class ID
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing buffer output: ${e.message}")
        }
        
        return detections
    }
    
    // Helper function to describe output array contents for logging
    fun describeOutputArray(output: Any?): String {
        return when (output) {
            is FloatArray -> "FloatArray size=${output.size}, sample values=[${
                if (output.isNotEmpty()) output.take(5).joinToString(", ") else "empty"
            }]"
            is Array<*> -> {
                if (output.isNotEmpty()) {
                    "Array size=${output.size} containing ${output[0]?.javaClass?.simpleName}"
                } else {
                    "Empty Array"
                }
            }
            is ByteBuffer -> {
                val bufferCopy = output.duplicate()
                bufferCopy.rewind()
                val capacity = bufferCopy.capacity()
                val sampleSize = Math.min(5, capacity / 4)
                val sample = ArrayList<Float>(sampleSize)
                
                for (i in 0 until sampleSize) {
                    if (bufferCopy.remaining() >= 4) {
                        sample.add(bufferCopy.float)
                    }
                }
                
                "ByteBuffer capacity=${capacity}, sample values=[${sample.joinToString(", ")}]"
            }
            else -> output?.javaClass?.simpleName ?: "null"
        }
    }

    // Extension function to convert ImageProxy to Bitmap
    private fun ImageProxy.toBitmap(): Bitmap {
        // Log the original image dimensions
        Log.d(TAG, "Converting ImageProxy: ${width}x${height}, format: ${format}, rotation: ${imageInfo.rotationDegrees}")
        
        val yBuffer = planes[0].buffer // Y
        val uBuffer = planes[1].buffer // U
        val vBuffer = planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        
        Log.d(TAG, "YUV buffer sizes - Y: $ySize, U: $uSize, V: $vSize")

        val nv21 = ByteArray(ySize + uSize + vSize)

        // Copy Y
        yBuffer.get(nv21, 0, ySize)
        // Copy U and V
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 90, out)
        val imageBytes = out.toByteArray()
        val bitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        
        // Log the resulting bitmap dimensions
        Log.d(TAG, "Converted to bitmap: ${bitmap.width}x${bitmap.height}, config: ${bitmap.config}")
        
        return bitmap
    }
    
    // Direct bitmap analysis method for use without camera frames
    fun analyzeBitmap(bitmap: Bitmap): AnalysisResult? {
        if (interpreter == null) {
            Log.e(TAG, "Cannot analyze bitmap: interpreter is null")
            return null
        }
        
        try {
            // Run inference directly on the provided bitmap
            val result = runInference(bitmap)
            
            // Save the result
            lastAnalysisResult = result
            
            // Only notify listener if it's not null
            if (listener != null) {
                listener.onAnalysisComplete(result)
            }
            
            return result
        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing bitmap: ${e.message}")
            e.printStackTrace()
            return null
        }
    }
    
    fun close() {
        interpreter?.close()
    }

    companion object {
        private const val TAG = "ImageAnalyzer"
    }
} 