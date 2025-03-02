package com.example.pytorchimplement

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import java.nio.ByteBuffer
import java.nio.ByteOrder
import android.content.Intent

class ModelInferenceActivity : AppCompatActivity() {
    private var interpreter: Interpreter? = null
    private lateinit var resultTextView: TextView
    
    // Model config
    private val modelInputWidth = 224
    private val modelInputHeight = 224
    private val modelFileName = "model_fp16.tflite"
    
    companion object {
        private const val TAG = "ModelInferenceActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_inference)
        
        resultTextView = findViewById(R.id.resultsTextView)

        // Load TFLite model
        loadModel()
        
        // Process images based on the source
        val acneCounts = processImagesFromIntent()
        
        // Display results
        displayResults(acneCounts)
    }
    
    private fun loadModel() {
        try {
            val modelFile = FileUtil.loadMappedFile(this, modelFileName)
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            interpreter = Interpreter(modelFile, options)
            Log.d(TAG, "Model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model: ${e.message}")
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
        val processedImage = preprocessImage(bitmap)
        
        // Prepare output buffer (adjust based on your model)
        val outputBuffer = ByteBuffer.allocateDirect(4 * 100 * 7)
        outputBuffer.order(ByteOrder.nativeOrder())
        
        // Run inference
        interpreter?.run(processedImage.buffer, outputBuffer)
        
        // Process results
        return processDetections(outputBuffer)
    }
    
    private fun preprocessImage(bitmap: Bitmap): TensorImage {
        // Resize and normalize the image
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(modelInputHeight, modelInputWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f)) // Normalize to [0,1]
            .build()
        
        var tensorImage = TensorImage.fromBitmap(bitmap)
        tensorImage = imageProcessor.process(tensorImage)
        
        return tensorImage
    }
    
    private fun processDetections(outputBuffer: ByteBuffer): Map<String, Int> {
        // Reset the buffer position
        outputBuffer.rewind()
        
        val acneCounts = mutableMapOf(
            "comedone" to 0,
            "pustule" to 0,
            "papule" to 0,
            "nodule" to 0
        )
        
        // Process detections (similar to ImageAnalyzer.kt)
        val numDetections = 100
        
        for (i in 0 until numDetections) {
            val confidence = outputBuffer.getFloat((i * 7 + 4) * 4)
            
            if (confidence > 0.5) {
                val classId = outputBuffer.getFloat((i * 7 + 5) * 4).toInt()
                
                when (classId) {
                    0 -> acneCounts["comedone"] = acneCounts["comedone"]!! + 1
                    1 -> acneCounts["pustule"] = acneCounts["pustule"]!! + 1
                    2 -> acneCounts["papule"] = acneCounts["papule"]!! + 1
                    3 -> acneCounts["nodule"] = acneCounts["nodule"]!! + 1
                }
            }
        }
        
        return acneCounts
    }

    private fun displayResults(acneCounts: Map<String, Int>) {
        // Calculate severity
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
        val normalizedSeverity = (normalizedSeverityFloat * 10).toInt()
        
        val severityLevel = when {
            normalizedSeverity < 3 -> "Low"
            normalizedSeverity < 7 -> "Medium"
            else -> "High"
        }
        
        // Prepare display text
        val resultsText = StringBuilder()
        resultsText.append("Acne Severity: $severityLevel ($normalizedSeverity/10)\n\n")
        resultsText.append("Acne Counts:\n")
        
        acneCounts.forEach { (type, count) ->
            resultsText.append("${type.capitalize()}: $count\n")
        }
        
        resultsText.append("\nTotal Acne Count: $totalCount")
        
        // Update the UI
        resultTextView.text = resultsText.toString()
        
        // Start Result activity with the analysis
        val intent = Intent(this, ResultActivity::class.java)
        intent.putExtra("severity", normalizedSeverity)
        intent.putExtra("total_count", totalCount)
        intent.putExtra("comedone_count", acneCounts["comedone"] ?: 0)
        intent.putExtra("pustule_count", acneCounts["pustule"] ?: 0)
        intent.putExtra("papule_count", acneCounts["papule"] ?: 0)
        intent.putExtra("nodule_count", acneCounts["nodule"] ?: 0)
        intent.putExtra("timestamp", System.currentTimeMillis())
        
        // Get image bytes from any of the potential sources
        val imageBytes = when {
            intent.hasExtra("image_bytes") -> intent.getByteArrayExtra("image_bytes")
            intent.hasExtra("forehead") -> intent.getByteArrayExtra("forehead") // Just use forehead as example
            else -> null
        }
        intent.putExtra("image", imageBytes)
        
        // Add detection data for bounding boxes (currently empty)
        intent.putExtra("detections_count", 0) // No detections yet
        
        startActivity(intent)
        finish() // Close this activity
    }
    
    override fun onDestroy() {
        super.onDestroy()
        interpreter?.close()
    }
}
