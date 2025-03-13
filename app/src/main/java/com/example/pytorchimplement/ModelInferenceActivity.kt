package com.example.pytorchimplement

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import android.content.Intent
import android.view.View
import android.widget.Button
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ModelInferenceActivity : AppCompatActivity() {
    private lateinit var resultTextView: TextView
    private lateinit var postScanImageAnalyzer: PostScanImageAnalyzer
    private lateinit var retryButton: Button
    
    // Storage directory for analysis results
    private lateinit var storageDir: File
    
    // Add these variables to track progress
    private var totalRegions = 5
    private var processedRegions = 0
    
    companion object {
        private const val TAG = "ModelInferenceActivity"
        private const val RESULTS_DIR = "acne_analysis_results"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_inference)

        resultTextView = findViewById(R.id.resultsTextView)
        
        // Initialize storage directory
        storageDir = File(getExternalFilesDir(null), RESULTS_DIR)
        if (!storageDir.exists()) {
            storageDir.mkdirs()
        }
        
        // Initialize the analyzer
        postScanImageAnalyzer = PostScanImageAnalyzer(this)
        
        // Show initial status message
        resultTextView.text = "Preparing facial regions for analysis...\n\nThis may take a moment."
        
        // Process images in a background thread
        Thread {
            try {
                val images = loadImagesFromIntent()
                if (images.size == 5) {
                    // Update UI to show analysis is starting
                    runOnUiThread {
                        resultTextView.text = "Analyzing facial regions...\n\nPlease wait while we process your images."
                    }
                    
                    // Perform the analysis
                    val analysisStartTime = System.currentTimeMillis()
                    val result = postScanImageAnalyzer.analyzeAllRegions(images)
                    val analysisTime = System.currentTimeMillis() - analysisStartTime
                    
                    // Save results and images
                    val resultId = saveAnalysisResult(result, images)
                    
                    // Update UI on main thread with completed analysis
                    runOnUiThread {
                        // Enhanced results display
                        val resultBuilder = StringBuilder()
                        resultBuilder.append("✓ Analysis complete!\n\n")
                        resultBuilder.append("📊 GAGS Score: ${result.totalGAGSScore}\n")
                        resultBuilder.append("📋 Severity: ${result.severity}\n")
                        resultBuilder.append("⏱️ Total Inference Time: ${result.totalInferenceTimeMs}ms\n")
                        resultBuilder.append("⌛ Total Analysis Time: ${analysisTime}ms\n\n")
                        resultBuilder.append("Navigating to detailed results...")
                        
                        resultTextView.text = resultBuilder.toString()
                        
                        // Store in singleton for easy access
                        GAGSData.totalGAGSScore = result.totalGAGSScore
                        GAGSData.severity = result.severity
                        GAGSData.totalAcneCounts = result.totalAcneCounts
                        GAGSData.inferenceTimeMs = result.totalInferenceTimeMs
                        
                        // Add a short delay before navigating to results screen
                        resultTextView.postDelayed({
                            navigateToResultActivity(resultId)
                        }, 1500)
                    }
                } else {
                    runOnUiThread {
                        resultTextView.text = "❌ Error: Not all facial regions were provided.\n" +
                                              "Found ${images.size}/5 regions.\n\n" +
                                              "Please try capturing all regions again."
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing images: ${e.message}")
                e.printStackTrace()
                
                runOnUiThread {
                    resultTextView.text = "❌ Error analyzing images:\n${e.message}\n\n" +
                                          "Please try again or check logs for details."
                }
            } finally {
                // Clean up resources
                postScanImageAnalyzer.close()
            }
        }.start()
    }

    private fun loadImagesFromIntent(): Map<String, Bitmap> {
        val images = mutableMapOf<String, Bitmap>()
        
        try {
            // Check for facial region images from CaptureActivity
            val regions = listOf("forehead", "nose", "left_cheek", "right_cheek", "chin")
            
            regions.forEach { region ->
                intent.getByteArrayExtra(region)?.let { imageBytes ->
                    val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                    images[region] = bitmap
                    Log.d(TAG, "Loaded image for region: $region, size: ${bitmap.width}x${bitmap.height}")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading images from intent: ${e.message}")
            e.printStackTrace()
        }
        
        return images
    }
    
    private fun saveAnalysisResult(
        result: PostScanImageAnalyzer.AnalysisResult,
        images: Map<String, Bitmap>
    ): String {
        // Create a unique ID for this analysis session
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val resultId = "analysis_$timestamp"
        
        // Create directory for this analysis
        val resultDir = File(storageDir, resultId)
        resultDir.mkdirs()
        
        try {
            // Save each image
            images.forEach { (region, bitmap) ->
                val imageFile = File(resultDir, "$region.jpg")
                FileOutputStream(imageFile).use { out ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
                }
                Log.d(TAG, "Saved image for $region to ${imageFile.absolutePath}")
            }
            
            // Save detections for each region
            result.regionResults.forEach { (region, regionResult) ->
                // Save region analysis summary with inference time
                val summaryFile = File(resultDir, "${region.id}_summary.txt")
                summaryFile.writeText(
                    "Region: ${region.displayName}\n" +
                    "Dominant Acne Type: ${regionResult.dominantAcneType}\n" +
                    "Region Score: ${regionResult.regionScore}\n" +
                    "Acne Counts: ${regionResult.acneCounts}\n" +
                    "Detections: ${regionResult.detections.size}\n" +
                    "Inference Time: ${regionResult.inferenceTimeMs}ms\n"
                )
                
                // Save detailed detection data
                if (regionResult.detections.isNotEmpty()) {
                    val detectionsFile = File(resultDir, "${region.id}_detections.csv")
                    val header = "class,confidence,x,y,width,height\n"
                    val detectionRows = regionResult.detections.joinToString("\n") { detection ->
                        "${detection.className},${detection.confidence}," +
                        "${detection.boundingBox.x},${detection.boundingBox.y}," +
                        "${detection.boundingBox.width},${detection.boundingBox.height}"
                    }
                    detectionsFile.writeText(header + detectionRows)
                }
            }
            
            // Save overall analysis summary
            val overallSummaryFile = File(resultDir, "analysis_summary.txt")
            overallSummaryFile.writeText(
                "Analysis ID: $resultId\n" +
                "Timestamp: ${Date(result.timestamp)}\n" +
                "Total GAGS Score: ${result.totalGAGSScore}\n" +
                "Severity: ${result.severity}\n" +
                "Total Acne Counts: ${result.totalAcneCounts}\n" +
                "Total Inference Time: ${result.totalInferenceTimeMs}ms\n"
            )
            
            Log.d(TAG, "Saved analysis results to ${resultDir.absolutePath}")
            
            return resultId
            
        } catch (e: Exception) {
            Log.e(TAG, "Error saving analysis results: ${e.message}")
            e.printStackTrace()
            return resultId
        }
    }
    
    private fun navigateToResultActivity(resultId: String) {
        val intent = Intent(this, ResultActivity::class.java).apply {
            putExtra("analysis_id", resultId)
            putExtra("severity", GAGSData.severity)
            putExtra("total_score", GAGSData.totalGAGSScore)
            putExtra("total_count", GAGSData.totalAcneCounts.values.sum())
            putExtra("comedone_count", GAGSData.totalAcneCounts["comedone"] ?: 0)
            putExtra("pustule_count", GAGSData.totalAcneCounts["pustule"] ?: 0)
            putExtra("papule_count", GAGSData.totalAcneCounts["papule"] ?: 0)
            putExtra("nodule_count", GAGSData.totalAcneCounts["nodule"] ?: 0)
            putExtra("timestamp", System.currentTimeMillis())
            putExtra("inference_time", GAGSData.inferenceTimeMs)
        }
        
        startActivity(intent)
        finish()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        postScanImageAnalyzer.close()
    }
}
