package com.example.pytorchimplement

import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class ModelInferenceActivity : AppCompatActivity() {
    private lateinit var model: Module

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_inference)

        // Load model
        model = Module.load(assetFilePath("MODEL NAME"))

        // Retrieve images
        val imageKeys = listOf("image1", "image2", "image3", "image4", "image5")
        val images = imageKeys.mapNotNull { key ->
            intent.getByteArrayExtra(key)?.let { byteArrayToBitmap(it) }
        }

        // Run inference and count acne occurrences
        val totalCounts = mutableMapOf("comedone" to 0, "pustule" to 0, "papule" to 0, "nodules" to 0)
        images.forEach { bitmap ->
            val outputTensor = runModel(bitmap)
            val counts = parseResults(outputTensor)

            totalCounts.keys.forEach { key ->
                totalCounts[key] = totalCounts[key]!! + counts[key]!!
            }
        }

        // Display results (you can update a TextView or RecyclerView)
        displayResults(totalCounts)
    }

    private fun runModel(bitmap: Bitmap): Tensor {
        val inputTensor = preprocessImage(bitmap)
        return model.forward(IValue.from(inputTensor)).toTensor()
    }

    private fun parseResults(outputTensor: Tensor): Map<String, Int> {
        val acneCount = mutableMapOf("comedone" to 0, "pustule" to 0, "papule" to 0, "nodules" to 0)
        val results = outputTensor.dataAsFloatArray

        for (i in results.indices step 6) {  // Assuming output format is [x, y, w, h, confidence, class_id]
            val classId = results[i + 5].toInt()
            when (classId) {
                0 -> acneCount["comedone"] = acneCount["comedone"]!! + 1
                1 -> acneCount["pustule"] = acneCount["pustule"]!! + 1
                2 -> acneCount["papule"] = acneCount["papule"]!! + 1
                3 -> acneCount["nodules"] = acneCount["nodules"]!! + 1
            }
        }
        return acneCount
    }

    private fun displayResults(totalCounts: Map<String, Int>) {
        // Update UI elements with counts
        findViewById<TextView>(R.id.resultsTextView).text = totalCounts.toString()
    }
}
