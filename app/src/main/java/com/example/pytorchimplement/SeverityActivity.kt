package com.example.pytorchimplement

import android.annotation.SuppressLint
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.io.ByteArrayOutputStream
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.provider.MediaStore

class SeverityActivity : AppCompatActivity() {

    private val TAG = "SeverityActivity"
    
    // List of facial regions that need to be selected
    private val facialRegions = listOf(
        "forehead", "nose", "left_cheek", "right_cheek", "chin"
    )
    
    // Map to store selected images for each region
    private val selectedImages = mutableMapOf<String, ByteArray>()
    
    // Current region being selected
    private var currentRegion = ""
    
    // Image picker launcher
    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            try {
                // Load and process the selected image
                val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
                
                // Resize bitmap if needed (to reduce memory usage)
                val resizedBitmap = resizeBitmapIfNeeded(bitmap, 640, 640)
                
                // Convert to byte array
                val byteArray = bitmapToByteArray(resizedBitmap)
                
                // Store the image for the current region
                selectedImages[currentRegion] = byteArray
                
                // Show success message
                Toast.makeText(
                    this,
                    "Image selected for ${getRegionDisplayName(currentRegion)}",
                    Toast.LENGTH_SHORT
                ).show()
                
                // Move to next region or proceed to analysis
                selectNextRegion()
                
            } catch (e: Exception) {
                Log.e(TAG, "Error processing selected image: ${e.message}")
                Toast.makeText(
                    this,
                    "Error processing image. Please try again.",
                    Toast.LENGTH_SHORT
                ).show()
            }
        } else {
            Toast.makeText(
                this,
                "No image selected for ${getRegionDisplayName(currentRegion)}",
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_severity)

        // Set up button click listeners
        val realtimeButton = findViewById<Button>(R.id.realtime_severity)
        val storageButton = findViewById<Button>(R.id.storage_severity)

        realtimeButton.setOnClickListener {
            // Redirect to CaptureActivity for capturing facial region images with camera
            val intent = Intent(this, CaptureActivity::class.java)
            startActivity(intent)
        }

        storageButton.setOnClickListener {
            // Launch the StorageSelectionActivity for selecting images from device storage
            val intent = Intent(this, StorageSelectionActivity::class.java)
            startActivity(intent)
        }
    }
    @SuppressLint("MissingSuperCall")
    override fun onBackPressed() {
        // Do Here what ever you want do on back press;
        val intent = Intent(this, MainActivity::class.java)
        startActivity(intent)
        finish()
    }
    
    private fun startImageSelectionProcess() {
        // Reset selected images
        selectedImages.clear()
        
        // Start with the first region
        currentRegion = facialRegions.first()
        
        // Show instruction to the user
        Toast.makeText(
            this,
            "Please select an image for ${getRegionDisplayName(currentRegion)}",
            Toast.LENGTH_LONG
        ).show()
        
        // Launch image picker
        imagePickerLauncher.launch("image/*")
    }
    
    private fun selectNextRegion() {
        // Find the index of the current region
        val currentIndex = facialRegions.indexOf(currentRegion)
        
        // Check if there are more regions to select
        if (currentIndex < facialRegions.size - 1) {
            // Move to the next region
            currentRegion = facialRegions[currentIndex + 1]
            
            // Show instruction to the user
            Toast.makeText(
                this,
                "Please select an image for ${getRegionDisplayName(currentRegion)}",
                Toast.LENGTH_LONG
            ).show()
            
            // Launch image picker
            imagePickerLauncher.launch("image/*")
        } else {
            // All regions selected, proceed to analysis
            proceedToAnalysis()
        }
    }
    
    private fun proceedToAnalysis() {
        // Check if all regions have been selected
        if (selectedImages.size == facialRegions.size) {
            // Create intent for ModelInferenceActivity
            val intent = Intent(this, ModelInferenceActivity::class.java)
            
            // Add all selected images to the intent
            selectedImages.forEach { (region, imageBytes) ->
                intent.putExtra(region, imageBytes)
            }
            
            // Start the activity
            startActivity(intent)
        } else {
            // Some regions are missing
            val missingRegions = facialRegions.filter { !selectedImages.containsKey(it) }
            
            Toast.makeText(
                this,
                "Missing images for: ${missingRegions.joinToString(", ") { getRegionDisplayName(it) }}",
                Toast.LENGTH_LONG
            ).show()
            
            // Restart the process
            startImageSelectionProcess()
        }
    }
    
    private fun getRegionDisplayName(regionId: String): String {
        return when (regionId) {
            "forehead" -> "Forehead"
            "nose" -> "Nose"
            "left_cheek" -> "Left Cheek"
            "right_cheek" -> "Right Cheek"
            "chin" -> "Chin"
            else -> regionId.capitalize()
        }
    }
    
    private fun bitmapToByteArray(bitmap: Bitmap): ByteArray {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream)
        return stream.toByteArray()
    }
    
    private fun resizeBitmapIfNeeded(bitmap: Bitmap, maxWidth: Int, maxHeight: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        
        // Check if resize is needed
        if (width <= maxWidth && height <= maxHeight) {
            return bitmap
        }
        
        // Calculate new dimensions while maintaining aspect ratio
        val ratio = Math.min(
            maxWidth.toFloat() / width.toFloat(),
            maxHeight.toFloat() / height.toFloat()
        )
        
        val newWidth = (width * ratio).toInt()
        val newHeight = (height * ratio).toInt()
        
        // Resize the bitmap
        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }
}
