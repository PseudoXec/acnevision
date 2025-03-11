package com.example.pytorchimplement

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.io.ByteArrayOutputStream

/**
 * Activity for selecting images from device storage for each facial region
 */
class StorageSelectionActivity : AppCompatActivity() {

    private val TAG = "StorageSelectionActivity"
    
    // UI elements
    private lateinit var titleTextView: TextView
    private lateinit var foreheadButton: Button
    private lateinit var noseButton: Button
    private lateinit var leftCheekButton: Button
    private lateinit var rightCheekButton: Button
    private lateinit var chinButton: Button
    private lateinit var proceedButton: Button
    
    private lateinit var foreheadImageView: ImageView
    private lateinit var noseImageView: ImageView
    private lateinit var leftCheekImageView: ImageView
    private lateinit var rightCheekImageView: ImageView
    private lateinit var chinImageView: ImageView
    
    // Map of region IDs to their corresponding buttons and image views
    private lateinit var regionButtons: Map<String, Button>
    private lateinit var regionImageViews: Map<String, ImageView>
    
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
                
                // Update the UI
                updateRegionUI(currentRegion, resizedBitmap)
                
                // Show success message
                Toast.makeText(
                    this,
                    "Image selected for ${getRegionDisplayName(currentRegion)}",
                    Toast.LENGTH_SHORT
                ).show()
                
                // Update proceed button state
                updateProceedButtonState()
                
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
        setContentView(R.layout.activity_storage_selection)
        
        // Initialize UI elements
        titleTextView = findViewById(R.id.selection_title)
        
        foreheadButton = findViewById(R.id.forehead_button)
        noseButton = findViewById(R.id.nose_button)
        leftCheekButton = findViewById(R.id.left_cheek_button)
        rightCheekButton = findViewById(R.id.right_cheek_button)
        chinButton = findViewById(R.id.chin_button)
        proceedButton = findViewById(R.id.proceed_button)
        
        foreheadImageView = findViewById(R.id.forehead_image)
        noseImageView = findViewById(R.id.nose_image)
        leftCheekImageView = findViewById(R.id.left_cheek_image)
        rightCheekImageView = findViewById(R.id.right_cheek_image)
        chinImageView = findViewById(R.id.chin_image)
        
        // Map regions to their UI elements
        regionButtons = mapOf(
            "forehead" to foreheadButton,
            "nose" to noseButton,
            "left_cheek" to leftCheekButton,
            "right_cheek" to rightCheekButton,
            "chin" to chinButton
        )
        
        regionImageViews = mapOf(
            "forehead" to foreheadImageView,
            "nose" to noseImageView,
            "left_cheek" to leftCheekImageView,
            "right_cheek" to rightCheekImageView,
            "chin" to chinImageView
        )
        
        // Set up button click listeners
        foreheadButton.setOnClickListener { selectImageForRegion("forehead") }
        noseButton.setOnClickListener { selectImageForRegion("nose") }
        leftCheekButton.setOnClickListener { selectImageForRegion("left_cheek") }
        rightCheekButton.setOnClickListener { selectImageForRegion("right_cheek") }
        chinButton.setOnClickListener { selectImageForRegion("chin") }
        
        proceedButton.setOnClickListener { proceedToAnalysis() }
        
        // Initially disable the proceed button
        proceedButton.isEnabled = false
        
        // Show initial instructions
        Toast.makeText(
            this,
            "Please select images for all facial regions",
            Toast.LENGTH_LONG
        ).show()
    }
    
    private fun selectImageForRegion(region: String) {
        currentRegion = region
        imagePickerLauncher.launch("image/*")
    }
    
    private fun updateRegionUI(region: String, bitmap: Bitmap) {
        // Update the image view
        regionImageViews[region]?.setImageBitmap(bitmap)
        
        // Update the button text
        regionButtons[region]?.text = "Change ${getRegionDisplayName(region)}"
    }
    
    private fun updateProceedButtonState() {
        // Enable the proceed button if all regions have been selected
        proceedButton.isEnabled = selectedImages.size == 5
        
        if (proceedButton.isEnabled) {
            proceedButton.setBackgroundColor(resources.getColor(R.color.green, theme))
            Toast.makeText(
                this,
                "All regions selected! You can now proceed to analysis.",
                Toast.LENGTH_SHORT
            ).show()
        }
    }
    
    private fun proceedToAnalysis() {
        // Create intent for ModelInferenceActivity
        val intent = Intent(this, ModelInferenceActivity::class.java)
        
        // Add all selected images to the intent
        selectedImages.forEach { (region, imageBytes) ->
            intent.putExtra(region, imageBytes)
        }
        
        // Start the activity
        startActivity(intent)
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