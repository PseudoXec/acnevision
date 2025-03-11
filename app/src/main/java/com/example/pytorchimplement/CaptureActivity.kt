package com.example.pytorchimplement

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream

class CaptureActivity : AppCompatActivity() {

    private val TAG = "CaptureActivity"

    // Region data structure
    private data class FacialRegion(
        val id: String,
        val displayName: String,
        val imageView: ImageView,
        val button: Button,
        val placeholder: TextView
    )
    
    // Map of regions
    private val regions = mutableMapOf<String, FacialRegion>()
    
    // Map to store captured images
    private val capturedImages = mutableMapOf<String, Bitmap?>()
    
    // Current region being captured
    private var currentRegion: String? = null
    
    // Proceed button
    private lateinit var proceedButton: Button
    
    // Flag to detect if running on an emulator (for testing)
    private val isEmulator: Boolean by lazy {
        (Build.BRAND.startsWith("generic") && Build.DEVICE.startsWith("generic"))
            || Build.FINGERPRINT.startsWith("generic")
            || Build.FINGERPRINT.startsWith("unknown")
            || Build.HARDWARE.contains("goldfish")
            || Build.HARDWARE.contains("ranchu")
            || Build.MODEL.contains("google_sdk")
            || Build.MODEL.contains("Emulator")
            || Build.MODEL.contains("Android SDK built for x86")
            || Build.MANUFACTURER.contains("Genymotion")
            || Build.PRODUCT.contains("sdk_google")
            || Build.PRODUCT.contains("google_sdk")
            || Build.PRODUCT.contains("sdk")
            || Build.PRODUCT.contains("sdk_x86")
            || Build.PRODUCT.contains("sdk_gphone64_arm64")
            || Build.PRODUCT.contains("vbox86p")
            || Build.PRODUCT.contains("emulator")
            || Build.PRODUCT.contains("simulator")
    }

    // Camera permission request
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            // Permission granted, proceed with camera
            currentRegion?.let { launchCamera(it) }
        } else {
            // Permission denied
            Toast.makeText(
                this,
                "Camera permission is required to capture facial regions",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    // Camera result handler
    private val cameraLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val imageBitmap = result.data?.extras?.get("data") as? Bitmap
            if (imageBitmap != null && currentRegion != null) {
                // Resize bitmap if needed (to reduce memory usage)
                val resizedBitmap = resizeBitmapIfNeeded(imageBitmap, 640, 640)
                
                // Store the captured image
                capturedImages[currentRegion!!] = resizedBitmap
                
                // Update the UI
                updateRegionUI(currentRegion!!, resizedBitmap)
                
                // Show success message
                Toast.makeText(
                    this,
                    "Image captured for ${getRegionDisplayName(currentRegion!!)}",
                    Toast.LENGTH_SHORT
                ).show()
                
                // Update proceed button state
                updateProceedButtonState()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_capture_activity)

        // Initialize proceed button
        proceedButton = findViewById(R.id.done_capture_button)
        proceedButton.isEnabled = false
        
        // Initialize facial regions
        setupFacialRegion(
            "forehead",
            findViewById(R.id.forehead_region_image),
            findViewById(R.id.forehead_region_button),
            findViewById(R.id.forehead_placeholder)
        )
        
        setupFacialRegion(
            "nose",
            findViewById(R.id.nose_region_image),
            findViewById(R.id.nose_region_button),
            findViewById(R.id.nose_placeholder)
        )
        
        setupFacialRegion(
            "left_cheek",
            findViewById(R.id.left_cheek_region_image),
            findViewById(R.id.left_cheek_region_button),
            findViewById(R.id.left_cheek_placeholder)
        )
        
        setupFacialRegion(
            "right_cheek",
            findViewById(R.id.right_cheek_region_image),
            findViewById(R.id.right_cheek_region_button),
            findViewById(R.id.right_cheek_placeholder)
        )
        
        setupFacialRegion(
            "chin",
            findViewById(R.id.chin_region_image),
            findViewById(R.id.chin_region_button),
            findViewById(R.id.chin_placeholder)
        )

        // Set up "Proceed" button
        proceedButton.setOnClickListener { processCapturedImages() }
        
        // Show initial instructions
        Toast.makeText(
            this,
            "Capture images for all facial regions",
            Toast.LENGTH_LONG
        ).show()
    }

    // Setup a facial region with its views and listeners
    private fun setupFacialRegion(
        regionId: String,
        imageView: ImageView,
        button: Button,
        placeholder: TextView
    ) {
        // Get display name
        val displayName = getRegionDisplayName(regionId)
        
        // Store region info
        regions[regionId] = FacialRegion(
            id = regionId,
            displayName = displayName,
            imageView = imageView,
            button = button,
            placeholder = placeholder
        )
        
        // Set click listener
        button.setOnClickListener { openCamera(regionId) }
    }

    // Check and request camera permission
    private fun openCamera(region: String) {
        currentRegion = region
        
        when {
            // Permission already granted
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                launchCamera(region)
            }
            
            // Should show rationale for permission
            shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) -> {
                Toast.makeText(
                    this,
                    "Camera permission is needed to capture facial regions",
                    Toast.LENGTH_LONG
                ).show()
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
            
            // Request permission
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    // Launch camera after permission is granted
    private fun launchCamera(region: String) {
        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        
        // Check if there's an app that can handle the camera intent
        if (cameraIntent.resolveActivity(packageManager) != null) {
            cameraLauncher.launch(cameraIntent)
        } else {
            // No app to handle camera intent (common in emulators)
            handleMissingCameraApp(region)
        }
    }
    
    // Handle case when no camera app is available (likely in emulator)
    private fun handleMissingCameraApp(region: String) {
        if (isEmulator) {
            // Show options dialog when running on emulator
            AlertDialog.Builder(this)
                .setTitle("Testing Options")
                .setMessage("You appear to be running on an emulator which may not support camera. Choose an option for testing:")
                .setPositiveButton("Use Placeholder Image") { _, _ ->
                    // Generate a colored test image for the specific region
                    useTestImage(region)
                }
                .setNegativeButton("Cancel") { dialog, _ ->
                    dialog.dismiss()
                }
                .show()
        } else {
            // Running on a real device but still no camera app
            Toast.makeText(
                this, 
                "No camera application available", 
                Toast.LENGTH_LONG
            ).show()
        }
    }
    
    // Create a test image for emulator testing
    private fun useTestImage(region: String) {
        // Create different colored test images for different regions
        val color = when(region) {
            "forehead" -> Color.rgb(255, 200, 200) // Light red
            "nose" -> Color.rgb(200, 255, 200)     // Light green
            "right_cheek" -> Color.rgb(200, 200, 255) // Light blue
            "left_cheek" -> Color.rgb(255, 255, 200)  // Light yellow
            "chin" -> Color.rgb(255, 200, 255)     // Light magenta
            else -> Color.LTGRAY
        }
        
        // Create a bitmap with the specified color
        val bitmap = Bitmap.createBitmap(300, 300, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(color)
        
        // Save and display the test image
        capturedImages[region] = bitmap
        updateRegionUI(region, bitmap)
        
        // Update proceed button state
        updateProceedButtonState()
        
        Toast.makeText(
            this,
            "Using test image for ${getRegionDisplayName(region)}",
            Toast.LENGTH_SHORT
        ).show()
    }

    // Update UI for a region after capturing an image
    private fun updateRegionUI(region: String, bitmap: Bitmap) {
        val facialRegion = regions[region] ?: return
        
        // Update image view
        facialRegion.imageView.setImageBitmap(bitmap)
        
        // Hide placeholder text
        facialRegion.placeholder.visibility = View.GONE
        
        // Update button text
        facialRegion.button.text = "Retake ${facialRegion.displayName}"
    }
    
    // Update the proceed button state based on captured images
    private fun updateProceedButtonState() {
        // Enable the proceed button if all regions have been captured
        val allCaptured = regions.keys.all { capturedImages.containsKey(it) }
        
        proceedButton.isEnabled = allCaptured
        
        if (proceedButton.isEnabled) {
            proceedButton.setBackgroundColor(resources.getColor(R.color.green, theme))
            Toast.makeText(
                this,
                "All regions captured! You can now proceed to analysis.",
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    // Convert Bitmap to ByteArray
    private fun bitmapToByteArray(bitmap: Bitmap): ByteArray {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream)
        return stream.toByteArray()
    }

    // Handle Done Button Click
    private fun processCapturedImages() {
        if (capturedImages.size == 5) {
            val intent = Intent(this, ModelInferenceActivity::class.java)

            // Convert each captured image to ByteArray and send it via Intent
            capturedImages.forEach { (key, bitmap) ->
                bitmap?.let {
                    intent.putExtra(key, bitmapToByteArray(it))
                }
            }

            startActivity(intent) // Start inference activity
        } else {
            Toast.makeText(this, "Please capture all 5 images before proceeding!", Toast.LENGTH_LONG).show()
        }
    }
    
    // Helper function to get display name for a region
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
    
    // Resize bitmap if needed to reduce memory usage
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
