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
import android.view.LayoutInflater
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.graphics.drawable.toBitmap
import com.example.pytorchimplement.databinding.ActivityCaptureActivityBinding
import com.example.pytorchimplement.databinding.ItemFacialRegionBinding
import java.io.ByteArrayOutputStream

class CaptureActivity : AppCompatActivity() {

    // View binding
    private lateinit var binding: ActivityCaptureActivityBinding

    // Region data 
    private data class FacialRegion(
        val id: String,
        val displayName: String,
        val binding: ItemFacialRegionBinding
    )
    
    private val regions = mutableMapOf<String, FacialRegion>()
    private var capturedImages = mutableMapOf<String, Bitmap?>()
    private var currentCaptureKey: String? = null
    
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
            currentCaptureKey?.let { launchCamera(it) }
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
            if (imageBitmap != null && currentCaptureKey != null) {
                capturedImages[currentCaptureKey!!] = imageBitmap
                updateImageView(currentCaptureKey!!, imageBitmap)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCaptureActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize facial regions
        setupFacialRegion("forehead", "Forehead Region", binding.foreheadRegion)
        setupFacialRegion("nose", "Nose Region", binding.noseRegion)
        setupFacialRegion("right_cheek", "Right Cheek Region", binding.rightCheekRegion)
        setupFacialRegion("left_cheek", "Left Cheek Region", binding.leftCheekRegion)
        setupFacialRegion("chin", "Chin Region", binding.chinRegion)

        // Initialize "Done" button
        binding.doneCaptureButton.setOnClickListener { processCapturedImages() }
    }

    // Setup a facial region with its views and listeners
    private fun setupFacialRegion(regionId: String, displayName: String, regionBinding: ItemFacialRegionBinding) {
        // Setup button text
        regionBinding.captureButton.text = displayName
        
        // Store region info
        regions[regionId] = FacialRegion(regionId, displayName, regionBinding)
        
        // Set click listener
        regionBinding.captureButton.setOnClickListener { openCamera(regionId) }
    }

    // Check and request camera permission
    private fun openCamera(region: String) {
        currentCaptureKey = region
        
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
        updateImageView(region, bitmap)
        
        Toast.makeText(
            this,
            "Using test image for $region",
            Toast.LENGTH_SHORT
        ).show()
    }

    // Update ImageView Based on Region
    private fun updateImageView(region: String, bitmap: Bitmap) {
        regions[region]?.binding?.regionImage?.setImageBitmap(bitmap)
    }

    // Convert Bitmap to ByteArray
    private fun bitmapToByteArray(bitmap: Bitmap): ByteArray {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
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
}
