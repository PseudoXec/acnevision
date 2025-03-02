package com.example.pytorchimplement

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.io.ByteArrayOutputStream

class CaptureActivity : AppCompatActivity() {

    private lateinit var chinButton: Button
    private lateinit var lCheekButton: Button
    private lateinit var rCheekButton: Button
    private lateinit var noseButton: Button
    private lateinit var foreheadButton: Button
    private lateinit var doneButton: Button

    private lateinit var chinImage: ImageView
    private lateinit var lCheekImage: ImageView
    private lateinit var rCheekImage: ImageView
    private lateinit var noseImage: ImageView
    private lateinit var foreheadImage: ImageView

    private var capturedImages = mutableMapOf<String, Bitmap?>()
    private var currentCaptureKey: String? = null

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
        setContentView(R.layout.activity_capture_activity)

        // Initialize Buttons
        chinButton = findViewById(R.id.chinButton)
        lCheekButton = findViewById(R.id.LCheekButton)
        rCheekButton = findViewById(R.id.RCheekButton)
        noseButton = findViewById(R.id.noseButton)
        foreheadButton = findViewById(R.id.foreheadButton)
        doneButton = findViewById(R.id.DoneCaptureImage)

        // Initialize ImageViews
        chinImage = findViewById(R.id.chinImage)
        lCheekImage = findViewById(R.id.LCheekImage)
        rCheekImage = findViewById(R.id.RCheekImage)
        noseImage = findViewById(R.id.noseImage)
        foreheadImage = findViewById(R.id.foreheadImage)

        // Set Click Listeners
        chinButton.setOnClickListener { openCamera("chin") }
        lCheekButton.setOnClickListener { openCamera("left_cheek") }
        rCheekButton.setOnClickListener { openCamera("right_cheek") }
        noseButton.setOnClickListener { openCamera("nose") }
        foreheadButton.setOnClickListener { openCamera("forehead") }
        doneButton.setOnClickListener { processCapturedImages() }
    }

    // Open Camera for Capture
    private fun openCamera(region: String) {
        currentCaptureKey = region
        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        if (cameraIntent.resolveActivity(packageManager) != null) {
            cameraLauncher.launch(cameraIntent)
        } else {
            Toast.makeText(this, "Camera not available", Toast.LENGTH_SHORT).show()
        }
    }

    // Update ImageView Based on Region
    private fun updateImageView(region: String, bitmap: Bitmap) {
        when (region) {
            "chin" -> chinImage.setImageBitmap(bitmap)
            "left_cheek" -> lCheekImage.setImageBitmap(bitmap)
            "right_cheek" -> rCheekImage.setImageBitmap(bitmap)
            "nose" -> noseImage.setImageBitmap(bitmap)
            "forehead" -> foreheadImage.setImageBitmap(bitmap)
        }
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
