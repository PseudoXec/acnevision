package com.example.pytorchimplement

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.appcompat.app.AppCompatActivity
import java.io.ByteArrayOutputStream
import java.io.InputStream
import androidx.activity.result.contract.ActivityResultContracts

class SeverityActivity : AppCompatActivity() {
    private val selectedImages: MutableList<ByteArray> = ArrayList()
    private lateinit var pickImagesLauncher: ActivityResultLauncher<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_severity)

        pickImagesLauncher = registerForActivityResult(
            ActivityResultContracts.GetMultipleContents()
        ) { uris: List<Uri> ->
            if (uris.isNotEmpty()) {
                selectedImages.clear()
                for (uri in uris.take(5)) {
                    val imageBytes = convertUriToByteArray(uri)
                    if (imageBytes != null) {
                        selectedImages.add(imageBytes)
                    }
                }

                if (selectedImages.isNotEmpty()) {
                    Toast.makeText(this, "Images selected, processing...", Toast.LENGTH_SHORT).show()
                    processImages()
                }
            }
        }

        val openCameraButton = findViewById<Button>(R.id.realtime_severity)
        val openGalleryButton = findViewById<Button>(R.id.storage_severity)

        openCameraButton.setOnClickListener {
            val intent = Intent(this, CaptureActivity::class.java)
            startActivity(intent)
        }

        openGalleryButton.setOnClickListener {
            openGallery()
        }
    }

    private fun openGallery() {
        pickImagesLauncher.launch("image/*")
    }

    //Convert images from URI to ByteArray
    private fun convertUriToByteArray(uri: Uri): ByteArray? {
        return try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            val byteArrayOutputStream = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream)
            inputStream?.close()
            byteArrayOutputStream.toByteArray()
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
    // Pass to Inference Activity
    private fun processImages() {
        val intent = Intent(this, ModelInferenceActivity::class.java)
        intent.putExtra("selected_images", selectedImages.toTypedArray())
        startActivity(intent)
    }
}
