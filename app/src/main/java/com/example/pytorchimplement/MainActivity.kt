package com.example.pytorchimplement

import android.content.Intent
import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import android.widget.Button

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        val realTimeButton = findViewById<Button>(R.id.Realtime_button)
        val severityButton = findViewById<Button>(R.id.severity_button)

        realTimeButton.setOnClickListener {
            val intent = Intent(this, RealTimeActivity::class.java)
            startActivity(intent)
        }

        severityButton.setOnClickListener {
            val intent = Intent(this, SeverityActivity::class.java)
            startActivity(intent)
        }
    }
}