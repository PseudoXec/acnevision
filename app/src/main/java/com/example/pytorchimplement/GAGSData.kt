package com.example.pytorchimplement

object GAGSData {
    var totalGAGSScore: Int = 0
    var severity: String = "Unknown"
    var totalAcneCounts: Map<String, Int> = emptyMap()
    var inferenceTimeMs: Long = 0

    fun clear() {
        totalGAGSScore = 0
        severity = ""
        totalAcneCounts = emptyMap()
    }
}
