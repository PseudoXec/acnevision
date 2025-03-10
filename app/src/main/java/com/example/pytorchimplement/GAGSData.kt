package com.example.pytorchimplement

object GAGSData {
    var totalGAGSScore: Int = 0
    var severity: String = ""
    var totalAcneCounts: Map<String, Int> = emptyMap()

    fun clear() {
        totalGAGSScore = 0
        severity = ""
        totalAcneCounts = emptyMap()
    }
}
