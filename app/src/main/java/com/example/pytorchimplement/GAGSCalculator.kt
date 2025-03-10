package com.example.pytorchimplement

import android.graphics.RectF
import android.util.Log

/**
 * Implementation of the Global Acne Severity Grading (GAGS) methodology
 * which calculates severity based on acne lesion types and face regions.
 */
class GAGSCalculator {
    private val TAG = "GAGSCalculator"

    // Lesion Scores based on GAGS methodology
    enum class LesionType(val score: Int) {
        NO_LESION(0),
        COMEDONE(1),
        PAPULE(2),
        PUSTULE(3),
        NODULE_CYST(4)
    }

    // Face regions and their area factors based on GAGS methodology
    enum class FaceRegion(val areaFactor: Int) {
        FOREHEAD(2),
        RIGHT_CHEEK(2),
        LEFT_CHEEK(2),
        NOSE(1),
        CHIN(1)
    }

    // Approximate coordinates for face regions (normalized 0-1)
    // These boundaries need to be calibrated based on the actual face detection
    private val regionBoundaries = mapOf(
        FaceRegion.FOREHEAD to RectF(0.2f, 0.0f, 0.8f, 0.33f),
        FaceRegion.RIGHT_CHEEK to RectF(0.0f, 0.33f, 0.4f, 0.7f),
        FaceRegion.LEFT_CHEEK to RectF(0.6f, 0.33f, 1.0f, 0.7f),
        FaceRegion.NOSE to RectF(0.4f, 0.33f, 0.6f, 0.7f),
        FaceRegion.CHIN to RectF(0.3f, 0.7f, 0.7f, 1.0f)
    )

    /**
     * Maps detection class names to GAGS lesion types
     */
    private fun mapDetectionToLesionType(className: String): LesionType {
        return when (className.lowercase()) {
            "comedone" -> LesionType.COMEDONE
            "papule" -> LesionType.PAPULE
            "pustule" -> LesionType.PUSTULE
            "nodule" -> LesionType.NODULE_CYST
            else -> LesionType.NO_LESION
        }
    }

    /**
     * Determines which face region a detection belongs to based on its coordinates
     */
    private fun determineRegion(centerX: Float, centerY: Float): FaceRegion? {
        regionBoundaries.forEach { (region, bounds) ->
            if (bounds.contains(centerX, centerY)) {
                return region
            }
        }
        return null
    }

    /**
     * Calculates the GAGS score for a list of detections
     * @param detections List of detected acne lesions with bounding boxes
     * @return Calculated GAGS severity score
     */
    fun calculateGAGSScore(detections: List<ImageAnalyzer.Detection>): Int {
        // Group detections by region
        val regionDetections = mutableMapOf<FaceRegion, MutableList<ImageAnalyzer.Detection>>()

        // Initialize all regions with empty lists
        FaceRegion.entries.forEach { region ->
            regionDetections[region] = mutableListOf()
        }

        // Assign each detection to a region
        detections.forEach { detection ->
            // Get center point of the bounding box
            val centerX = detection.boundingBox.x
            val centerY = detection.boundingBox.y

            val region = determineRegion(centerX, centerY)
            if (region != null) {
                regionDetections[region]?.add(detection)
            } else {
                Log.d(TAG, "Detection at ($centerX, $centerY) couldn't be assigned to a region")
            }
        }

        // Calculate score for each region based on the most severe lesion
        var totalScore = 0

        regionDetections.forEach { (region, regionDetections) ->
            // If the region has no detections, it is considered to have NO_LESION
            // with a score of 0, but we should still log it for completeness
            if (regionDetections.isEmpty()) {
                val areaScore = region.areaFactor * LesionType.NO_LESION.score
                Log.d(
                    TAG, "Region: ${region.name}, Lesion: NO_LESION, " +
                            "Area Score: $areaScore (${region.areaFactor} × ${LesionType.NO_LESION.score})"
                )
                // The area score will be 0, so we don't need to add it to totalScore
            } else {
                // Find the most severe lesion in this region
                val mostSevereLesion = regionDetections.maxByOrNull {
                    mapDetectionToLesionType(it.className).score
                }

                if (mostSevereLesion != null) {
                    val lesionType = mapDetectionToLesionType(mostSevereLesion.className)
                    val areaScore = region.areaFactor * lesionType.score
                    totalScore += areaScore

                    Log.d(
                        TAG, "Region: ${region.name}, Lesion: ${lesionType.name}, " +
                                "Area Score: $areaScore (${region.areaFactor} × ${lesionType.score})"
                    )
                }
            }
        }

        // The maximum possible GAGS score is 40 (all regions with most severe lesions)
        // Convert to a 0-10 scale for display consistency
        val normalizedScore = (totalScore / 40.0f * 10).toInt().coerceIn(0, 10)

        Log.d(TAG, "Final GAGS Score: $totalScore/40, Normalized: $normalizedScore/10")
        return normalizedScore
    }

    /**
     * Returns the severity level description based on the GAGS score
     */
    fun getSeverityDescription(gagsScore: Int): String {
        // Convert normalized score (0-10) back to approximate GAGS score (0-40)
        val approxGagsScore = (gagsScore / 10.0f * 40).toInt()

        return when {
            approxGagsScore == 0 -> "No Acne"       // 0 on GAGS scale
            approxGagsScore <= 18 -> "Mild"        // 1-18 on GAGS scale
            approxGagsScore <= 30 -> "Moderate"    // 19-30 on GAGS scale
            approxGagsScore <= 38 -> "Severe"      // 31-38 on GAGS scale
            else -> "Very Severe"                  // >39 on GAGS scale
        }
    }

    fun severityTotalGAGSMultipleImage(totalGAGSScore: Int): String {
        return when {
            totalGAGSScore == 0 -> "No Acne"       // 0 on GAGS scale
            totalGAGSScore <= 18 -> "Mild"        // 1-18 on GAGS scale
            totalGAGSScore <= 30 -> "Moderate"    // 19-30 on GAGS scale
            totalGAGSScore <= 38 -> "Severe"      // 31-38 on GAGS scale
            else -> "Very Severe"                  // >39 on GAGS scale
        }
    }
}