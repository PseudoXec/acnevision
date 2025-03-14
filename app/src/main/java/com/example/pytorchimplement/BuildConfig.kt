/**
 * Kotlin version of BuildConfig for better Kotlin compatibility.
 * This complements the Java BuildConfig class.
 */
package com.example.pytorchimplement

/**
 * Provides app build configuration in Kotlin
 */
object BuildConfig {
    /**
     * Debug flag for the application.
     * You can change this to false for production builds.
     */
    const val DEBUG = true
    
    /**
     * Application ID from your build.gradle file.
     */
    const val APPLICATION_ID = "com.example.pytorchimplement"
    
    /**
     * Application version name.
     */
    const val VERSION_NAME = "1.0"
    
    /**
     * Application version code.
     */
    const val VERSION_CODE = 1
    
    /**
     * Build type (debug or release).
     */
    const val BUILD_TYPE = "debug"
} 