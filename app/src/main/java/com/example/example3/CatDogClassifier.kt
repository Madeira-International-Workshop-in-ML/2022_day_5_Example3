package com.example.example3

import android.content.Context
import android.content.res.AssetManager
import android.graphics.*
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class CatDogClassifier(private val context: Context) {

    private val modelPath: String = "catsdogs.tflite"
    private lateinit var interpreter: Interpreter

    private val INPUT_SIZE = 224
    private val PIXEL_SIZE = 3
    private val OUTPUT_CLASSES_COUNT = 2
    private val THRESHOLD = 0.95

    /**
     * Loads the SavedModel
     */
    private fun loadModelFile(assetManager: AssetManager): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Initializes the SavedModel
     */
    fun initialize() {
        val compatList = CompatibilityList()

        // Let's use the GPU if its available
        val options = Interpreter.Options().apply {
            if (compatList.isDelegateSupportedOnThisDevice) {
                // if the device has a supported GPU, add the GPU delegate
                val delegateOptions = compatList.bestOptionsForThisDevice
                this.addDelegate(GpuDelegate(delegateOptions))
            } else {
                // if the GPU is not supported, run on 4 threads
                this.setNumThreads(4)
            }
        }
        interpreter = Interpreter(loadModelFile(context.assets), options)
    }


    /**
     * Returns the result after running the recognition with the help of interpreter
     * on the passed bitmap
     */
    fun classify(image: ImageProxy, rotation: Int): String {
        // Converts the image to bitmap and rotate
        val bitmap = rotateBitmap(convertImageProxyToBitmap(image), rotation)

        val resizedImage = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val byteBuffer = convertBitmapToByteBuffer(resizedImage)

        // Define an array to store the model output.
        val output = Array(1) { FloatArray(OUTPUT_CLASSES_COUNT) }

        // Run inference with the input data.
        interpreter.run(byteBuffer, output)

        return getSortedResult(output)

    }

    /**
     * This function serves to rotate the bitmap given an angle of rotation
     */
    private fun rotateBitmap(img: Bitmap, degree: Int): Bitmap {
        if (degree == 0) return img
        val matrix = Matrix()
        matrix.postRotate(degree.toFloat())
        return Bitmap.createBitmap(img, 0, 0, img.width, img.height, matrix, true)
    }

    /**
     * This function serves to convert an [ImageProxy] to bitmap
     */
    private fun convertImageProxyToBitmap(image: ImageProxy): Bitmap {
        val yBuffer = image.planes[0].buffer // Y
        val uBuffer = image.planes[1].buffer // U
        val vBuffer = image.planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    /**
     * Converts the bitmap to byte buffer
     */

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (pixelValue in pixels) {
            val r = (pixelValue shr 16 and 0xFF)
            val g = (pixelValue shr 8 and 0xFF)
            val b = (pixelValue and 0xFF)

            // Convert RGB to grayscale and normalize pixel value to [0..1].
            val normalizedPixelValue = (r + g + b) / 3.0f / 255.0f
            byteBuffer.putFloat(normalizedPixelValue)
        }

        return byteBuffer

    }

    /**
     * Sorts the results and outputs the formatted text
     */
    private fun getSortedResult(output: Array<FloatArray>): String {

        val result = output[0]
        val maxIndex = result.indices.maxByOrNull { result[it] } ?: -1
        val confidence = result[maxIndex]
        return if (confidence > THRESHOLD) {
            val animal = when (maxIndex) {
                0 -> "Cat"
                1 -> "Dog"
                else -> ""
            }
            "Prediction Result: %s\nConfidence: %.2f%%".format(animal, confidence * 100)
        } else {
            ""
        }
    }

}