package com.johnsnowlabs.nlp.annotators.audio.feature_extractor

import breeze.linalg.csvread
import com.johnsnowlabs.nlp.annotators.audio.feature_extractor.AudioUtils.matrixToFloatArray
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.TestUtils.{readFile, tolerantFloatEq}
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File

class WhisperPreprocessorTest extends AnyFlatSpec {

  val modelPath =
    "/home/ducha/Workspace/JSL/spark-nlp-dev-things/hf_exports/whisper/exported/openai/whisper-tiny.en/saved_model/1/"

  val rawFloats: Array[Float] =
    readFile("src/test/resources/audio/txt/librispeech_asr_0.txt").split("\n").map(_.toFloat)

  val ppPath: String = modelPath + "assets/preprocessor_config.json"

  val ppJsonString: String = readFile(ppPath)

  val preprocessor: WhisperPreprocessor =
    Preprocessor.loadPreprocessorConfig(ppJsonString).asInstanceOf[WhisperPreprocessor]

  behavior of "AudioPreprocessor"

  it should "pad" in {
    val padded = Preprocessor.pad(
      rawFloats,
      preprocessor.padding_value,
      preprocessor.n_samples,
      preprocessor.padding_side)

    assert(padded.length == preprocessor.n_samples)
  }

  it should "extract features" taggedAs FastTest in {
    val extractedFeatures: Array[Array[Float]] =
      preprocessor.extractFeatures(rawFloats)

    val expectedFeatures: Array[Array[Float]] = matrixToFloatArray(
      csvread(new File("src/test/resources/audio/txt/librispeech_asr_0_features.csv")))

    assert(extractedFeatures.length == expectedFeatures.length)
    assert(extractedFeatures(0).length == expectedFeatures(0).length)

    expectedFeatures.indices.foreach { row =>
      expectedFeatures(0).indices.foreach { col =>
        assert(extractedFeatures(row)(col) === expectedFeatures(row)(col))
      }
    }
  }

}
