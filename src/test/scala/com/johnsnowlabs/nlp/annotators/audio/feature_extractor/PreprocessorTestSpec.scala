package com.johnsnowlabs.nlp.annotators.audio.feature_extractor

import com.johnsnowlabs.util.TestUtils.{readFile, tolerantFloatEq}
import org.scalatest.flatspec.AnyFlatSpec
class PreprocessorTestSpec extends AnyFlatSpec {

  val modelPath =
    "/home/ducha/Workspace/JSL/spark-nlp-dev-things/hf_exports/whisper/exported/openai/whisper-tiny.en/saved_model/1/"

  val rawFloats: Array[Float] =
    readFile("src/test/resources/audio/txt/librispeech_asr_0.txt").split("\n").map(_.toFloat)

  behavior of "AudioPreprocessor"

  it should "load whisper preprocessors" in {
    val ppPath = modelPath + "assets/preprocessor_config.json"

    val ppJsonString = readFile(ppPath)

    val preprocessor =
      Preprocessor.loadPreprocessorConfig(ppJsonString)

    print(preprocessor)

    //    model.predict(Array(rawFloats), 0, preprocessor)
  }

  it should "normalize" in {
    val normalized = Preprocessor.normalize(rawFloats)
    assert(Preprocessor.mean(normalized) === 0f)
    assert(Preprocessor.variance(normalized).toFloat === 1f)
  }

  it should "pad" in {
    val ppPath = modelPath + "assets/preprocessor_config.json"

    val ppJsonString = readFile(ppPath)

    val preprocessor: WhisperPreprocessor =
      Preprocessor.loadPreprocessorConfig(ppJsonString).asInstanceOf[WhisperPreprocessor]

    val padded = Preprocessor.pad(
      rawFloats,
      preprocessor.padding_value,
      preprocessor.n_samples,
      preprocessor.padding_side)

    assert(padded.length == preprocessor.n_samples)
  }

}
