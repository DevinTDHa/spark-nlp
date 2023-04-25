package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.audio.feature_extractor.Preprocessor
import org.scalatest.flatspec.AnyFlatSpec

import scala.io.Source
import scala.util.Using

class TensorflowWhisperTest extends AnyFlatSpec {

  val modelPath = "/home/ducha/Workspace/python/transformers/whisper-tiny-en/saved_model/1"
  val (tf, signatures) =
    TensorflowWrapper.read(modelPath, zipped = false, useBundle = true)

  val configProtoBytes = None
  val model =
    new TensorflowWhisper(tf, configProtoBytes = configProtoBytes, signatures = signatures)

  val expected_decode =
    " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."

  behavior of "TensorflowWhisperTest"

  val rawFloats: Array[String] = {
    val path = "src/test/resources/audio/csv/audio_floats.csv"
    Using(Source.fromFile(path)) {
      _.getLines().map(_.split(",").head).toArray
    }.getOrElse {
      throw new Exception("Couldn't read file.")
    }
  }

  it should "tag" in {}

}
