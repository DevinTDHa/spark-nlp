package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.LoadExternalModel._
import com.johnsnowlabs.nlp.annotators.audio.feature_extractor.{Preprocessor, WhisperPreprocessor}
import com.johnsnowlabs.util.TestUtils.readFile
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.scalatest.flatspec.AnyFlatSpec
import org.tensorflow.{Session, Tensor}

import scala.collection.JavaConverters._

class WhisperForCTCTest extends AnyFlatSpec {
  val modelPath =
    "/home/ducha/Workspace/JSL/spark-nlp-dev-things/hf_exports/whisper/exported/openai/whisper-tiny.en_sep/"

  val rawFloats: Array[Float] =
    readFile("src/test/resources/audio/txt/librispeech_asr_0.txt").split("\n").map(_.toFloat)

  val ppPath: String = modelPath + "assets/preprocessor_config.json"

  val ppJsonString: String = readFile(ppPath)

  val preprocessor: WhisperPreprocessor =
    Preprocessor.loadPreprocessorConfig(ppJsonString).asInstanceOf[WhisperPreprocessor]

  behavior of "Whisper"

  it should "run model" in {
    val (localModelPath, _) = modelSanityCheck(modelPath)

    val vocabJsonMap = {
      val vocabJsonContent = loadJsonStringAsset(localModelPath, "vocab.json")
      parse(vocabJsonContent, useBigIntForLong = true).values
        .asInstanceOf[Map[String, BigInt]]
        .map {
          case (key, value) if value.isValidInt => (key, value.toInt)
          case _ => throw new Exception("Could not convert BigInt to Int")
        }
    }

    val merges = loadTextAsset(localModelPath, "merges.txt")
      .map(_.split(" "))
      .filter(w => w.length == 2)
      .map { case Array(c1, c2) => (c1, c2) }
      .zipWithIndex
      .toMap

    // TODO
    //    val generationConfig =

    val (wrapper, signatures) =
      TensorflowWrapper.read(modelPath, zipped = false, useBundle = true, tags = Array("serve"))

    print(signatures.get)

    //    val whisperModel = new Whisper(
    //      tensorflow = wrapper,
    //      configProtoBytes = None,
    //      signatures = signatures,
    //      preprocessor,
    //      merges,
    //      vocabulary = vocabJsonMap)

    val features: Array[Array[Float]] = preprocessor.extractFeatures(rawFloats)

    val runner: Session#Runner =
      wrapper.getTFSessionWithSignature(savedSignatures = signatures).runner

    val tensorEncoder = new TensorResources()

    val featuresTensors = tensorEncoder.createTensor[Array[Array[Array[Float]]]](Array(features))

    val encoderInputOp = "encoder_input_features:0"
    val encoderOutputOp = "StatefulPartitionedCall_1:0"

    val encoderOutputs: Tensor = runner
      .feed(encoderInputOp, featuresTensors)
      .fetch(encoderOutputOp)
      .run()
      .asScala
      .head

    val encoderOutputsOp = "decoder_encoder_outputs:0"
    val decoderInputIds = tensorEncoder.createTensor[Array[Array[Int]]](Array(Array(50257)))
    val decoderInputIdsOp = "decoder_decoder_input_ids:0"
    val decoderPositionsIds = tensorEncoder.createTensor[Array[Array[Int]]](Array(Array(0)))
    val decoderPositionsIdsOp = "decoder_decoder_position_ids:0"
    val decoderOutputOp = "StatefulPartitionedCall:0"

    val runnerDecoder: Session#Runner =
      wrapper.getTFSessionWithSignature(savedSignatures = signatures).runner

    // TODO: Only produces the right output with a new runner?
    val decoderOut = runnerDecoder
      .feed(decoderInputIdsOp, decoderInputIds)
      .feed(encoderOutputsOp, encoderOutputs)
      .feed(decoderPositionsIdsOp, decoderPositionsIds)
      .fetch(decoderOutputOp)
      .run()
      .asScala

    print(decoderOut)

  }
}
