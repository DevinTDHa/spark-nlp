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
    "/home/ducha/spark-nlp/dev-things/hf_exports/whisper/exported/openai/whisper-tiny.en_sepV2/"

  val rawFloats: Array[Float] =
    readFile("src/test/resources/audio/txt/librispeech_asr_0.txt").split("\n").map(_.toFloat)

  val ppPath: String = modelPath + "assets/preprocessor_config.json"

  val ppJsonString: String = readFile(ppPath)

  val preprocessor: WhisperPreprocessor =
    Preprocessor.loadPreprocessorConfig(ppJsonString).asInstanceOf[WhisperPreprocessor]

  val (localModelPath, _) = modelSanityCheck(modelPath)

  val vocabJsonMap: Map[String, Int] = {
    val vocabJsonContent = loadJsonStringAsset(localModelPath, "vocab.json")
    parse(vocabJsonContent, useBigIntForLong = true).values
      .asInstanceOf[Map[String, BigInt]]
      .map {
        case (key, value) if value.isValidInt => (key, value.toInt)
        case _ => throw new Exception("Could not convert BigInt to Int")
      }
  }

  val merges: Map[(String, String), Int] = loadTextAsset(localModelPath, "merges.txt")
    .map(_.split(" "))
    .filter(w => w.length == 2)
    .map { case Array(c1, c2) => (c1, c2) }
    .zipWithIndex
    .toMap

  // TODO
  //    val generationConfig =

  val (wrapper, signatures) =
    TensorflowWrapper.read(modelPath, zipped = false, useBundle = true, tags = Array("serve"))

  val tensorResources = new TensorResources()

  private val tfSession: Session = wrapper.getTFSessionWithSignature(savedSignatures = signatures)

  private val encodedFeatures: Tensor = {
    val features: Array[Array[Float]] = preprocessor.extractFeatures(rawFloats)

    whisperModel.encode(Array(features), tfSession)
  }

  val maxLength = 448

  lazy val whisperModel = new Whisper(
    tensorflow = wrapper,
    configProtoBytes = None,
    signatures = signatures,
    preprocessor,
    merges,
    vocabulary = vocabJsonMap)

  behavior of "Whisper"

  private val startToken: Int = whisperModel.bosTokenId
  it should "run model" in {

    //    val whisperModel = new Whisper(
    //      tensorflow = wrapper,
    //      configProtoBytes = None,
    //      signatures = signatures,
    //      preprocessor,
    //      merges,
    //      vocabulary = vocabJsonMap)

    val encoderOutputs: Tensor = encodedFeatures

    val encoderOutputsOp = "decoder_encoder_outputs:0"
    val decoderInputIds =
      tensorResources.createTensor[Array[Array[Int]]](Array(Array(startToken)))
    val decoderInputIdsOp = "decoder_decoder_input_ids:0"
    val decoderPositionsIds = tensorResources.createTensor[Array[Array[Int]]](Array(Array(0)))
    val decoderPositionsIdsOp = "decoder_decoder_position_ids:0"
    val decoderOutputOp = "StatefulPartitionedCall:0"

    val runnerDecoder: Session#Runner =
      tfSession.runner

    // TODO: Only produces the right output with a new runner?
    val decoderOut = runnerDecoder
      .feed(decoderInputIdsOp, decoderInputIds)
      .feed(encoderOutputsOp, encoderOutputs)
      .feed(decoderPositionsIdsOp, decoderPositionsIds)
      .fetch(decoderOutputOp)
      .run()
      .asScala

    println(decoderOut)

  }

  it should "getModelOutput" in {

    val decoderInputIds = Array(startToken)

    val batchDecoderInputIds = Seq(decoderInputIds)
    val modelOutput: Array[Array[Float]] = whisperModel.getModelOutput(
      encodedFeatures,
      batchDecoderInputIds,
      maxLength = maxLength,
      tfSession)

    println(modelOutput.map(_.mkString("(", ", ", ")")).mkString("(", ", ", ")"))
  }

  it should "getModelOutput for batch > 1" in {

    val decoderInputIds = Array(startToken)

    val batchDecoderInputIds = Seq(decoderInputIds, decoderInputIds)

    val batchFeatureTensor: Tensor = {
      val rawFloats: Array[Array[Float]] =
        TensorResources
          .extractFloats(encodedFeatures)
          .grouped(384)
          .toArray
          .grouped(1500)
          .toArray
          .head
      tensorResources.createTensor(Array(rawFloats, rawFloats))
    }

    val modelOutput: Array[Array[Float]] = whisperModel.getModelOutput(
      batchFeatureTensor,
      batchDecoderInputIds,
      maxLength = maxLength,
      tfSession)
  }

  it should "continuously get output" in {

    val decoderInputIds = Array(startToken)

    val batchDecoderInputIds: Seq[Array[Int]] = Seq(decoderInputIds)

    def callModel(in: Seq[Array[Int]]): Array[Float] = {
      val output = whisperModel
        .getModelOutput(encodedFeatures, in, maxLength = maxLength, tfSession)

      require(output.length == 1, s"Shape of output is wrong (Batch size: ${output.length}).")
      output.head
    }

    def argmax(x: Array[Float]): Int =
      x.zipWithIndex.maxBy { case (out, _) =>
        out
      }._2

    var nextDecoderInputIds: Seq[Array[Int]] = batchDecoderInputIds

    (0 to 30).foreach { _ =>
      val currentOutput = callModel(nextDecoderInputIds)
      val nextTokenId = argmax(currentOutput)

      val appendedInputBatch = Seq(nextDecoderInputIds.head ++ Array(nextTokenId))
      nextDecoderInputIds = appendedInputBatch
    }

    println(nextDecoderInputIds)
    val sentence: Array[Int] =
      nextDecoderInputIds.head.slice(2, nextDecoderInputIds.head.length)

    println(whisperModel.bpeTokenizer.decodeTokens(sentence))
  }

  it should "generate" in {
    val batchDecoderInputIds: Array[Array[Int]] = Array({
      val decoderInputIds = Array(startToken)
      decoderInputIds
    })

    val generatedIds = whisperModel.generate(
      decoderEncoderStateTensors = encodedFeatures,
      decoderInputIds = batchDecoderInputIds,
      maxOutputLength = maxLength,
      minOutputLength = 0,
      doSample = false,
      beamSize = 1,
      numReturnSequences = 10,
      temperature = 0.0,
      topK = 1,
      topP = 1.0,
      repetitionPenalty = 0.0,
      noRepeatNgramSize = 0,
      randomSeed = None,
      ignoreTokenIds = Array.empty[Int],
      session = tfSession)

    println(generatedIds.mkString("Array(", ", ", ")"))

  }

}
