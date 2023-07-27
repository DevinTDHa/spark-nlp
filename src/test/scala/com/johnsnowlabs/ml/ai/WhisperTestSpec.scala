package com.johnsnowlabs.ml.ai

import ai.onnxruntime.{OnnxTensor, OrtSession}
import com.johnsnowlabs.ml.onnx.OnnxWrapper
import com.johnsnowlabs.ml.onnx.OnnxWrapper.EncoderDecoderWrappers
import com.johnsnowlabs.ml.onnx.TensorResources.implicits._
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.LoadExternalModel._
import com.johnsnowlabs.nlp.annotators.audio.feature_extractor.{Preprocessor, WhisperPreprocessor}
import com.johnsnowlabs.nlp.{Annotation, AnnotationAudio, AnnotatorType}
import com.johnsnowlabs.util.TestUtils.readFile
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.scalatest.flatspec.AnyFlatSpec
import org.tensorflow.{Session, Tensor}

import scala.collection.JavaConverters._
import scala.util.{Failure, Success, Using}

class WhisperTestSpec extends AnyFlatSpec {
  implicit val formats: DefaultFormats.type = DefaultFormats

  lazy val modelPath =
    "/home/ducha/spark-nlp/dev-things/hf_exports/whisper/exported/openai/whisper-tiny.en_sepV2/"

  lazy val ppPath: String = modelPath + "assets/preprocessor_config.json"

  lazy val ppJsonString: String = readFile(ppPath)

  lazy val preprocessor: WhisperPreprocessor =
    Preprocessor.loadPreprocessorConfig(ppJsonString).asInstanceOf[WhisperPreprocessor]

  lazy val (localModelPath, _) = modelSanityCheck(modelPath)

  lazy val addedTokens: Map[String, Int] =
    try {
      parse(loadJsonStringAsset(localModelPath, "added_tokens.json")).values
        .asInstanceOf[Map[String, BigInt]]
        .map {
          case (key, value) if value.isValidInt => (key, value.toInt)
          case _ => throw new IllegalArgumentException("Could not convert BigInt to Int")
        }
    } catch {
      case _: IllegalArgumentException =>
        Map.empty
    }

  lazy val vocabMap: Map[String, Int] = {
    val vocabJsonContent = loadJsonStringAsset(localModelPath, "vocab.json")
    parse(vocabJsonContent, useBigIntForLong = true).values
      .asInstanceOf[Map[String, BigInt]]
      .map {
        case (key, value) if value.isValidInt => (key, value.toInt)
        case _ => throw new Exception("Could not convert BigInt to Int")
      }
  }

  lazy val merges: Map[(String, String), Int] = loadTextAsset(localModelPath, "merges.txt")
    .map(_.split(" "))
    .filter(w => w.length == 2)
    .map { case Array(c1, c2) => (c1, c2) }
    .zipWithIndex
    .toMap

  lazy val modelConfig: Map[String, Any] =
    parse(loadJsonStringAsset(localModelPath, "config.json")).values
      .asInstanceOf[Map[String, Any]]

  // TODO Should be moved to processor?
  lazy val generationConfig: JValue = {
    val configString = loadJsonStringAsset(localModelPath, "generation_config.json")
    parse(configString)
  }

  lazy val suppressTokenIds: Array[Int] =
    (generationConfig \ "suppress_tokens").extract[Array[Int]]

  lazy val (wrapper, signatures) =
    TensorflowWrapper.read(modelPath, zipped = false, useBundle = true, tags = Array("serve"))

  lazy val tensorResources = new TensorResources()

  private lazy val tfSession: Session =
    wrapper.getTFSessionWithSignature(savedSignatures = signatures)

  val rawFloats: Seq[Array[Float]] = (0 to 0).map { i =>
    readFile(s"src/test/resources/audio/txt/librispeech_asr_$i.txt").split("\n").map(_.toFloat)
  }

  val maxLength = 448
  val bosTokenId: Int = 50257
  val paddingTokenId: Int = 50256
  val eosTokenId: Int = 50256
  val logitsOutputSize: Int = 51864

  lazy val whisperModelTf = new Whisper(
    tensorflowWrapper = Some(wrapper),
    onnxWrappers = None,
    configProtoBytes = None,
    signatures = signatures,
    preprocessor = preprocessor,
    vocabulary = vocabMap,
    addedSpecialTokens = addedTokens,
    bosTokenId = bosTokenId,
    paddingTokenId = paddingTokenId,
    eosTokenId = eosTokenId,
    logitsSize = logitsOutputSize)

  lazy val batchFeatures: Array[Array[Array[Float]]] =
    rawFloats.map(preprocessor.extractFeatures).toArray

  lazy val encodedBatchFeatures: Tensor =
    whisperModelTf.encode(batchFeatures, Some(tfSession), None).asInstanceOf[Tensor]

  lazy val whisperModelOnnx: Whisper = {
    val onnxPath =
      "/home/ducha/Workspace/JSL/spark-nlp-dev-things/hf_exports/whisper/onnx/exported_onnx/openai/whisper-tiny.en"

    val onnxWrapperEncoder =
      OnnxWrapper.read(onnxPath, zipped = false, useBundle = true, modelName = "encoder_model")

    val onnxWrapperDecoder =
      OnnxWrapper.read(onnxPath, zipped = false, useBundle = true, modelName = "decoder_model")

    val onnxWrapperDecoderWithPast =
      OnnxWrapper.read(
        onnxPath,
        zipped = false,
        useBundle = true,
        modelName = "decoder_with_past_model")

    val onnxWrappers =
      EncoderDecoderWrappers(onnxWrapperEncoder, onnxWrapperDecoder, onnxWrapperDecoderWithPast)

    new Whisper(
      tensorflowWrapper = None,
      onnxWrappers = Some(onnxWrappers),
      configProtoBytes = None,
      signatures = None,
      preprocessor = preprocessor,
      vocabulary = vocabMap,
      addedSpecialTokens = addedTokens,
      bosTokenId = bosTokenId,
      paddingTokenId = paddingTokenId,
      eosTokenId = eosTokenId,
      logitsSize = logitsOutputSize)

  }

  private val startToken: Int = 50257

  private lazy val audioAnnotations = rawFloats.map { rawFloats =>
    AnnotationAudio(AnnotatorType.AUDIO, rawFloats, Map.empty)
  }
  behavior of "Whisper"

  it should "run model" in {

    val encoderOutputs: Tensor = encodedBatchFeatures

    val encoderOutputsOp = "decoder_encoder_outputs:0"
    val decoderInputIds =
      tensorResources.createTensor[Array[Array[Int]]](Array(Array(startToken)))
    val decoderInputIdsOp = "decoder_decoder_input_ids:0"
    val decoderPositionsIds = tensorResources.createTensor[Array[Array[Int]]](Array(Array(0)))
    val decoderPositionsIdsOp = "decoder_decoder_position_ids:0"
    val decoderOutputOp = "StatefulPartitionedCall:0"

    val runnerDecoder: Session#Runner =
      tfSession.runner

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
    val modelOutput: Array[Array[Float]] = whisperModelTf.getModelOutput(
      encodedBatchFeatures,
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
          .extractFloats(encodedBatchFeatures)
          .grouped(384)
          .toArray
          .grouped(1500)
          .toArray
          .head
      tensorResources.createTensor(Array(rawFloats, rawFloats))
    }

    val modelOutput: Array[Array[Float]] = whisperModelTf.getModelOutput(
      batchFeatureTensor,
      batchDecoderInputIds,
      maxLength = maxLength,
      tfSession)
  }

  it should "continuously get output" in {

    val decoderInputIds = Array(startToken)

    val batchDecoderInputIds: Seq[Array[Int]] = Seq(decoderInputIds)

    def callModel(in: Seq[Array[Int]]): Array[Float] = {
      val output = whisperModelTf
        .getModelOutput(encodedBatchFeatures, in, maxLength = maxLength, tfSession)

      require(output.length == 1, s"Shape of output is wrong (Batch size: ${output.length}).")
      output.head
    }

    def argmax(x: Array[Float]): Int =
      x.zipWithIndex.maxBy { case (value, _) =>
        value
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

    println(whisperModelTf.tokenDecoder.decodeTokens(sentence))
  }

  it should "generate for batch" in {

    val batchDecoderInputIds: Array[Array[Int]] = Array.fill(2, 1)(startToken)

    val generatedIds = whisperModelTf
      .generate(
        decoderEncoderStateTensors = encodedBatchFeatures,
        decoderInputIds = batchDecoderInputIds,
        maxOutputLength = maxLength,
        minOutputLength = 0,
        doSample = false,
        beamSize = 1,
        numReturnSequences = 1,
        temperature = 1.0,
        topK = 1,
        topP = 1.0,
        repetitionPenalty = 1.0,
        noRepeatNgramSize = 0,
        randomSeed = None,
        ignoreTokenIds = suppressTokenIds,
        session = tfSession)

    println(generatedIds.last.mkString("Array(", ", ", ")"))

    whisperModelTf.decode(generatedIds).foreach(println)

  }

  it should "generate multiple beams" in {
    val batchDecoderInputIds: Array[Array[Int]] = Array({
      val decoderInputIds = Array(startToken)
      decoderInputIds
    })

    val suppressTokenIds = (generationConfig \ "suppress_tokens").extract[Array[Int]]

    val generatedIds = whisperModelTf
      .generate(
        decoderEncoderStateTensors = encodedBatchFeatures,
        decoderInputIds = batchDecoderInputIds,
        maxOutputLength = maxLength,
        minOutputLength = 0,
        doSample = false,
        beamSize = 2, // 1 for greedy search
        numReturnSequences = 2, // 1 for greedy search
        temperature = 1.0,
        topK = 5, // 1 for greedy search
        topP = 0.7,
        repetitionPenalty = 1.0,
        noRepeatNgramSize = 0,
        randomSeed = None,
        ignoreTokenIds = suppressTokenIds,
        session = tfSession)

    println(generatedIds.head.mkString("Array(", ", ", ")"))

    whisperModelTf.decode(generatedIds).foreach(println)

  }

  it should "generate token ids" in {
    val generatedTokens: Seq[Annotation] = whisperModelTf.generateFromAudio(
      audios = audioAnnotations,
      batchSize = 2,
      maxOutputLength = maxLength,
      minOutputLength = 0,
      doSample = false,
      beamSize = 1,
      numReturnSequences = 1,
      temperature = 1.0,
      topK = 1,
      topP = 1.0,
      repetitionPenalty = 1.0,
      noRepeatNgramSize = 0,
      randomSeed = None,
      ignoreTokenIds = suppressTokenIds)

    println(generatedTokens.mkString(", "))
  }

  it should "work with ONNX" in {
    val onnxPath =
      "/home/ducha/Workspace/JSL/spark-nlp-dev-things/hf_exports/whisper/onnx/exported_onnx/openai/whisper-tiny.en"

    val onnxWrapperEncoder =
      OnnxWrapper.read(onnxPath, zipped = false, useBundle = true, modelName = "encoder_model")

    val onnxWrapperDecoder =
      OnnxWrapper.read(onnxPath, zipped = false, useBundle = true, modelName = "decoder_model")

    val onnxWrapperDecoderWithPast =
      OnnxWrapper.read(
        onnxPath,
        zipped = false,
        useBundle = true,
        modelName = "decoder_with_past_model")

    val onnxWrappers =
      EncoderDecoderWrappers(onnxWrapperEncoder, onnxWrapperDecoder, onnxWrapperDecoderWithPast)

    def replaceStateKeys(outputs: Map[String, OnnxTensor]): Map[String, OnnxTensor] =
      outputs.map { case (key, t) =>
        (key.replace("present", "past_key_values"), t)
      }

    val decoderWithPastOutputs = Using.Manager { use =>
      val (encoderSession, env) = onnxWrapperEncoder.getSession() match {
        case (session, env) => (use(session), env)
      }
      val decoderSession = use(onnxWrapperDecoder.getSession()._1)
      val decoderWithPastSession = use(onnxWrapperDecoderWithPast.getSession()._1)

      val encoderOutputs: OnnxTensor = {
        val encoderInputTensor = use(OnnxTensor.createTensor(env, batchFeatures))

        val inputs =
          Map("input_features" -> encoderInputTensor).asJava
        encoderSession.run(inputs).getOnnxTensor("last_hidden_state")
      }

      println(encoderOutputs)

      val logitsKey = "logits"

      val encoderStateKeys = Array(
        "present.0.encoder.key",
        "present.0.encoder.value",
        "present.1.encoder.key",
        "present.1.encoder.value",
        "present.2.encoder.key",
        "present.2.encoder.value",
        "present.3.encoder.key",
        "present.3.encoder.value")

      val decoderStateKeys = Array(
        "present.0.decoder.key",
        "present.0.decoder.value",
        "present.1.decoder.key",
        "present.1.decoder.value",
        "present.2.decoder.key",
        "present.2.decoder.value",
        "present.3.decoder.key",
        "present.3.decoder.value")

      val (logitsArray, encoderStates, decoderStates) = {

        val decoderInputIds = Array(Array(bosTokenId.toLong))
        val decoderInputTensor: OnnxTensor = use(OnnxTensor.createTensor(env, decoderInputIds))

        val decoderInputs: java.util.Map[String, OnnxTensor] =
          Map("input_ids" -> decoderInputTensor, "encoder_hidden_states" -> encoderOutputs).asJava

        val sessionOutput: OrtSession.Result =
          decoderSession.run(decoderInputs)

        val logits = sessionOutput.getOnnxTensor(logitsKey)

        val encoderStates =
          sessionOutput.getOnnxTensors(encoderStateKeys)

        val decoderStates =
          sessionOutput.getOnnxTensors(decoderStateKeys)

        val logitsArray = logits.getFloatBuffer.array().clone()

        (logitsArray, encoderStates, decoderStates)
      }

      def argmax(scores: Array[Float]): Int =
        scores.zipWithIndex.maxBy { case (score, _) =>
          score
        }._2

      val nextTokenId: Int = argmax(logitsArray)

      val nextGeneratedTokenIds: Array[Array[Long]] =
        Array(Array(nextTokenId.toLong))

      val decoderWithPastOutputs = {
        val generatedInputTensor: OnnxTensor = OnnxTensor.createTensor(env, nextGeneratedTokenIds)

        val decoderWithPastInputs: java.util.Map[String, OnnxTensor] =
          (Map("input_ids" -> generatedInputTensor) ++ replaceStateKeys(
            encoderStates) ++ replaceStateKeys(decoderStates)).asJava

        // Only requires the last generated token
        val sessionOutput = decoderWithPastSession.run(decoderWithPastInputs)

        val logits = sessionOutput.getOnnxTensor(logitsKey)

        val updatedDecoderStates =
          sessionOutput.getOnnxTensors(decoderStateKeys)

        (logits, updatedDecoderStates)
      }

      decoderWithPastOutputs
    } match {
      case Success(value) => value
      case Failure(exception) => throw exception
    }

    println(decoderWithPastOutputs)

  }

  it should "generate with Onnx" in {

    val generatedTokens: Seq[Annotation] = whisperModelOnnx.generateFromAudio(
      audios = audioAnnotations,
      batchSize = 2,
      maxOutputLength = maxLength,
      minOutputLength = 0,
      doSample = false,
      beamSize = 1,
      numReturnSequences = 1,
      temperature = 1.0,
      topK = 1,
      topP = 1.0,
      repetitionPenalty = 1.0,
      noRepeatNgramSize = 0,
      randomSeed = None,
      ignoreTokenIds = suppressTokenIds)

    println(generatedTokens.mkString(", "))

  }

}
