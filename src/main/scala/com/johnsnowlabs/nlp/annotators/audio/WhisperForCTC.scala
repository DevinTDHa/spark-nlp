/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.audio

import com.johnsnowlabs.ml.ai.Whisper
import com.johnsnowlabs.ml.onnx.OnnxWrapper.EncoderDecoderWrappers
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.tensorflow.{
  ReadTensorflowModel,
  TensorflowWrapper,
  WriteTensorflowModel
}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.audio.feature_extractor.{Preprocessor, WhisperPreprocessor}
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.json4s._
import org.json4s.jackson.JsonMethods._

/** Whisper Model with a language modeling head on top for Connectionist Temporal Classification
  * (CTC).
  *
  * TODO: Description
  *
  * The annotator takes audio files and transcribes it as text. The audio needs to be provided
  * pre-processed an array of floats.
  *
  * Note that this annotator is currently not supported on Apple Silicon processors such as the
  * M1/M2 (Apple Silicon). This is due to the processor not supporting instructions for XLA.
  *
  * TODO Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val speechToText = WhisperForCTC.pretrained()
  *   .setInputCols("audio_assembler")
  *   .setOutputCol("text")
  * }}}
  * The default model is `"asr_whisper_large_ls960"`, if no name is provided.
  *
  * For available pretrained models please see the [[https://sparknlp.org/models Models Hub]].
  *
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/audio/WhisperForCTCTestSpec.scala WhisperForCTCTestSpec]].
  *
  * '''References:'''
  *
  * TODO
  *
  * [[https://arxiv.org/abs/2106.07447 whisper: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units]]
  *
  * '''Paper Abstract:'''
  *
  * ''TODO''
  *
  * ==Example==
  *
  * TODO
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotators._
  * import com.johnsnowlabs.nlp.annotators.audio.WhisperForCTC
  * import org.apache.spark.ml.Pipeline
  *
  * val audioAssembler: AudioAssembler = new AudioAssembler()
  *   .setInputCol("audio_content")
  *   .setOutputCol("audio_assembler")
  *
  * val speechToText: WhisperForCTC = WhisperForCTC
  *   .pretrained()
  *   .setInputCols("audio_assembler")
  *   .setOutputCol("text")
  *
  * val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))
  *
  * val bufferedSource =
  *   scala.io.Source.fromFile("src/test/resources/audio/csv/audio_floats.csv")
  *
  * val rawFloats = bufferedSource
  *   .getLines()
  *   .map(_.split(",").head.trim.toFloat)
  *   .toArray
  * bufferedSource.close
  *
  * val processedAudioFloats = Seq(rawFloats).toDF("audio_content")
  *
  * val result = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)
  * result.select("text.result").show(truncate = false)
  * +------------------------------------------------------------------------------------------+
  * |result                                                                                    |
  * +------------------------------------------------------------------------------------------+
  * |[MISTER QUILTER IS THE APOSTLE OF THE MIDLE CLASES AND WE ARE GLAD TO WELCOME HIS GOSPEL ]|
  * +------------------------------------------------------------------------------------------+
  * }}}
  *
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class WhisperForCTC(override val uid: String)
    extends AnnotatorModel[Wav2Vec2ForCTC]
    with HasBatchedAnnotateAudio[Wav2Vec2ForCTC]
    with HasAudioFeatureProperties
    with WriteTensorflowModel
    with WriteOnnxModel
    with HasEngine
    with HasGeneratorProperties
    with HasProtectedParams {

  override val outputAnnotatorType: AnnotatorType = AnnotatorType.DOCUMENT

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(AnnotatorType.AUDIO)

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("WhisperForCTC"))

  /** Optional language to set for the transcription. The imported model needs to support multiple
    * languages. TODO: Implement and add flag to know if it is multi-lang.
    * @group param
    */
  val language =
    new Param[String](
      this,
      "language",
      "Optional parameter to set the language for the transcription.")

  /** @group setParam */
  def setLanguage(value: String): this.type = {
    set(language, value)
    this
  }

  /** @group getParam */
  def getLanguage: Option[String] = get(this.language)

  /** Sets the task for the audio. Either `translate` or `transcribe`.
    *
    * @group setParam
    */
  override def setTask(value: String): this.type = {
    require(
      value == "translate" || value == "transcribe",
      "Task should be either 'translate' or 'transcribe'")
    set(task, value)
    this
  }

  /** It contains TF model signatures for the laded saved model
    *
    * @group param
    */
  val signatures: MapFeature[AnnotatorType, AnnotatorType] =
    new MapFeature[String, String](model = this, name = "signatures").setProtected()

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    set(signatures, value)
    this
  }

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

  // TODO: Consolidate preprocessor into Feature.

  /** Hop Length for the window of the preprocessor
    * @group param
    */
  val hopLength: ProtectedParam[Int] =
    new IntParam(this, "hopLength", "Hop Length for the window of the preprocessor")
      .setProtected()

  /** @group setParam */
  def setHopLength(value: Int): this.type = set(hopLength, value)

  /** @group getParam */
  def getHopLength: Int = $(hopLength)

  /** Number of frequencies to extract for FFT.
    *
    * @group param
    */
  val nFFT: ProtectedParam[Int] =
    new IntParam(this, "nFFT", "Number of frequencies to extract for FFT.").setProtected()

  /** @group setParam */
  def setNFFT(value: Int): this.type = set(nFFT, value)

  /** @group getParam */
  def getNFFT: Int = $(nFFT)

  /** Maximum number of samples to take from the audio.
    *
    * @group param
    */
  val nSamples: ProtectedParam[Int] =
    new IntParam(this, "nSamples", "Maximum number of samples to take from the audio.")
      .setProtected()

  /** @group setParam */
  def setNSamples(value: Int): this.type = set(nSamples, value)

  /** @group getParam */
  def getNSamples: Int = $(nSamples)

  /** Maximum number of frames to process (during preprocessing). TODO: Needed?
    *
    * @group param
    */
  val nMaxFrames: ProtectedParam[Int] = new IntParam(
    this,
    "nMaxFrames",
    "Maximum number of frames to process (during preprocessing).").setProtected()

  /** @group setParam */
  def setNMaxFrames(value: Int): this.type = set(nMaxFrames, value)

  /** @group getParam */
  def getNMaxFrames: Int = $(nMaxFrames)

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group setParam
    */
  def setConfigProtoBytes(bytes: Array[Int]): this.type =
    set(this.configProtoBytes, bytes)

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group getParam
    */
  def getConfigProtoBytes: Option[Array[Byte]] =
    get(this.configProtoBytes).map(_.map(_.toByte))

  /** Vocabulary used to encode the words to ids */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  def getVocabulary: Map[String, Int] = $$(vocabulary)

  val addedSpecialTokens: MapFeature[String, Int] =
    new MapFeature(this, "addedSpecialTokens").setProtected()

  def setAddedSpecialTokens(value: Map[String, Int]): this.type = set(addedSpecialTokens, value)

  val generationTokens: StructFeature[GenerationTokens] =
    new StructFeature(this, "generationConfig").setProtected()

  def setGenerationConfig(value: GenerationTokens): this.type = set(generationTokens, value)
  def getGenerationTokens: GenerationTokens = $$(generationTokens)

  setDefault(
    minOutputLength -> 0,
    maxOutputLength -> 448,
    doSample -> false,
    temperature -> 1.0,
    topK -> 1,
    topP -> 1.0,
    repetitionPenalty -> 1.0,
    noRepeatNgramSize -> 0,
    ignoreTokenIds -> Array(),
    batchSize -> 2,
    beamSize -> 1,
    nReturnSequences -> 1)

  private var _model: Option[Broadcast[Whisper]] = None

  /** @group getParam */
  def getModelIfNotSet: Whisper = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrappers: Option[EncoderDecoderWrappers]): this.type = {
    if (_model.isEmpty) {
      // TODO: Use StructFeature?
      val preprocessor =
        new WhisperPreprocessor(
          getFeatureSize,
          getHopLength,
          getNFFT,
          getNSamples,
          getNMaxFrames,
          getPaddingSide,
          getPaddingValue,
          getSamplingRate)

      val GenerationTokens(bosTokenId, padTokenId, eosTokenId, vocabSize) = getGenerationTokens

      _model = Some(
        spark.sparkContext.broadcast(
          new Whisper(
            tensorflowWrapper,
            onnxWrappers,
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures,
            preprocessor = preprocessor,
            vocabulary = getVocabulary,
            addedSpecialTokens = $$(addedSpecialTokens),
            bosTokenId = bosTokenId,
            paddingTokenId = padTokenId,
            eosTokenId = eosTokenId,
            logitsSize = vocabSize)))
    }
    this
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    val suffix = "_whisper_ctc"

    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          suffix,
          WhisperForCTC.tfFile,
          configProtoBytes = getConfigProtoBytes,
          savedSignatures = getSignatures)
      case ONNX.name =>
        val wrappers = getModelIfNotSet.onnxWrappers.get
        ???
//        writeOnnxModels(
//          path,
//          spark,
//          Seq(wrappers.encoder, wrappers.decoder, wrappers.decoderWithPast),
//          suffix,
//          WhisperForCTC.onnxFile)
    }

  }

  /** Takes audio annotations and produces transcribed document annotations.
    *
    * @param batchedAnnotations
    *   Audio annotations in batches
    * @return
    *   Transcribed audio as DOCUMENT type annotation
    */
  override def batchAnnotate(
      batchedAnnotations: Seq[Array[AnnotationAudio]]): Seq[Seq[Annotation]] = {
    batchedAnnotations.map { audioAnnotations =>
      getModelIfNotSet.generateFromAudio(
        audioAnnotations,
        getBatchSize,
        getMaxOutputLength,
        getMinOutputLength,
        getDoSample,
        getBeamSize,
        getNReturnSequences,
        getTemperature,
        getTopK,
        getTopP,
        getRepetitionPenalty,
        getNoRepeatNgramSize,
        getRandomSeed,
        getIgnoreTokenIds)
    }
  }

}

trait ReadablePretrainedWhisperForCTCModel
    extends ParamsAndFeaturesReadable[WhisperForCTC]
    with HasPretrained[WhisperForCTC] {
  override val defaultModelName: Some[String] = Some("") // TODO

  /** Java compliant-overrides */
  override def pretrained(): WhisperForCTC = super.pretrained()

  override def pretrained(name: String): WhisperForCTC = super.pretrained(name)

  override def pretrained(name: String, lang: String): WhisperForCTC =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): WhisperForCTC =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadWhisperForCTCDLModel extends ReadTensorflowModel with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[WhisperForCTC] =>

  override val tfFile: String = "whisper_ctc_tensorflow"
  override val onnxFile: String = "whisper_ctc_onnx"

  def readModel(instance: WhisperForCTC, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case TensorFlow.name =>
        val tfWrapper = readTensorflowModel(path, spark, "_whisper_tf")
        instance.setModelIfNotSet(spark, Some(tfWrapper), None)

      case ONNX.name =>
        ???
//        val onnxWrapper =
//          readOnnxModel(path, spark, "_whisper_onnx", zipped = true, useBundle = false, None)
//        instance.setModelIfNotSet(spark, None, Some(onnxWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): WhisperForCTC = {
    implicit val formats: DefaultFormats.type = DefaultFormats // for json4s

    val (localModelPath, detectedEngine) =
      modelSanityCheck(modelPath, isEncoderDecoder = true, withPast = true)

    val ppJsonString: String = loadJsonStringAsset(localModelPath, "preprocessor_config.json")

    val preprocessor: WhisperPreprocessor =
      Preprocessor.loadPreprocessorConfig(ppJsonString).asInstanceOf[WhisperPreprocessor]

    val addedTokens: Map[String, Int] =
      try {
        parse(loadJsonStringAsset(localModelPath, "added_tokens.json")).values
          .asInstanceOf[Map[String, BigInt]]
          .map {
            case (key, value) if value.isValidInt => (key, value.toInt)
            case _ =>
              throw new IllegalArgumentException(
                "Could not convert BigInt to Int while parsing added_tokens.json")
          }
      } catch {
        case _: IllegalArgumentException =>
          Map.empty
      }

    val vocabMap: Map[String, Int] = {
      val vocabJsonContent = loadJsonStringAsset(localModelPath, "vocab.json")
      parse(vocabJsonContent, useBigIntForLong = true).values
        .asInstanceOf[Map[String, BigInt]]
        .map {
          case (key, value) if value.isValidInt => (key, value.toInt)
          case _ =>
            throw new IllegalArgumentException(
              "Could not convert BigInt to Int while parsing vocab.json")
        }
    }

    val modelConfig: JValue =
      parse(loadJsonStringAsset(localModelPath, "config.json"))

    val suppressTokenIds: Array[Int] =
      (modelConfig \ "suppress_tokens").extract[Array[Int]]

    val maxOutputLength = (modelConfig \ "max_length").extract[Int]
    val bosTokenId = (modelConfig \ "bos_token_id").extract[Int]
    val eosTokenId = (modelConfig \ "eos_token_id").extract[Int]
    val padTokenId = (modelConfig \ "pad_token_id").extract[Int]
    val vocabSize = (modelConfig \ "vocab_size").extract[Int]

    val annotatorModel = new WhisperForCTC()
      .setHopLength(preprocessor.hop_length)
      .setNFFT(preprocessor.n_fft)
      .setNSamples(preprocessor.n_samples)
      .setNMaxFrames(preprocessor.nb_max_frames)
      .setVocabulary(vocabMap)
      .setMaxOutputLength(maxOutputLength)
      .setIgnoreTokenIds(suppressTokenIds)
      .setDoNormalize(preprocessor.do_normalize)
      .setReturnAttentionMask(preprocessor.return_attention_mask)
      .setPaddingSide(preprocessor.padding_side)
      .setPaddingValue(preprocessor.padding_value)
      .setFeatureSize(preprocessor.feature_size)
      .setSamplingRate(preprocessor.sampling_rate)
      .setAddedSpecialTokens(addedTokens)
      .setGenerationConfig(GenerationTokens(bosTokenId, padTokenId, eosTokenId, vocabSize))

    //      .setTask() // TODO: default <startoftranscription>?
    //      .setMinOutputLength(0) // TODO: default 0?
    //      .setDoSample() // TODO: Default false?
    //      .setTemperature() // Default
    //      .setTopK() // Default
    //      .setTopP() // Default
    //      .setRepetitionPenalty() // Default
    //      .setNoRepeatNgramSize() // Default
    //      .setRandomSeed() // Default
    //      .setBeamSize() // Default
    //      .setNReturnSequences() // Default

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        val (tfWrapper, signatures) =
          TensorflowWrapper.read(localModelPath, zipped = false, useBundle = true)

        val _signatures = signatures match {
          case Some(s) => s
          case None => throw new Exception("Cannot load signature definitions from model!")
        }

        /** the order of setSignatures is important if we use getSignatures inside
          * setModelIfNotSet
          */
        annotatorModel
          .setSignatures(_signatures)
          .setModelIfNotSet(spark, Some(tfWrapper), None)

      case ONNX.name =>
        val onnxWrapperEncoder =
          OnnxWrapper.read(
            modelPath,
            zipped = false,
            useBundle = true,
            modelName = "encoder_model")

        val onnxWrapperDecoder =
          OnnxWrapper.read(
            modelPath,
            zipped = false,
            useBundle = true,
            modelName = "decoder_model")

        val onnxWrapperDecoderWithPast =
          OnnxWrapper.read(
            modelPath,
            zipped = false,
            useBundle = true,
            modelName = "decoder_with_past_model")

        val onnxWrappers = EncoderDecoderWrappers(
          onnxWrapperEncoder,
          onnxWrapperDecoder,
          onnxWrapperDecoderWithPast)

        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrappers))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[WhisperForCTC]]. Please refer to that class for the
  * documentation.
  */
object WhisperForCTC extends ReadablePretrainedWhisperForCTCModel with ReadWhisperForCTCDLModel
