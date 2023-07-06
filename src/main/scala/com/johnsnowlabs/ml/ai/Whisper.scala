/*
 * Copyright 2017 - 2023  John Snow Labs
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.ml.ai.util.Generation.Generate
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.nlp.annotators.audio.feature_extractor.WhisperPreprocessor
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{SpecialTokens, WhisperTokenDecoder}
import com.johnsnowlabs.nlp.{Annotation, AnnotationAudio, AnnotatorType}
import org.tensorflow.{Session, Tensor}

import scala.collection.JavaConverters._

/** Class representing a Whisper model. Used to call the model and generate tokens.
  *
  * @param tensorflowWrapper
  *   Tensorflow Wrapper
  * @param configProtoBytes
  *   Config ProtoBytes
  * @param signatures
  *   Signatures of the model
  * @param preprocessor
  *   Whisper preprocessor to extract features
  * @param vocabulary
  *   Vocabulary for decoding
  * @param addedSpecialTokens
  *   Added special tokens
  */
private[johnsnowlabs] class Whisper(
    val tensorflowWrapper: TensorflowWrapper,
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None,
    preprocessor: WhisperPreprocessor,
    vocabulary: Map[String, Int],
    addedSpecialTokens: Map[String, Int] = Map.empty)
    extends Serializable
    with Generate {

  // TODO: Keep this static?
  val bosTokenId: Int = 50257
  val paddingTokenId: Int = 50256
  val eosTokenId: Int = 50256
  val logitsOutputSize: Int = 51864

  private val vocabWithAddedTokens: Map[String, Int] = vocabulary ++ addedSpecialTokens

  private val tokenizerSpecialTokens: SpecialTokens =
    SpecialTokens(
      vocabWithAddedTokens,
      startTokenId = bosTokenId,
      endTokenId = eosTokenId,
      unkTokenId = eosTokenId,
      maskTokenId = eosTokenId,
      padTokenId = eosTokenId,
      additionalTokenIds = addedSpecialTokens.values.toArray)

  val tokenDecoder: WhisperTokenDecoder =
    new WhisperTokenDecoder(vocabWithAddedTokens, tokenizerSpecialTokens)

  private val _tfWhisperSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  var tensorResources = new TensorResources()

  private val encoderInputOp: String = _tfWhisperSignatures.getOrElse(
    ModelSignatureConstants.EncoderInputIds.key,
    ModelSignatureConstants.EncoderInputIds.value)
  private val encoderOutputOp: String = _tfWhisperSignatures.getOrElse(
    ModelSignatureConstants.EncoderOutput.key,
    ModelSignatureConstants.EncoderOutput.value)
  private val decoderEncoderOutputsOp: String = _tfWhisperSignatures.getOrElse(
    ModelSignatureConstants.DecoderEncoderInputIds.key,
    ModelSignatureConstants.DecoderEncoderInputIds.value)
  private val decoderInputIdsOp: String = _tfWhisperSignatures.getOrElse(
    ModelSignatureConstants.DecoderInputIds.key,
    ModelSignatureConstants.DecoderInputIds.value)
  private val decoderOutputOp: String = _tfWhisperSignatures.getOrElse(
    ModelSignatureConstants.LogitsOutput.key,
    ModelSignatureConstants.LogitsOutput.value)

  /** @param audios
    *   Sequence of audio floats
    * @param batchSize
    *   Batch size
    * @param minOutputLength
    *   Minimum length of output
    * @param maxOutputLength
    *   Maximum length of output
    * @param doSample
    *   Whether to sample or not
    * @param temperature
    *   Temperature for sampling
    * @param topK
    *   Top K for sampling
    * @param topP
    *   Top P for sampling
    * @param repetitionPenalty
    *   Repetition penalty for sampling
    * @param noRepeatNgramSize
    *   No repeat ngram size for sampling
    * @param randomSeed
    *   Random seed
    * @param ignoreTokenIds
    *   Ignore token ids
    * @param beamSize
    *   Beam size
    * @return
    */
  def generateFromAudio(
      audios: Seq[AnnotationAudio],
      batchSize: Int,
      maxOutputLength: Int,
      minOutputLength: Int,
      doSample: Boolean,
      beamSize: Int,
      numReturnSequences: Int,
      temperature: Double,
      topK: Int,
      topP: Double,
      repetitionPenalty: Double,
      noRepeatNgramSize: Int,
      randomSeed: Option[Long],
      ignoreTokenIds: Array[Int]): Seq[Annotation] = {

    val session =
      tensorflowWrapper.getTFSessionWithSignature(configProtoBytes, savedSignatures = signatures)

    def freeResources(): Unit = {
      tensorResources.clearTensors()
      session.close()
    }

    val batchedAudio = audios.grouped(batchSize).toArray
    val batchDecodedIds =
      batchedAudio.flatMap { batch: Seq[AnnotationAudio] =>
        val featuresBatch = batch.map { case AnnotationAudio(_, rawFloats, _) =>
          preprocessor.extractFeatures(rawFloats)
        }.toArray

        val encodedBatchFeatures = encode(featuresBatch, session)

        // TODO: Add language or other special tokens at the start
        val batchDecoderStartIds = Array.fill(batchSize, 1)(bosTokenId)

        // Generate the tokens
        val tokenIds: Array[Array[Int]] = generate(
          encodedBatchFeatures,
          batchDecoderStartIds,
          maxOutputLength,
          minOutputLength,
          doSample,
          beamSize,
          numReturnSequences,
          temperature,
          topK,
          topP,
          repetitionPenalty,
          noRepeatNgramSize,
          randomSeed,
          ignoreTokenIds,
          session)

        freeResources()
        decode(tokenIds)
      }

    // TODO: begin and end index?
    var sentBegin, nextSentEnd = 0
    batchDecodedIds.zip(audios).map { case (content, audio) =>
      nextSentEnd += content.length - 1
      val annotation = new Annotation(
        annotatorType = AnnotatorType.DOCUMENT,
        begin = sentBegin,
        end = nextSentEnd,
        result = content,
        metadata = audio.metadata)
      sentBegin += nextSentEnd + 1
      annotation
    }
  }

  /** Decode a sequence of sentences
    *
    * @param sentences
    *   Sequence of sentences
    * @return
    *   Sequence of decoded sentences
    */
  def decode(sentences: Array[Array[Int]]): Seq[String] = {
    sentences.map(s => tokenDecoder.decodeTokens(s))
  }

  /** Encodes a batch of preprocessed input audio.
    *
    * @param features
    *   Batch of Whisper features
    * @return
    *   Tensor with encoded features for each batch
    */
  def encode(features: Array[Array[Array[Float]]], session: Session): Tensor = {
    val runner: Session#Runner =
      session.runner

    val featuresTensors =
      tensorResources.createTensor[Array[Array[Array[Float]]]](features)

    val encoderOutputs: Tensor = runner
      .feed(encoderInputOp, featuresTensors)
      .fetch(encoderOutputOp)
      .run()
      .asScala
      .head

    encoderOutputs
  }

  override def getModelOutput(
      encoderInputIds: Seq[Array[Int]],
      decoderInputIds: Seq[Array[Int]],
      decoderEncoderStateTensors: Tensor,
      encoderAttentionMaskTensors: Tensor,
      maxLength: Int,
      session: Session): Array[Array[Float]] = {
    getModelOutput(decoderEncoderStateTensors, decoderInputIds, maxLength, session)
  }

  /** Get model output for a batch of input sequences
    *
    * TODO: Caching
    *
    * @param encodedInputsTensor
    *   Batch of encoded features as a Tensor
    * @param decoderInputIds
    *   Batch of decoder input ids
    * @param maxLength
    *   Max length of the output
    * @param session
    *   tensorflow session
    * @return
    *   Model output logits for the last input token for the batches
    */
  def getModelOutput(
      encodedInputsTensor: Tensor,
      decoderInputIds: Seq[Array[Int]],
      maxLength: Int,
      session: Session): Array[Array[Float]] = {

    //    val sequencesLength = decoderInputIds.map(x => x.length).toArray

    // TODO: If max length exceeded?
    val truncatedInputIds = decoderInputIds.map(_.slice(0, maxLength))

    val decoderInputIdsTensor: Tensor =
      tensorResources.createTensor[Array[Array[Int]]](truncatedInputIds.toArray)

    val runner = session.runner
      .feed(decoderInputIdsOp, decoderInputIdsTensor)
      .feed(decoderEncoderOutputsOp, encodedInputsTensor)
      .fetch(decoderOutputOp)

    val decoderOuts = runner.run().asScala
    val logitsRaw = TensorResources.extractFloats(decoderOuts.head)
    decoderOuts.head.close()

    val nextTokenLogits =
      logitsRaw.grouped(logitsOutputSize).toArray // Should result in length batch size
    tensorResources.clearTensors()
    nextTokenLogits
  }

  def generate(
      decoderEncoderStateTensors: Tensor,
      decoderInputIds: Array[Array[Int]],
      maxOutputLength: Int,
      minOutputLength: Int,
      doSample: Boolean,
      beamSize: Int,
      numReturnSequences: Int,
      temperature: Double,
      topK: Int,
      topP: Double,
      repetitionPenalty: Double,
      noRepeatNgramSize: Int,
      randomSeed: Option[Long],
      ignoreTokenIds: Array[Int],
      session: Session): Array[Array[Int]] = {

    val dummyEncoderInput =
      Seq.fill(decoderInputIds.length)(Array.empty[Int]) // Needs to be size of batch
    val dummyEncoderAttentionMaskTensors: Tensor = null // not needed

    if (beamSize == 1) // Equivalent to greedy search
      super.generateGreedy(
        encoderInputIds = dummyEncoderInput,
        decoderEncoderStateTensors = decoderEncoderStateTensors,
        encoderAttentionMaskTensors = dummyEncoderAttentionMaskTensors,
        decoderInputs = decoderInputIds,
        maxOutputLength = maxOutputLength,
        minOutputLength = minOutputLength,
        vocabSize = logitsOutputSize,
        eosTokenId = eosTokenId,
        paddingTokenId = paddingTokenId,
        ignoreTokenIds = ignoreTokenIds,
        applySoftmax = false,
        logitProcessor = None,
        session = session)
    else
      super.generate(
        inputIds = dummyEncoderInput,
        decoderEncoderStateTensors = decoderEncoderStateTensors,
        encoderAttentionMaskTensors = dummyEncoderAttentionMaskTensors,
        decoderInputs = decoderInputIds,
        maxOutputLength = maxOutputLength,
        minOutputLength = minOutputLength,
        doSample = doSample,
        beamSize = beamSize,
        numReturnSequences = numReturnSequences,
        temperature = temperature,
        topK = topK,
        topP = topP,
        repetitionPenalty = repetitionPenalty,
        noRepeatNgramSize = noRepeatNgramSize,
        vocabSize = logitsOutputSize,
        eosTokenId = eosTokenId,
        paddingTokenId = paddingTokenId,
        randomSeed = randomSeed,
        ignoreTokenIds = ignoreTokenIds,
        session = session,
        applySoftmax = false)
  }

  private def sessionWarmup(): Unit = {
    val dummyInput = Seq(AnnotationAudio(AnnotatorType.AUDIO, Array.ofDim(1), Map.empty))

    generateFromAudio(
      dummyInput,
      batchSize = 2,
      maxOutputLength = 1,
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
      ignoreTokenIds = Array.empty)
  }
}
