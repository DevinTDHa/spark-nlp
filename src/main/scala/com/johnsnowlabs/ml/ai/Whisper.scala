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
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{BpeTokenizer, Gpt2Tokenizer}
import com.johnsnowlabs.nlp.{Annotation, AnnotationAudio, AnnotatorType}
import org.tensorflow.{Session, Tensor}

import scala.collection.JavaConverters._

/** This class is used to run the Whisper model.
 *
 * Input for this model must be tokenized with a SentencePieceModel,
 *
 * @param tensorflow
 * BART Model wrapper with TensorFlowWrapper
 * @param configProtoBytes
 * Configuration for TensorFlow session
 */

private[johnsnowlabs] class Whisper(
                                     val tensorflow: TensorflowWrapper,
                                     configProtoBytes: Option[Array[Byte]] = None,
                                     signatures: Option[Map[String, String]] = None,
                                     preprocessor: WhisperPreprocessor,
                                     merges: Map[(String, String), Int],
                                     vocabulary: Map[String, Int])
  extends Serializable
    with Generate {

  val bpeTokenizer: Gpt2Tokenizer = BpeTokenizer
    .forModel("gpt2", merges = merges, vocab = vocabulary)
    .asInstanceOf[Gpt2Tokenizer]

  private val _tfWhisperSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  // TODO: get from config
  val bosTokenId = 50257
  val paddingTokenId = 50256
  val eosTokenId = 50256
  val vocabSize = 51864 // vocabulary.length?

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
   * Sequence of audio floats
   * @param batchSize
   * Batch size
   * @param minOutputLength
   * Minimum length of output
   * @param maxOutputLength
   * Maximum length of output
   * @param doSample
   * Whether to sample or not
   * @param temperature
   * Temperature for sampling
   * @param topK
   * Top K for sampling
   * @param topP
   * Top P for sampling
   * @param repetitionPenalty
   * Repetition penalty for sampling
   * @param noRepeatNgramSize
   * No repeat ngram size for sampling
   * @param task
   * Task
   * @param randomSeed
   * Random seed
   * @param ignoreTokenIds
   * Ignore token ids
   * @param beamSize
   * Beam size
   * @return
   */
  def predict(
               audios: Seq[AnnotationAudio],
               batchSize: Int,
               minOutputLength: Int,
               maxOutputLength: Int,
               doSample: Boolean,
               temperature: Double,
               topK: Int,
               topP: Double,
               repetitionPenalty: Double,
               noRepeatNgramSize: Int,
               task: String,
               randomSeed: Option[Long] = None,
               ignoreTokenIds: Array[Int] = Array(),
               beamSize: Int): Seq[Annotation] = {

    val batchedAudio = audios.grouped(batchSize).toArray
    val batchDecoder =
      batchedAudio.flatMap { batch: Seq[AnnotationAudio] =>
        val featuresBatch = batch.map { case AnnotationAudio(_, rawFloats, _) =>
          preprocessor.extractFeatures(rawFloats)
        }.toArray

        // Generate the tokens
        val tokenIds: Array[Array[Int]] = generateTokensIds(
          featuresBatch,
          minOutputLength,
          maxOutputLength,
          doSample,
          temperature,
          topK,
          topP,
          repetitionPenalty,
          noRepeatNgramSize,
          randomSeed,
          ignoreTokenIds,
          beamSize)

        decode(tokenIds)
      }

    var sentBegin, nextSentEnd = 0
    batchDecoder.zip(audios).map { case (content, sent) =>
      nextSentEnd += content.length - 1
      val annots = new Annotation(
        annotatorType = AnnotatorType.DOCUMENT,
        begin = sentBegin,
        end = nextSentEnd,
        result = content,
        metadata = sent.metadata)
      sentBegin += nextSentEnd + 1
      annots
    }
  }

  /** Generates token ids for each batch.
   *
   * TODO: Do for batch
   *
   * @param batchFeatures
   * Sequence of extracted features
   * @param minOutputLength
   * Minimum length of output
   * @param maxOutputLength
   * Maximum length of output
   * @param doSample
   * Whether to sample or not
   * @param temperature
   * Temperature for sampling
   * @param topK
   * Top K for sampling
   * @param topP
   * Top P for sampling
   * @param repetitionPenalty
   * Repetition penalty for sampling
   * @param noRepeatNgramSize
   * No repeat ngram size for sampling
   * @param randomSeed
   * Random seed
   * @param ignoreTokenIds
   * Ignore token ids
   * @param beamSize
   * Beam size
   * @return
   * Sequence of WordpieceTokenizedSentence
   */
  private def generateTokensIds(
                                 batchFeatures: Seq[Array[Array[Float]]],
                                 minOutputLength: Int,
                                 maxOutputLength: Int,
                                 doSample: Boolean,
                                 temperature: Double,
                                 topK: Int,
                                 topP: Double,
                                 repetitionPenalty: Double,
                                 noRepeatNgramSize: Int,
                                 randomSeed: Option[Long],
                                 ignoreTokenIds: Array[Int] = Array(),
                                 beamSize: Int): Array[Array[Int]] = {
    ???

    //    val batch: Seq[Array[Int]] = Seq.fill(batchFeatures.length)(Array(bosTokenId))
    //
    //    val ignoreTokenIdsInt = ignoreTokenIds
    //    val expandedEncoderInputIdsVals = batch.flatMap(x => List.fill(beamSize)(x))
    //    val sequencesLength = expandedEncoderInputIdsVals.map(x => x.length).toArray
    //    val maxSentenceLength = sequencesLength.max // - curLen
    //
    //    val numReturn_sequences = 1
    //
    //    // from config
    //    var effectiveBatch_size = 1
    //    var effectiveBatch_mult = 1
    //
    //    // set effective batch size and effective batch multiplier according to do_sample
    //    if (doSample) {
    //      effectiveBatch_size = expandedEncoderInputIdsVals.length * numReturn_sequences
    //      effectiveBatch_mult = numReturn_sequences
    //    } else {
    //      effectiveBatch_size = expandedEncoderInputIdsVals.length
    //      effectiveBatch_mult = 1
    //    }
    //
    //    // Run encoder
    //    val tensorEncoder = new TensorResources()
    //    val inputDim = expandedEncoderInputIdsVals.length * maxSentenceLength
    //
    //    val encoderInputBuffers = tensorEncoder.createIntBuffer(inputDim)
    //    val encoderAttentionMaskBuffers = tensorEncoder.createIntBuffer(inputDim)
    //
    //    val shape = Array(expandedEncoderInputIdsVals.length.toLong, maxSentenceLength)
    //
    //    expandedEncoderInputIdsVals.zipWithIndex.foreach { case (tokenIds, idx) =>
    //      val offset = idx * maxSentenceLength
    //      val diff = maxSentenceLength - tokenIds.length
    //
    //      val s = tokenIds.take(maxSentenceLength) ++ Array.fill[Int](diff)(this.paddingTokenId)
    //      encoderInputBuffers.offset(offset).write(s)
    //      val mask = s.map(x => if (x != this.paddingTokenId) 1 else 0)
    //      encoderAttentionMaskBuffers.offset(offset).write(mask)
    //    }
    //
    //    val session = tensorflow.getTFSessionWithSignature(
    //      configProtoBytes = configProtoBytes,
    //      initAllTables = false,
    //      savedSignatures = signatures)
    //
    //    val encoderInputTensors = tensorEncoder.createIntBufferTensor(shape, encoderInputBuffers)
    //    val encoderAttentionMaskTensors =
    //      tensorEncoder.createIntBufferTensor(shape, encoderAttentionMaskBuffers)
    //
    //    val encoderOuts = encode(batchFeatures, session)
    //    val dim = encoderOutsFloats.length / inputDim
    //    val encoderOutsBatch =
    //      encoderOutsFloats.grouped(dim).toArray.grouped(maxSentenceLength).toArray
    //
    //
    //    // Run decoder
    //    val decoderEncoderStateTensorResources = new TensorResources()
    //    val decoderEncoderStateBuffers =
    //      decoderEncoderStateTensorResources.createFloatBuffer(
    //        expandedEncoderInputIdsVals.length * maxSentenceLength * dim)
    //    expandedEncoderInputIdsVals.zipWithIndex.foreach { case (_, index) =>
    //      var offset = index * maxSentenceLength * dim
    //      encoderOutsBatch(index).foreach(encoderOutput => {
    //        decoderEncoderStateBuffers.offset(offset).write(encoderOutput)
    //        offset += dim
    //      })
    //    }
    //
    //    val decoderEncoderStateTensors = tensorEncoder.createFloatBufferTensor(
    //      Array(expandedEncoderInputIdsVals.length, maxSentenceLength, dim),
    //      decoderEncoderStateBuffers)
    //    val decoderInputs = batch.map(_ => Array(this.eosTokenId)).toArray
    //    val modelOutputs = generate(
    //      batch,
    //      decoderEncoderStateTensors,
    //      encoderAttentionMaskTensors,
    //      decoderInputs,
    //      maxOutputLength,
    //      minOutputLength,
    //      doSample,
    //      beamSize,
    //      1,
    //      temperature,
    //      topK,
    //      topP,
    //      repetitionPenalty,
    //      noRepeatNgramSize,
    //      this.vocabSize,
    //      this.eosTokenId,
    //      this.paddingTokenId,
    //      randomSeed,
    //      ignoreTokenIdsInt,
    //      session)
    //
    //    tensorEncoder.clearTensors()
    //    tensorEncoder.clearSession(encoderOuts)
    //    decoderEncoderStateTensorResources.clearTensors()
    //    decoderEncoderStateTensors.close()
    //    encoderAttentionMaskTensors.close()
    //    encoderInputTensors.close()
    //    modelOutputs
  }

  /** Decode a sequence of sentences
   *
   * @param sentences
   * Sequence of sentences
   * @return
   * Sequence of decoded sentences
   */
  def decode(sentences: Array[Array[Int]]): Seq[String] = {
    sentences.map(s => bpeTokenizer.decodeTokens(s))
  }

  /** Encodes a batch of preprocessed input audio.
   *
   * @param features
   * Batch of Whisper features
   * @return
   * Tensor with encoded features for each batch
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

    // TODO: Close this Tensor
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
   * Batch of encoded features as a Tensor
   * @param decoderInputIds
   * Batch of decoder input ids
   * @param maxLength
   * Max length of the output
   * @param session
   * tensorflow session
   * @return
   * Model output logits for the last input token for the batches
   */
  def getModelOutput(
                      encodedInputsTensor: Tensor,
                      decoderInputIds: Seq[Array[Int]],
                      maxLength: Int,
                      session: Session): Array[Array[Float]] = {

    //    val sequencesLength = decoderInputIds.map(x => x.length).toArray

    // TODO: If max length exceeded, just stop?
    //    val maxSentenceLength = Math.max(sequencesLength.max, maxLength)
    // require(maxSentenceLength <= maxSentenceLength)

    val decoderInputLength = decoderInputIds.head.length

    val decoderInputIdsTensor: Tensor =
      tensorResources.createTensor[Array[Array[Int]]](decoderInputIds.toArray)

    val runner = session.runner
      .feed(decoderInputIdsOp, decoderInputIdsTensor)
      .feed(decoderEncoderOutputsOp, encodedInputsTensor)
      .fetch(decoderOutputOp)

    val decoderOuts = runner.run().asScala
    val logitsRaw = TensorResources.extractFloats(decoderOuts.head)
    decoderOuts.head.close()

    val nextTokenLogits =
      logitsRaw.grouped(vocabSize).toArray // Should result in length batch size
    tensorResources.clearTensors()
    //    encodedInputsTensor.close() // TODO: Do this later
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

    super.generate(
      dummyEncoderInput,
      decoderEncoderStateTensors,
      dummyEncoderAttentionMaskTensors,
      decoderInputIds,
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
      vocabSize,
      eosTokenId,
      paddingTokenId,
      randomSeed,
      ignoreTokenIds,
      session)
  }

  private def sessionWarmup(): Unit = {
    val dummyInput = Array.fill(1, 1)(0.0f)
    generateTokensIds(
      Array(dummyInput),
      minOutputLength = 0,
      maxOutputLength = 1,
      doSample = false,
      temperature = 0f,
      topK = 0,
      topP = 0f,
      repetitionPenalty = 0f,
      noRepeatNgramSize = 0,
      randomSeed = Option(0),
      ignoreTokenIds = Array(0),
      beamSize = 1)
  }
}
