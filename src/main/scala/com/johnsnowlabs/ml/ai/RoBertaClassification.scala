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

package com.johnsnowlabs.ml.ai

import ai.onnxruntime.OnnxTensor
import com.johnsnowlabs.ml.ai.util.PrepareEmbeddings
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.BpeTokenizer
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.{ActivationFunction, Annotation, AnnotatorType}
import org.tensorflow.ndarray.buffer.IntDataBuffer
import org.slf4j.{Logger, LoggerFactory}
import spire.math.interval.Open

import scala.collection.JavaConverters._

/** @param tensorflowWrapper
  *   Bert Model wrapper with TensorFlow Wrapper
  * @param sentenceStartTokenId
  *   Id of sentence start Token
  * @param sentenceEndTokenId
  *   Id of sentence end Token.
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  * @param tags
  *   labels which model was trained with in order
  * @param signatures
  *   TF v2 signatures in Spark NLP
  */
private[johnsnowlabs] class RoBertaClassification(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrapper: Option[OnnxWrapper],
    val openvinoWrapper: Option[OpenvinoWrapper],
    val sentenceStartTokenId: Int,
    val sentenceEndTokenId: Int,
    val sentencePadTokenId: Int,
    configProtoBytes: Option[Array[Byte]] = None,
    tags: Map[String, Int],
    signatures: Option[Map[String, String]] = None,
    merges: Map[(String, String), Int],
    vocabulary: Map[String, Int],
    threshold: Float = 0.5f)
    extends Serializable
    with XXXForClassification {

  protected val logger: Logger = LoggerFactory.getLogger("RoBertaClassification")
  val _tfRoBertaSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())
  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else if (onnxWrapper.isDefined) ONNX.name
    else TensorFlow.name
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  protected val sigmoidThreshold: Float = threshold

  def tokenizeWithAlignment(
      sentences: Seq[TokenizedSentence],
      maxSeqLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = {

    val bpeTokenizer =
      BpeTokenizer.forModel("roberta", merges, vocabulary, alwaysAddPrefix = false)

    sentences.map { tokenIndex =>
      // filter empty and only whitespace tokens
      val bertTokens =
        tokenIndex.indexedTokens.filter(x => x.token.nonEmpty && !x.token.equals(" ")).map {
          token =>
            val content = if (caseSensitive) token.token else token.token.toLowerCase()
            val sentenceBegin = token.begin
            val sentenceEnd = token.end
            val sentenceIndex = tokenIndex.sentenceIndex
            val result =
              bpeTokenizer.tokenize(Sentence(content, sentenceBegin, sentenceEnd, sentenceIndex))
            if (result.nonEmpty) result.head else IndexedToken("")
        }
      val wordpieceTokens =
        bertTokens.flatMap(token => bpeTokenizer.encode(token)).take(maxSeqLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
  }

  def tokenizeSeqString(
      candidateLabels: Seq[String],
      maxSeqLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = {

    val basicTokenizer = new BasicTokenizer(caseSensitive)
    val encoder = new WordpieceEncoder(vocabulary, unkToken = "<unk>")

    val labelsToSentences = candidateLabels.map { s => Sentence(s, 0, s.length - 1, 0) }

    labelsToSentences.map(label => {
      val tokens = basicTokenizer.tokenize(label)
      val wordpieceTokens = tokens.flatMap(token => encoder.encode(token)).take(maxSeqLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    })
  }
  def tokenizeDocument(
      docs: Seq[Annotation],
      maxSeqLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = {
    // we need the original form of the token
    // let's lowercase if needed right before the encoding
    val bpeTokenizer =
      BpeTokenizer.forModel("roberta", merges, vocabulary, alwaysAddPrefix = false)
    val sentences = docs.map { s => Sentence(s.result, s.begin, s.end, 0) }

    sentences.map { sentence =>
      val content = if (caseSensitive) sentence.content else sentence.content.toLowerCase()
      val sentenceBegin = sentence.start
      val sentenceEnd = sentence.end
      val sentenceIndex = sentence.index

      val indexedTokens =
        bpeTokenizer.tokenize(Sentence(content, sentenceBegin, sentenceEnd, sentenceIndex))

      val wordpieceTokens = bpeTokenizer.encode(indexedTokens).take(maxSeqLength)

      WordpieceTokenizedSentence(wordpieceTokens)
    }

  }

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {
    val batchLength = batch.length
    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max

    val rawScores = detectedEngine match {
      case ONNX.name => getRawScoresWithOnnx(batch)
      case Openvino.name => getRawScoresWithOv(batch)
      case _ => getRawScoresWithTF(batch, maxSentenceLength)
    }

    val dim = rawScores.length / (batchLength * maxSentenceLength)
    val batchScores: Array[Array[Array[Float]]] = rawScores
      .grouped(dim)
      .map(scores => calculateSoftmax(scores))
      .toArray
      .grouped(maxSentenceLength)
      .toArray

    batchScores
  }

  private def getRawScoresWithTF(batch: Seq[Array[Int]], maxSentenceLength: Int): Array[Float] = {
    val tensors = new TensorResources()

    val batchLength = batch.length

    val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)

    // [nb of encoded sentences , maxSentenceLength]
    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex
      .foreach { case (sentence, idx) =>
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(sentence)
        maskBuffers
          .offset(offset)
          .write(sentence.map(x => if (x == sentencePadTokenId) 0 else 1))
      }

    val session = tensorflowWrapper.get.getTFSessionWithSignature(
      configProtoBytes = configProtoBytes,
      savedSignatures = signatures,
      initAllTables = false)
    val runner = session.runner

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createIntBufferTensor(shape, maskBuffers)

    runner
      .feed(
        _tfRoBertaSignatures
          .getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfRoBertaSignatures
          .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
        maskTensors)
      .fetch(_tfRoBertaSignatures
        .getOrElse(ModelSignatureConstants.LogitsOutput.key, "missing_logits_key"))

    val outs = runner.run().asScala
    val rawScores = TensorResources.extractFloats(outs.head)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()

    rawScores
  }

  private def getRawScoresWithOnnx(batch: Seq[Array[Int]]): Array[Float] = {

    // [nb of encoded sentences , maxSentenceLength]
    val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)

    val tokenTensors =
      OnnxTensor.createTensor(env, batch.map(x => x.map(x => x.toLong)).toArray)
    val maskTensors =
      OnnxTensor.createTensor(
        env,
        batch.map(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray)

    val inputs =
      Map("input_ids" -> tokenTensors, "attention_mask" -> maskTensors).asJava

    try {
      val results = runner.run(inputs)
      try {
        val embeddings = results
          .get("logits")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()

        embeddings
      } finally if (results != null) results.close()
    } catch {
      case e: Exception =>
        // Handle exceptions by logging or other means.
        e.printStackTrace()
        Array.empty[Float] // Return an empty array or appropriate error handling
    } finally {
      // Close tensors outside the try-catch to avoid repeated null checks.
      // These resources are initialized before the try-catch, so they should be closed here.
      tokenTensors.close()
      maskTensors.close()
    }
  }

  private def padArrayWithZeros(arr: Array[Int], maxLength: Int): Array[Int] = {
    if (arr.length >= maxLength) {
      arr
    } else {
      arr ++ Array.fill(maxLength - arr.length)(sentencePadTokenId)
    }
  }

  def tagSequence(batch: Seq[Array[Int]], activation: String): Array[Array[Float]] = {
    val batchLength = batch.length
    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max

    val rawScores = detectedEngine match {
      case ONNX.name => getRawScoresWithOnnx(batch)
      case Openvino.name => getRawScoresWithOv(batch)
      case TensorFlow.name => getRawScoresWithTF(batch, maxSentenceLength)
    }

    val dim = rawScores.length / batchLength
    val batchScores: Array[Array[Float]] =
      rawScores
        .grouped(dim)
        .map(scores =>
          activation match {
            case ActivationFunction.softmax => calculateSoftmax(scores)
            case ActivationFunction.sigmoid => calculateSigmoid(scores)
            case _ => calculateSoftmax(scores)
          })
        .toArray

    batchScores
  }

  private def getRawScoresWithOv(batch: Seq[Array[Int]]): Array[Float] = {

    val maxSentenceLength = batch.map(_.length).max
    val batchLength = batch.length
    val shape = Array(batchLength, maxSentenceLength)
    val (tokenTensors, maskTensors) =
      PrepareEmbeddings.prepareOvLongBatchTensors(batch, maxSentenceLength, batchLength)

    val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
    inferRequest.set_tensor("input_ids", tokenTensors)
    inferRequest.set_tensor("attention_mask", maskTensors)

    inferRequest.infer()

    try {
      try {
        inferRequest
          .get_tensor("logits")
          .data()
      }
    } catch {
      case e: Exception =>
        // Log the exception as a warning
        logger.warn("Exception in getRawScoresWithOv", e)
        // Rethrow the exception to propagate it further
        throw e
    }

  }

  def computeZeroShotLogitsWithOv(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int): Array[Float] = {

    val batchLength = batch.length
    val shape = Array(batchLength, maxSentenceLength)
    val (tokenTensors, maskTensors) =
      PrepareEmbeddings.prepareOvLongBatchTensors(batch, maxSentenceLength, batchLength)

    val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
    inferRequest.set_tensor("input_ids", tokenTensors)
    inferRequest.set_tensor("attention_mask", maskTensors)

    inferRequest.infer()

    try {
      try {
        inferRequest
          .get_tensor("logits")
          .data()
      }
    } catch {
      case e: Exception =>
        // Log the exception as a warning
        logger.warn("Exception in computeZeroShotLogitsWithOv", e)
        // Rethrow the exception to propagate it further
        throw e

    }
  }

  def computeZeroShotLogitsWithONNX(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int): Array[Float] = {

    val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)

    val tokenTensors =
      OnnxTensor.createTensor(env, batch.map(x => x.map(x => x.toLong)).toArray)
    val maskTensors =
      OnnxTensor.createTensor(
        env,
        batch.map(sentence => sentence.map(x => if (x == sentencePadTokenId) 0L else 1L)).toArray)

    val inputs =
      Map("input_ids" -> tokenTensors, "attention_mask" -> maskTensors).asJava

    try {
      val results = runner.run(inputs)
      try {
        val embeddings = results
          .get("logits")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()
        tokenTensors.close()
        maskTensors.close()

        embeddings
      } finally if (results != null) results.close()
    }

  }

  def tagZeroShotSequence(
      batch: Seq[Array[Int]],
      entailmentId: Int,
      contradictionId: Int,
      activation: String): Array[Array[Float]] = {

    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val paddedBatch = batch.map(arr => padArrayWithZeros(arr, maxSentenceLength))
    val batchLength = paddedBatch.length

    val rawScores = detectedEngine match {
      case ONNX.name => computeZeroShotLogitsWithONNX(paddedBatch, maxSentenceLength)
      case Openvino.name => computeZeroShotLogitsWithOv(paddedBatch, maxSentenceLength)
      case TensorFlow.name => computeZeroShotLogitsWithTF(paddedBatch, maxSentenceLength)
    }

    val dim = rawScores.length / batchLength
    rawScores
      .grouped(dim)
      .toArray
  }

  def computeZeroShotLogitsWithTF(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int): Array[Float] = {
    val tensors = new TensorResources()

    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val batchLength = batch.length

    val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)

    // [nb of encoded sentences , maxSentenceLength]
    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex
      .foreach { case (sentence, idx) =>
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(sentence)
        maskBuffers.offset(offset).write(sentence.map(x => if (x == sentencePadTokenId) 0 else 1))

      }

    val session = tensorflowWrapper.get.getTFSessionWithSignature(
      configProtoBytes = configProtoBytes,
      savedSignatures = signatures,
      initAllTables = false)
    val runner = session.runner

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createIntBufferTensor(shape, maskBuffers)

    runner
      .feed(
        _tfRoBertaSignatures.getOrElse(
          ModelSignatureConstants.InputIds.key,
          "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfRoBertaSignatures
          .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
        maskTensors)
      .fetch(_tfRoBertaSignatures
        .getOrElse(ModelSignatureConstants.LogitsOutput.key, "missing_logits_key"))

    val outs = runner.run().asScala
    val rawScores = TensorResources.extractFloats(outs.head)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()
    rawScores
  }

  def tagSpan(batch: Seq[Array[Int]]): (Array[Array[Float]], Array[Array[Float]]) = {
    val batchLength = batch.length
    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val (startLogits, endLogits) = detectedEngine match {
      case ONNX.name => computeLogitsWithOnnx(batch)
      case Openvino.name => computeLogitsWithOv(batch)
      case TensorFlow.name => computeLogitsWithTF(batch, maxSentenceLength)
    }

    val endDim = endLogits.length / batchLength
    val endScores: Array[Array[Float]] =
      endLogits.grouped(endDim).map(scores => calculateSoftmax(scores)).toArray

    val startDim = startLogits.length / batchLength
    val startScores: Array[Array[Float]] =
      startLogits.grouped(startDim).map(scores => calculateSoftmax(scores)).toArray

    (startScores, endScores)
  }

  override def tagSpanMultipleChoice(batch: Seq[Array[Int]]): Array[Float] = {
    val logits = detectedEngine match {
      case ONNX.name => computeLogitsMultipleChoiceWithOnnx(batch)
      case Openvino.name => computeLogitsMultipleChoiceWithOv(batch)
    }

    calculateSoftmax(logits)
  }

  private def computeLogitsMultipleChoiceWithOnnx(batch: Seq[Array[Int]]): Array[Float] = {
    val sequenceLength = batch.head.length
    val inputIds = Array(batch.map(x => x.map(_.toLong)).toArray)
    val attentionMask = Array(
      batch.map(sentence => sentence.map(x => if (x == 0L) 0L else 1L)).toArray)

    val (ortSession, ortEnv) = onnxWrapper.get.getSession(onnxSessionOptions)
    val tokenTensors = OnnxTensor.createTensor(ortEnv, inputIds)
    val maskTensors = OnnxTensor.createTensor(ortEnv, attentionMask)

    val inputs =
      Map("input_ids" -> tokenTensors, "attention_mask" -> maskTensors).asJava

    try {
      val output = ortSession.run(inputs)
      try {

        val logits = output
          .get("logits")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()

        tokenTensors.close()
        maskTensors.close()

        logits
      } finally if (output != null) output.close()
    } catch {
      case e: Exception =>
        // Log the exception as a warning
        println("Exception in computeLogitsMultipleChoiceWithOnnx: ", e)
        // Rethrow the exception to propagate it further
        throw e
    }
  }

  private def computeLogitsMultipleChoiceWithOv(batch: Seq[Array[Int]]): Array[Float] = {
    val (numChoices, sequenceLength) = (batch.length, batch.head.length)
    // batch_size, num_choices, sequence_length
    val shape = Some(Array(1, numChoices, sequenceLength))
    val (tokenTensors, maskTensors, _) =
      PrepareEmbeddings.prepareOvLongBatchTensorsWithSegment(
        batch,
        sequenceLength,
        numChoices,
        sentencePadTokenId,
        shape)

    val compiledModel = openvinoWrapper.get.getCompiledModel()
    val inferRequest = compiledModel.create_infer_request()
    inferRequest.set_tensor("input_ids", tokenTensors)
    inferRequest.set_tensor("attention_mask", maskTensors)

    inferRequest.infer()

    try {
      try {
        val logits = inferRequest
          .get_output_tensor()
          .data()

        logits
      }
    } catch {
      case e: Exception =>
        // Log the exception as a warning
        logger.warn("Exception in computeLogitsMultipleChoiceWithOv", e)
        // Rethrow the exception to propagate it further
        throw e
    }
  }

  private def computeLogitsWithTF(
      batch: Seq[Array[Int]],
      maxSentenceLength: Int): (Array[Float], Array[Float]) = {
    val batchLength = batch.length
    val tensors = new TensorResources()

    val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)

    // [nb of encoded sentences , maxSentenceLength]
    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex
      .foreach { case (sentence, idx) =>
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(sentence)
        maskBuffers
          .offset(offset)
          .write(sentence.map(x => if (x == sentencePadTokenId) 0 else 1))
      }

    val session = tensorflowWrapper.get.getTFSessionWithSignature(
      configProtoBytes = configProtoBytes,
      savedSignatures = signatures,
      initAllTables = false)
    val runner = session.runner

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createIntBufferTensor(shape, maskBuffers)

    runner
      .feed(
        _tfRoBertaSignatures
          .getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
        tokenTensors)
      .feed(
        _tfRoBertaSignatures
          .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
        maskTensors)
      .fetch(_tfRoBertaSignatures
        .getOrElse(ModelSignatureConstants.EndLogitsOutput.key, "missing_end_logits_key"))
      .fetch(_tfRoBertaSignatures
        .getOrElse(ModelSignatureConstants.StartLogitsOutput.key, "missing_start_logits_key"))

    val outs = runner.run().asScala
    val endLogits = TensorResources.extractFloats(outs.head)
    val startLogits = TensorResources.extractFloats(outs.last)

    outs.foreach(_.close())
    tensors.clearSession(outs)
    tensors.clearTensors()

    (startLogits, endLogits)
  }

  private def computeLogitsWithOv(batch: Seq[Array[Int]]): (Array[Float], Array[Float]) = {

    val batchLength = batch.length
    val maxSentenceLength = batch.map(_.length).max
    val (tokenTensors, maskTensors) =
      PrepareEmbeddings.prepareOvLongBatchTensors(batch, maxSentenceLength, batchLength)

    val inferRequest = openvinoWrapper.get.getCompiledModel().create_infer_request()
    inferRequest.set_tensor("input_ids", tokenTensors)
    inferRequest.set_tensor("attention_mask", maskTensors)

    inferRequest.infer()

    try {
      try {
        val startLogits = inferRequest
          .get_tensor("start_logits")
          .data()
        val endLogits = inferRequest
          .get_tensor("end_logits")
          .data()

        (startLogits, endLogits)
      }
    } catch {
      case e: Exception =>
        // Log the exception as a warning
        logger.warn("Exception in computeLogitsWithOv", e)
        // Rethrow the exception to propagate it further
        throw e
    }
  }
  private def computeLogitsWithOnnx(batch: Seq[Array[Int]]): (Array[Float], Array[Float]) = {
    // [nb of encoded sentences , maxSentenceLength]
    val (runner, env) = onnxWrapper.get.getSession(onnxSessionOptions)

    val tokenTensors =
      OnnxTensor.createTensor(env, batch.map(x => x.map(x => x.toLong)).toArray)
    val maskTensors =
      OnnxTensor.createTensor(env, batch.map(sentence => Array.fill(sentence.length)(1L)).toArray)

    val inputs =
      Map("input_ids" -> tokenTensors, "attention_mask" -> maskTensors).asJava

    try {
      val output = runner.run(inputs)
      try {
        val startLogits = output
          .get("start_logits")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()

        val endLogits = output
          .get("end_logits")
          .get()
          .asInstanceOf[OnnxTensor]
          .getFloatBuffer
          .array()

        tokenTensors.close()
        maskTensors.close()

        (startLogits, endLogits)
      } finally if (output != null) output.close()
    } catch {
      case e: Exception =>
        // Log the exception as a warning
        logger.warn("Exception: ", e)
        // Rethrow the exception to propagate it further
        throw e
    }
  }

  def findIndexedToken(
      tokenizedSentences: Seq[TokenizedSentence],
      sentence: (WordpieceTokenizedSentence, Int),
      tokenPiece: TokenPiece): Option[IndexedToken] = {
    tokenizedSentences(sentence._2).indexedTokens.find(p => p.begin == tokenPiece.begin)
  }

  /** Encodes two sequences to be compatible with the RoBerta models.
    *
    * Unlike other models, ReBerta requires two eos tokens to join two sequences.
    *
    * For example, the pair of sequences A, B should be joined to: `<s> A </s></s> B </s>`
    */
  override def encodeSequence(
      seq1: Seq[WordpieceTokenizedSentence],
      seq2: Seq[WordpieceTokenizedSentence],
      maxSequenceLength: Int): Seq[Array[Int]] = {

    val question = seq1
      .flatMap { wpTokSentence =>
        wpTokSentence.tokens.map(t => t.pieceId)
      }
      .toArray
      .take(maxSequenceLength - 2) ++ Array(sentenceEndTokenId, sentenceEndTokenId)

    val context = seq2
      .flatMap { wpTokSentence =>
        wpTokSentence.tokens.map(t => t.pieceId)
      }
      .toArray
      .take(maxSequenceLength - question.length - 2) ++ Array(sentenceEndTokenId)

    Seq(Array(sentenceStartTokenId) ++ question ++ context)
  }

  /** Processes logits, so that undesired logits do contribute to the output probabilities (such
    * as question and special tokens).
    *
    * @param startLogits
    *   Raw logits for the start index
    * @param endLogits
    *   Raw logits for the end index
    * @param questionLength
    *   Length of the question tokens
    * @param contextLength
    *   Length of the context tokens
    * @return
    *   Probabilities for the start and end indexes
    */
  private def processLogits(
      startLogits: Array[Float],
      endLogits: Array[Float],
      questionLength: Int,
      contextLength: Int): (Array[Float], Array[Float]) = {

    /** Sets log-logits to (almost) 0 for question and padding tokens so they can't contribute to
      * the final softmax score.
      *
      * @param scores
      *   Logits of the combined sequences
      * @return
      *   Scores, with unwanted tokens set to log-probability 0
      */
    def maskUndesiredTokens(scores: Array[Float]): Array[Float] = {
      val numSpecialTokens = 4 // 4 added special tokens in encoded sequence (1 bos, 2 eos, 1 eos)
      val totalLength = scores.length
      scores.zipWithIndex.map { case (score, i) =>
        val inQuestionTokens = i > 0 && i < questionLength + numSpecialTokens
        val isEosToken = i == totalLength - 1

        if (inQuestionTokens || isEosToken) -10000.0f
        else score
      }
    }

    val processedStartLogits = calculateSoftmax(maskUndesiredTokens(startLogits))
    val processedEndLogits = calculateSoftmax(maskUndesiredTokens(endLogits))

    (processedStartLogits, processedEndLogits)
  }

  override def predictSpan(
      documents: Seq[Annotation],
      maxSentenceLength: Int,
      caseSensitive: Boolean,
      mergeTokenStrategy: String = MergeTokenStrategy.vocab,
      engine: String = TensorFlow.name): Seq[Annotation] = {

    val questionAnnot = Seq(documents.head)
    val contextAnnot = documents.drop(1)

    val wordPieceTokenizedQuestion =
      tokenizeDocument(questionAnnot, maxSentenceLength, caseSensitive)
    val wordPieceTokenizedContext =
      tokenizeDocument(contextAnnot, maxSentenceLength, caseSensitive)
    val contextLength = wordPieceTokenizedContext.head.tokens.length
    val questionLength = wordPieceTokenizedQuestion.head.tokens.length

    val encodedInput =
      encodeSequence(wordPieceTokenizedQuestion, wordPieceTokenizedContext, maxSentenceLength)
    val (rawStartLogits, rawEndLogits) = tagSpan(encodedInput)
    val (startScores, endScores) =
      processLogits(rawStartLogits.head, rawEndLogits.head, questionLength, contextLength)

    // Drop BOS token from valid results
    val startIndex = startScores.zipWithIndex.drop(1).maxBy(_._1)
    val endIndex = endScores.zipWithIndex.drop(1).maxBy(_._1)

    val offsetStartIndex = 3 // 3 added special tokens
    val offsetEndIndex = offsetStartIndex - 1

    val allTokenPieces =
      wordPieceTokenizedQuestion.head.tokens ++ wordPieceTokenizedContext.flatMap(x => x.tokens)
    val decodedAnswer =
      allTokenPieces.slice(startIndex._2 - offsetStartIndex, endIndex._2 - offsetEndIndex)
    val content =
      mergeTokenStrategy match {
        case MergeTokenStrategy.vocab =>
          decodedAnswer.filter(_.isWordStart).map(x => x.token).mkString(" ")
        case MergeTokenStrategy.sentencePiece =>
          val token = ""
          decodedAnswer
            .map(x =>
              if (x.isWordStart) " " + token + x.token
              else token + x.token)
            .mkString("")
            .trim
      }

    val totalScore = startIndex._1 * endIndex._1
    Seq(
      Annotation(
        annotatorType = AnnotatorType.CHUNK,
        begin = 0,
        end = if (content.isEmpty) 0 else content.length - 1,
        result = content,
        metadata = Map(
          "sentence" -> "0",
          "chunk" -> "0",
          "start" -> decodedAnswer.head.begin.toString,
          "start_score" -> startIndex._1.toString,
          "end" -> decodedAnswer.last.end.toString,
          "end_score" -> endIndex._1.toString,
          "score" -> totalScore.toString)))

  }

}
