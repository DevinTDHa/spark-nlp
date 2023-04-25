package com.johnsnowlabs.nlp.annotators.audio.feature_extractor

import breeze.linalg.{DenseMatrix, DenseVector, max}
import breeze.signal.support.WindowFunctions.hanningWindow
import com.johnsnowlabs.nlp.annotators.audio.feature_extractor.AudioUtils.matrixToFloatArray

class WhisperPreprocessor(
    val chunk_length: Int,
    val feature_extractor_type: String,
    override val feature_size: Int,
    val hop_length: Int,
    val n_fft: Int,
    val n_samples: Int,
    val nb_max_frames: Int,
    override val padding_side: String,
    override val padding_value: Float,
    val processor_class: String,
    override val return_attention_mask: Boolean,
    override val sampling_rate: Int)
    extends Preprocessor(
      do_normalize = false,
      feature_size = feature_size,
      padding_side = padding_side,
      padding_value = padding_value,
      return_attention_mask = return_attention_mask,
      sampling_rate = sampling_rate) {

  require(n_fft < n_samples, "n_fft should be smaller than n_samples.")
  require(hop_length > 0, "hop_length must be greater than 0.")

  def getHanningWindow(periodic: Boolean = true): DenseVector[Double] = {
    val windowLength = if (periodic) n_fft + 1 else n_fft
    val window = hanningWindow(windowLength)
    if (periodic) window(0 to -2) // Remove last element, so window is periodic
    else window
  }

  private val window: DenseVector[Double] = getHanningWindow()

  private val melFilterBank: DenseMatrix[Double] = AudioUtils.melFilterBank(
    numFrequencyBins = 1 + n_fft / 2,
    numMelFilters = feature_size,
    minFrequency = 0.0,
    maxFrequency = 8000.0,
    samplingRate = sampling_rate)

  /** Creates the log-mel spectrogram of given float waveform and transforms it into features for
   * the Whisper model. We assume, that the input has already been normalized and center zero
   * padded.
   *
   * Adapted from huggingface transformer.
   *
   * @param waveform
   * The waveform to transform into features
   * @return
   * Extracted Features
   */
  def extractFeatures(waveform: Array[Float]): Array[Array[Float]] = {

    val waveFormVector: DenseVector[Double] = DenseVector(waveform.map(_.toDouble))

    // Calculate spectrogram first
    val logSpectogram: DenseMatrix[Double] = AudioUtils.calculateSpectrogram(
      waveform = waveFormVector,
      window = window,
      frameLength = n_fft,
      hopLength = hop_length,
      power = 2.0d,
      melFilters = melFilterBank)

    val processedLogSpec: Array[Array[Float]] = {
      val logSpec = logSpectogram(::, 0 to -2)
      val maxes = max(logSpec, max(logSpec) - 8.0)
      val scaled = (maxes + 4.0) / 4.0

      matrixToFloatArray(scaled)
    }

    processedLogSpec
  }
}
