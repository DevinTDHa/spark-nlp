package com.johnsnowlabs.nlp.annotators.audio

import com.johnsnowlabs.nlp.AudioAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.base.LightPipeline
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.TestUtils.readFile
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec

class WhisperForCTCTest extends AnyFlatSpec {
  lazy val spark: SparkSession = ResourceHelper.spark
  import spark.implicits._

  behavior of "WhisperForCTCTest"

  val pathToFileWithFloats = "src/test/resources/audio/csv/librispeech_asr_0.csv"

  val audioAssembler: AudioAssembler = new AudioAssembler()
    .setInputCol("audio_content")
    .setOutputCol("audio_assembler")

  val processedAudioFloats: Dataset[Row] =
    spark.read
      .option("inferSchema", value = true)
      .json("src/test/resources/audio/json/audio_floats.json")
      .select($"float_array".cast("array<float>").as("audio_content"))

  // Needs to be added manually
  val modelPath =
    "/home/ducha/spark-nlp/dev-things/hf_exports/whisper/exported/openai/whisper-tiny.en_sepV2/"

  lazy val model = WhisperForCTC
    .loadSavedModel(modelPath, ResourceHelper.spark)
    .setInputCols("audio_assembler")
    .setOutputCol("document")

  it should "loadSavedModel" taggedAs SlowTest in {
    val whisper = WhisperForCTC.loadSavedModel(modelPath, ResourceHelper.spark)
    print(whisper)
  }

  it should "correctly transform speech to text from already processed audio files" taggedAs SlowTest in {

    val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, model))

    val rawFloats: Array[Float] =
      readFile(s"src/test/resources/audio/txt/librispeech_asr_0.txt").split("\n").map(_.toFloat)

    val processedAudioFloats = Seq(rawFloats).toDF("audio_content")
    processedAudioFloats.printSchema()

    val pipelineDF = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)

    pipelineDF.select("document").show(1, truncate = false)

  }

  it should "correctly work with Tokenizer" taggedAs SlowTest in {

    val token = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val pipeline: Pipeline =
      new Pipeline().setStages(Array(audioAssembler, model, token))

    val bufferedSource =
      scala.io.Source.fromFile(pathToFileWithFloats)

    val rawFloats = bufferedSource
      .getLines()
      .map(_.split(",").head.trim.toFloat)
      .toArray
    bufferedSource.close

    val processedAudioFloats = Seq(rawFloats).toDF("audio_content")
    processedAudioFloats.printSchema()

    val pipelineDF = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)

    pipelineDF.select("document").show(10, truncate = false)
    pipelineDF.select("token").show(10, truncate = false)

  }

  it should "transform speech to text with LightPipeline" taggedAs SlowTest in {
    val token = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val pipeline: Pipeline =
      new Pipeline().setStages(Array(audioAssembler, model, token))

    val bufferedSource =
      scala.io.Source.fromFile(pathToFileWithFloats)

    val rawFloats = bufferedSource
      .getLines()
      .map(_.split(",").head.trim.toFloat)
      .toArray
    bufferedSource.close

    val processedAudioFloats = Seq(rawFloats).toDF("audio_content")

    val pipelineModel = pipeline.fit(processedAudioFloats)
    val lightPipeline = new LightPipeline(pipelineModel)
    val result = lightPipeline.fullAnnotate(rawFloats)

    println(result("token"))
    assert(result("audio_assembler").nonEmpty)
    assert(result("document").nonEmpty)
    assert(result("token").nonEmpty)
  }

  it should "transform several speeches to text with LightPipeline" taggedAs SlowTest in {
    val token = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val pipeline: Pipeline =
      new Pipeline().setStages(Array(audioAssembler, model, token))

    val bufferedSource =
      scala.io.Source.fromFile(pathToFileWithFloats)

    val rawFloats = bufferedSource
      .getLines()
      .map(_.split(",").head.trim.toFloat)
      .toArray
    bufferedSource.close

    val processedAudioFloats = Seq(rawFloats).toDF("audio_content")

    val pipelineModel = pipeline.fit(processedAudioFloats)
    val lightPipeline = new LightPipeline(pipelineModel)
    val results = lightPipeline.fullAnnotate(Array(rawFloats, rawFloats))

    results.foreach { result =>
      println(result("token"))
      assert(result("audio_assembler").nonEmpty)
      assert(result("document").nonEmpty)
      assert(result("token").nonEmpty)
    }

  }

  it should "be serializable" taggedAs SlowTest in {

    val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, model))

    val pipelineModel = pipeline.fit(processedAudioFloats)
    pipelineModel.stages.last
      .asInstanceOf[WhisperForCTC]
      .write
      .overwrite()
      .save("./tmp_whisper_model")

    val loadedModel = WhisperForCTC.load("./tmp_whisper_model")
    val newPipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, loadedModel))

    newPipeline
      .fit(processedAudioFloats)
      .transform(processedAudioFloats)
      .select("document")
      .show(10, truncate = false)
  }

}
