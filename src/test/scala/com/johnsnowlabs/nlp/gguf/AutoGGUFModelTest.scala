package com.johnsnowlabs.nlp.gguf

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

import java.lang.management.ManagementFactory

class AutoGGUFModelTest extends AnyFlatSpec {
  import ResourceHelper.spark.implicits._

  behavior of "AutoGGUFModelTest"

  lazy val modelPath =
    "/home/ducha/Workspace/building/java-llama.cpp/models/codellama-7b.Q2_K.gguf"

  lazy val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  lazy val model = AutoGGUFModel
    .loadSavedModel(modelPath, ResourceHelper.spark)
    .setInputCols("document")
    .setOutputCol("completions")
    .setBatchSize(4)
    .setNPredict(20)
    .setNGpuLayers(99)
    .setTemperature(0.4f)
    .setTopK(40)
    .setTopP(0.9f)
    .setPenalizeNl(true)

  lazy val data = Seq(
    "The moons of Jupiter are ", // "The moons of Jupiter are 77 in total, with 79 confirmed natural satellites and 2 man-made ones. The four"
    "Earth is ", // "Earth is 4.5 billion years old. It has been home to countless species, some of which have gone extinct, while others have evolved into"
    "The moon is ", // "The moon is 1/400th the size of the sun. The sun is 1.39 million kilometers in diameter, while"
    "The sun is " //
  ).toDF("text").repartition(1)

  lazy val pipeline = new Pipeline().setStages(Array(documentAssembler, model))

  it should "create completions" in {
    val data = Seq("Hello, I am a").toDF("text")
    val result = pipeline.fit(data).transform(data)
    result.select("completions").show(truncate = false)
  }

  it should "create batch completions" in {
    val jvmName = ManagementFactory.getRuntimeMXBean.getName
    val pid = jvmName.split("@")(0)
    println(s"Running in PID $pid")

    lazy val pipeline = new Pipeline().setStages(Array(documentAssembler, model))

    val result = pipeline.fit(data).transform(data)
    result.select("completions").show(truncate = false)
  }

  it should "be serializable" in {

    val data = Seq("Hello, I am a").toDF("text")
    lazy val pipeline = new Pipeline().setStages(Array(documentAssembler, model))
    model.setNPredict(5)

    val pipelineModel = pipeline.fit(data)
    val savePath = "./tmp_autogguf_model"
    pipelineModel.stages.last
      .asInstanceOf[AutoGGUFModel]
      .write
      .overwrite()
      .save(savePath)

    val loadedModel = AutoGGUFModel.load(savePath)
    val newPipeline: Pipeline = new Pipeline().setStages(Array(documentAssembler, loadedModel))

    newPipeline
      .fit(data)
      .transform(data)
      .select("completions")
      .show(truncate = false)
  }

}
