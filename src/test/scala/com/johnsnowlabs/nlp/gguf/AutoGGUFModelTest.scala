package com.johnsnowlabs.nlp.gguf

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class AutoGGUFModelTest extends AnyFlatSpec {
  import ResourceHelper.spark.implicits._

  behavior of "AutoGGUFModelTest"

  lazy val modelPath =
    "/home/ducha/Workspace/building/java-llama.cpp/models/mistral-7b-instruct-v0.2.Q2_K.gguf"

  lazy val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  lazy val model = AutoGGUFModel
    .loadSavedModel(modelPath, ResourceHelper.spark)
    .setInputCols("document")
    .setOutputCol("completions")

  lazy val pipeline = new Pipeline().setStages(Array(documentAssembler, model))

  it should "create completions" in {
    val data = Seq("Hello, I am a").toDF("text")
    val result = pipeline.fit(data).transform(data)
    result.select("completions").show(truncate = false)
  }
}
