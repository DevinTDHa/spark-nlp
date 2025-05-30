/*
 * Copyright 2017-2024 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{SlowTest, FastTest}
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class M2M100TestSpec extends AnyFlatSpec {

  "m2m100" should "should translate chinese to english" taggedAs SlowTest in {
    // Even tough the Paper states temperature in interval [0,1), using temperature=0 will result in division by 0 error.
    // Also DoSample=True may result in infinities being generated and distFiltered.length==0 which results in exception if we don't return 0 instead internally.
    val testData = ResourceHelper.spark
      .createDataFrame(Seq((1, "生活就像一盒巧克力。")))
      .toDF("id", "text")
      .repartition(1)
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val m2m100 = M2M100Transformer
      .pretrained()
      .setInputCols(Array("documents"))
      .setSrcLang("zh")
      .setTgtLang("en")
      .setDoSample(false)
      .setMaxOutputLength(50)
      .setOutputCol("generation")
      .setBeamSize(1)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, m2m100))

    val pipelineModel = pipeline.fit(testData)

    val result = pipelineModel.transform(testData)

    result.show(truncate = false)

  }

  "m2m100" should "should translate hindi to french" taggedAs SlowTest in {
    // Even tough the Paper states temperature in interval [0,1), using temperature=0 will result in division by 0 error.
    // Also DoSample=True may result in infinities being generated and distFiltered.length==0 which results in exception if we don't return 0 instead internally.
    val testData = ResourceHelper.spark
      .createDataFrame(Seq((1, "जीवन एक चॉकलेट बॉक्स की तरह है।")))
      .toDF("id", "text")
      .repartition(1)
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val m2m100 = M2M100Transformer
      .pretrained()
      .setInputCols(Array("documents"))
      .setSrcLang("hi")
      .setTgtLang("fr")
      .setDoSample(false)
      .setMaxOutputLength(50)
      .setOutputCol("generation")
      .setBeamSize(1)

    new Pipeline()
      .setStages(Array(documentAssembler, m2m100))
      .fit(testData)
      .transform(testData)
      .show(truncate = false)

  }

  "m2m100" should "should translate Sinhala to English" taggedAs SlowTest in {
    // Even tough the Paper states temperature in interval [0,1), using temperature=0 will result in division by 0 error.
    // Also DoSample=True may result in infinities being generated and distFiltered.length==0 which results in exception if we don't return 0 instead internally.
    val testData = ResourceHelper.spark
      .createDataFrame(Seq((1, "ජීවිතය චොකලට් බෝතලයක් වගේ.")))
      .toDF("id", "text")
      .repartition(1)
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val m2m100 = M2M100Transformer
      .pretrained()
      .setInputCols(Array("documents"))
      .setSrcLang("si")
      .setTgtLang("en")
      .setDoSample(false)
      .setMaxOutputLength(50)
      .setOutputCol("generation")
      .setBeamSize(1)

    new Pipeline()
      .setStages(Array(documentAssembler, m2m100))
      .fit(testData)
      .transform(testData)
      .show(truncate = false)

  }
}
