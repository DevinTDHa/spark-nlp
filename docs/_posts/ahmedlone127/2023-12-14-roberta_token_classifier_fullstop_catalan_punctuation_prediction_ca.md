---
layout: model
title: Catalan RobertaForTokenClassification Cased model (from softcatala)
author: John Snow Labs
name: roberta_token_classifier_fullstop_catalan_punctuation_prediction
date: 2023-12-14
tags: [ca, open_source, roberta, token_classification, ner, onnx]
task: Named Entity Recognition
language: ca
edition: Spark NLP 5.2.1
spark_version: 3.0
supported: true
engine: onnx
annotator: RoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `fullstop-catalan-punctuation-prediction` is a Catalan model originally trained by `softcatala`.

## Predicted Entities

`.`, `?`, `-`, `:`, `0`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_fullstop_catalan_punctuation_prediction_ca_5.2.1_3.0_1702512604737.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_fullstop_catalan_punctuation_prediction_ca_5.2.1_3.0_1702512604737.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

tokenClassifier = RobertaForTokenClassification.pretrained("roberta_token_classifier_fullstop_catalan_punctuation_prediction","ca") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = RobertaForTokenClassification.pretrained("roberta_token_classifier_fullstop_catalan_punctuation_prediction","ca")
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_token_classifier_fullstop_catalan_punctuation_prediction|
|Compatibility:|Spark NLP 5.2.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|ca|
|Size:|456.8 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

References

- https://huggingface.co/softcatala/fullstop-catalan-punctuation-prediction
- https://github.com/oliverguhr/fullstop-deep-punctuation-prediction