---
layout: model
title: English DistilBertForSequenceClassification Cased model (from Danitg95)
author: John Snow Labs
name: distilbert_sequence_classifier_autotrain_kaggle_effective_arguments_1086739296
date: 2023-11-18
tags: [distilbert, sequence_classification, open_source, en, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.2.0
spark_version: 3.0
supported: true
engine: onnx
annotator: DistilBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `autotrain-kaggle-effective-arguments-1086739296` is a English model originally trained by `Danitg95`.

## Predicted Entities

`Ineffective`, `Adequate`, `Effective`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_autotrain_kaggle_effective_arguments_1086739296_en_5.2.0_3.0_1700336513539.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_autotrain_kaggle_effective_arguments_1086739296_en_5.2.0_3.0_1700336513539.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_autotrain_kaggle_effective_arguments_1086739296","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_autotrain_kaggle_effective_arguments_1086739296","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("en.classify.distil_bert.by_danitg95").predict("""PUT YOUR STRING HERE""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_sequence_classifier_autotrain_kaggle_effective_arguments_1086739296|
|Compatibility:|Spark NLP 5.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|246.0 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

References

- https://huggingface.co/Danitg95/autotrain-kaggle-effective-arguments-1086739296