---
layout: model
title: German ner_legal_german BertForTokenClassification from Sahajtomar
author: John Snow Labs
name: ner_legal_german
date: 2024-09-07
tags: [de, open_source, onnx, token_classification, bert, ner]
task: Named Entity Recognition
language: de
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ner_legal_german` is a German model originally trained by Sahajtomar.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_legal_german_de_5.5.0_3.0_1725726220672.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_legal_german_de_5.5.0_3.0_1725726220672.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')
    
tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

tokenClassifier  = BertForTokenClassification.pretrained("ner_legal_german","de") \
     .setInputCols(["documents","token"]) \
     .setOutputCol("ner")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, tokenClassifier])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCols("text")
    .setOutputCols("document")
    
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("ner_legal_german", "de")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("ner") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_legal_german|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|de|
|Size:|1.3 GB|

## References

https://huggingface.co/Sahajtomar/NER_legal_de