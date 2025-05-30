---
layout: model
title: English CamembertForTokenClassification Cased model (from HueyNemud)
author: John Snow Labs
name: camembert_classifier_das22_41_pretrained_finetuned_ref
date: 2024-01-21
tags: [camembert, ner, open_source, en, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: CamemBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamembertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `das22-41-camembert_pretrained_finetuned_ref` is a English model originally trained by `HueyNemud`.

## Predicted Entities

`TITRE`, `MISC`, `ACT`, `FT`, `LOC`, `ORG`, `PER`, `CARDINAL`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/camembert_classifier_das22_41_pretrained_finetuned_ref_en_5.2.4_3.0_1705831981939.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/camembert_classifier_das22_41_pretrained_finetuned_ref_en_5.2.4_3.0_1705831981939.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")
        
sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

sequenceClassifier_loaded = CamemBertForTokenClassification.pretrained("camembert_classifier_das22_41_pretrained_finetuned_ref","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler,sentenceDetector,tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
       .setInputCols(Array("document"))
       .setOutputCol("sentence")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

val sequenceClassifier_loaded = CamemBertForTokenClassification.pretrained("camembert_classifier_das22_41_pretrained_finetuned_ref","en") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector,tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("en.ner.camembert.finetuned").predict("""PUT YOUR STRING HERE""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|camembert_classifier_das22_41_pretrained_finetuned_ref|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|412.9 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

References

- https://huggingface.co/HueyNemud/das22-41-camembert_pretrained_finetuned_ref
- https://doi.org/10.1007/978-3-031-06555-2_30
- https://github.com/soduco/paper-ner-bench-das22