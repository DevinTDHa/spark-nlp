---
layout: model
title: Castilian, Spanish setfit_alpaca_spanish_unprocessable_sample_detection MPNetEmbeddings from hackathon-somos-nlp-2023
author: John Snow Labs
name: setfit_alpaca_spanish_unprocessable_sample_detection
date: 2023-09-07
tags: [mpnet, es, open_source, onnx]
task: Embeddings
language: es
edition: Spark NLP 5.1.1
spark_version: 3.0
supported: true
engine: onnx
annotator: MPNetEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MPNetEmbeddings  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`setfit_alpaca_spanish_unprocessable_sample_detection` is a Castilian, Spanish model originally trained by hackathon-somos-nlp-2023.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/setfit_alpaca_spanish_unprocessable_sample_detection_es_5.1.1_3.0_1694128554320.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/setfit_alpaca_spanish_unprocessable_sample_detection_es_5.1.1_3.0_1694128554320.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
embeddings =MPNetEmbeddings.pretrained("setfit_alpaca_spanish_unprocessable_sample_detection","es") \
            .setInputCols(["documents"]) \
            .setOutputCol("mpnet_embeddings")

pipeline = Pipeline().setStages([document_assembler, embeddings])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala


val document_assembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("documents")
    
val embeddings = MPNetEmbeddings 
    .pretrained("setfit_alpaca_spanish_unprocessable_sample_detection", "es")
    .setInputCols(Array("documents")) 
    .setOutputCol("mpnet_embeddings") 

val pipeline = new Pipeline().setStages(Array(document_assembler, embeddings))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|setfit_alpaca_spanish_unprocessable_sample_detection|
|Compatibility:|Spark NLP 5.1.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[mpnet_embeddings]|
|Language:|es|
|Size:|407.2 MB|

## References

https://huggingface.co/hackathon-somos-nlp-2023/setfit-alpaca-es-unprocessable-sample-detection