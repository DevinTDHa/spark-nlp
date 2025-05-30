---
layout: model
title: Croatian sent_crosloengual_bert_pipeline pipeline BertSentenceEmbeddings from EMBEDDIA
author: John Snow Labs
name: sent_crosloengual_bert_pipeline
date: 2024-08-31
tags: [hr, open_source, pipeline, onnx]
task: Embeddings
language: hr
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_crosloengual_bert_pipeline` is a Croatian model originally trained by EMBEDDIA.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_crosloengual_bert_pipeline_hr_5.4.2_3.0_1725121085096.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_crosloengual_bert_pipeline_hr_5.4.2_3.0_1725121085096.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_crosloengual_bert_pipeline", lang = "hr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_crosloengual_bert_pipeline", lang = "hr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_crosloengual_bert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|hr|
|Size:|464.0 MB|

## References

https://huggingface.co/EMBEDDIA/crosloengual-bert

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings