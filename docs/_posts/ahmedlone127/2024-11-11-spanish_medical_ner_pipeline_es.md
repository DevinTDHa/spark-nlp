---
layout: model
title: Castilian, Spanish spanish_medical_ner_pipeline pipeline BertForTokenClassification from HUMADEX
author: John Snow Labs
name: spanish_medical_ner_pipeline
date: 2024-11-11
tags: [es, open_source, pipeline, onnx]
task: Named Entity Recognition
language: es
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`spanish_medical_ner_pipeline` is a Castilian, Spanish model originally trained by HUMADEX.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spanish_medical_ner_pipeline_es_5.5.1_3.0_1731299333046.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/spanish_medical_ner_pipeline_es_5.5.1_3.0_1731299333046.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("spanish_medical_ner_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("spanish_medical_ner_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|spanish_medical_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|403.7 MB|

## References

https://huggingface.co/HUMADEX/spanish_medical_ner

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification