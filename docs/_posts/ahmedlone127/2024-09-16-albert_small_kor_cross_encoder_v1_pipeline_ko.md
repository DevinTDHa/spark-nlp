---
layout: model
title: Korean albert_small_kor_cross_encoder_v1_pipeline pipeline BertForSequenceClassification from bongsoo
author: John Snow Labs
name: albert_small_kor_cross_encoder_v1_pipeline
date: 2024-09-16
tags: [ko, open_source, pipeline, onnx]
task: Text Classification
language: ko
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`albert_small_kor_cross_encoder_v1_pipeline` is a Korean model originally trained by bongsoo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_small_kor_cross_encoder_v1_pipeline_ko_5.5.0_3.0_1726459196662.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_small_kor_cross_encoder_v1_pipeline_ko_5.5.0_3.0_1726459196662.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("albert_small_kor_cross_encoder_v1_pipeline", lang = "ko")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("albert_small_kor_cross_encoder_v1_pipeline", lang = "ko")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_small_kor_cross_encoder_v1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|
|Size:|43.9 MB|

## References

https://huggingface.co/bongsoo/albert-small-kor-cross-encoder-v1

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification