---
layout: model
title: Ukrainian ukrt5_base_pipeline pipeline T5Transformer from uaritm
author: John Snow Labs
name: ukrt5_base_pipeline
date: 2024-08-11
tags: [uk, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: uk
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ukrt5_base_pipeline` is a Ukrainian model originally trained by uaritm.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ukrt5_base_pipeline_uk_5.4.2_3.0_1723351000369.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ukrt5_base_pipeline_uk_5.4.2_3.0_1723351000369.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ukrt5_base_pipeline", lang = "uk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ukrt5_base_pipeline", lang = "uk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ukrt5_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|uk|
|Size:|533.7 MB|

## References

https://huggingface.co/uaritm/ukrt5-base

## Included Models

- DocumentAssembler
- T5Transformer