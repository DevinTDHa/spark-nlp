---
layout: model
title: English distilbart_mnli_12_3_pipeline pipeline BartForZeroShotClassification from valhalla
author: John Snow Labs
name: distilbart_mnli_12_3_pipeline
date: 2025-06-24
tags: [en, open_source, pipeline, onnx]
task: Zero-Shot Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BartForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbart_mnli_12_3_pipeline` is a English model originally trained by valhalla.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbart_mnli_12_3_pipeline_en_5.5.1_3.0_1750785072005.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbart_mnli_12_3_pipeline_en_5.5.1_3.0_1750785072005.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("distilbart_mnli_12_3_pipeline", lang = "en")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("distilbart_mnli_12_3_pipeline", lang = "en")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbart_mnli_12_3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|684.1 MB|

## References

References

https://huggingface.co/valhalla/distilbart-mnli-12-3

## Included Models

- DocumentAssembler
- TokenizerModel
- BartForZeroShotClassification