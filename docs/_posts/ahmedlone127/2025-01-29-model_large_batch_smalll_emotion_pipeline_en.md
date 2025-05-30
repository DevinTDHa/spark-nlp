---
layout: model
title: English model_large_batch_smalll_emotion_pipeline pipeline XlmRoBertaForSequenceClassification from Lareb00
author: John Snow Labs
name: model_large_batch_smalll_emotion_pipeline
date: 2025-01-29
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`model_large_batch_smalll_emotion_pipeline` is a English model originally trained by Lareb00.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/model_large_batch_smalll_emotion_pipeline_en_5.5.1_3.0_1738177036440.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/model_large_batch_smalll_emotion_pipeline_en_5.5.1_3.0_1738177036440.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("model_large_batch_smalll_emotion_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("model_large_batch_smalll_emotion_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|model_large_batch_smalll_emotion_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|809.4 MB|

## References

https://huggingface.co/Lareb00/model_large_batch-smalll-emotion

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification