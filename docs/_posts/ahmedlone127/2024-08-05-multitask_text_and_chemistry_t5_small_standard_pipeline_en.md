---
layout: model
title: English multitask_text_and_chemistry_t5_small_standard_pipeline pipeline T5Transformer from GT4SD
author: John Snow Labs
name: multitask_text_and_chemistry_t5_small_standard_pipeline
date: 2024-08-05
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`multitask_text_and_chemistry_t5_small_standard_pipeline` is a English model originally trained by GT4SD.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multitask_text_and_chemistry_t5_small_standard_pipeline_en_5.4.2_3.0_1722845563413.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multitask_text_and_chemistry_t5_small_standard_pipeline_en_5.4.2_3.0_1722845563413.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("multitask_text_and_chemistry_t5_small_standard_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("multitask_text_and_chemistry_t5_small_standard_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multitask_text_and_chemistry_t5_small_standard_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|349.3 MB|

## References

https://huggingface.co/GT4SD/multitask-text-and-chemistry-t5-small-standard

## Included Models

- DocumentAssembler
- T5Transformer