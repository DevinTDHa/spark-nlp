---
layout: model
title: Persian mt5_small_parsinlu_translation_english_persian_farsi_pipeline pipeline T5Transformer from persiannlp
author: John Snow Labs
name: mt5_small_parsinlu_translation_english_persian_farsi_pipeline
date: 2024-08-07
tags: [fa, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: fa
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_small_parsinlu_translation_english_persian_farsi_pipeline` is a Persian model originally trained by persiannlp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_small_parsinlu_translation_english_persian_farsi_pipeline_fa_5.4.2_3.0_1723032138804.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_small_parsinlu_translation_english_persian_farsi_pipeline_fa_5.4.2_3.0_1723032138804.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_small_parsinlu_translation_english_persian_farsi_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_small_parsinlu_translation_english_persian_farsi_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_small_parsinlu_translation_english_persian_farsi_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|819.6 MB|

## References

https://huggingface.co/persiannlp/mt5-small-parsinlu-translation_en_fa

## Included Models

- DocumentAssembler
- T5Transformer