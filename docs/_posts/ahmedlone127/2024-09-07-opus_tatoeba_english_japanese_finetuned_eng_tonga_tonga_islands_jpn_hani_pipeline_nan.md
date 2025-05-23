---
layout: model
title: None opus_tatoeba_english_japanese_finetuned_eng_tonga_tonga_islands_jpn_hani_pipeline pipeline MarianTransformer from julianty
author: John Snow Labs
name: opus_tatoeba_english_japanese_finetuned_eng_tonga_tonga_islands_jpn_hani_pipeline
date: 2024-09-07
tags: [nan, open_source, pipeline, onnx]
task: Translation
language: nan
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`opus_tatoeba_english_japanese_finetuned_eng_tonga_tonga_islands_jpn_hani_pipeline` is a None model originally trained by julianty.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/opus_tatoeba_english_japanese_finetuned_eng_tonga_tonga_islands_jpn_hani_pipeline_nan_5.5.0_3.0_1725747329366.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/opus_tatoeba_english_japanese_finetuned_eng_tonga_tonga_islands_jpn_hani_pipeline_nan_5.5.0_3.0_1725747329366.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("opus_tatoeba_english_japanese_finetuned_eng_tonga_tonga_islands_jpn_hani_pipeline", lang = "nan")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("opus_tatoeba_english_japanese_finetuned_eng_tonga_tonga_islands_jpn_hani_pipeline", lang = "nan")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|opus_tatoeba_english_japanese_finetuned_eng_tonga_tonga_islands_jpn_hani_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|nan|
|Size:|542.6 MB|

## References

https://huggingface.co/julianty/opus-tatoeba-en-ja-finetuned-eng-to-jpn_Hani

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer