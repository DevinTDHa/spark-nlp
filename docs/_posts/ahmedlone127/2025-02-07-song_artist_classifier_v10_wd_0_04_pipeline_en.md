---
layout: model
title: English song_artist_classifier_v10_wd_0_04_pipeline pipeline RoBertaForSequenceClassification from tjl223
author: John Snow Labs
name: song_artist_classifier_v10_wd_0_04_pipeline
date: 2025-02-07
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`song_artist_classifier_v10_wd_0_04_pipeline` is a English model originally trained by tjl223.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/song_artist_classifier_v10_wd_0_04_pipeline_en_5.5.1_3.0_1738941887675.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/song_artist_classifier_v10_wd_0_04_pipeline_en_5.5.1_3.0_1738941887675.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("song_artist_classifier_v10_wd_0_04_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("song_artist_classifier_v10_wd_0_04_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|song_artist_classifier_v10_wd_0_04_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|309.1 MB|

## References

https://huggingface.co/tjl223/song-artist-classifier-v10-wd-0-04

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification