---
layout: model
title: Finnish asr_wav2vec2_xlsr_1b_finnish_lm_v2_by_aapot_gpu TFWav2Vec2ForCTC from aapot
author: John Snow Labs
name: pipeline_asr_wav2vec2_xlsr_1b_finnish_lm_v2_by_aapot_gpu
date: 2022-09-26
tags: [wav2vec2, fi, audio, open_source, pipeline, asr]
task: Automatic Speech Recognition
language: fi
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2vec2  pipeline, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_wav2vec2_xlsr_1b_finnish_lm_v2_by_aapot` is a Finnish model originally trained by aapot.

NOTE: This pipeline only works on a CPU, if you need to use this pipeline on a GPU device please use pipeline_asr_wav2vec2_xlsr_1b_finnish_lm_v2_by_aapot

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_asr_wav2vec2_xlsr_1b_finnish_lm_v2_by_aapot_gpu_fi_4.2.0_3.0_1664210517795.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline('pipeline_asr_wav2vec2_xlsr_1b_finnish_lm_v2_by_aapot_gpu', lang = 'fi')
annotations =  pipeline.transform(audioDF)
    
```
```scala

val pipeline = new PretrainedPipeline("pipeline_asr_wav2vec2_xlsr_1b_finnish_lm_v2_by_aapot_gpu", lang = "fi")
val annotations = pipeline.transform(audioDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_asr_wav2vec2_xlsr_1b_finnish_lm_v2_by_aapot_gpu|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fi|
|Size:|3.6 GB|

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC