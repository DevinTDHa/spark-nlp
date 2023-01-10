---
layout: model
title: Bemba (Zambia) asr_wav2vec2_large_xls_r_1b_bemba_fds_gpu TFWav2Vec2ForCTC for GPU from csikasote
author: John Snow Labs
name: asr_wav2vec2_large_xls_r_1b_bemba_fds_gpu
date: 2022-09-26
tags: [wav2vec2, bem, audio, open_source, asr]
task: Automatic Speech Recognition
language: bem
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: Wav2Vec2ForCTC
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2vec2  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_wav2vec2_large_xls_r_1b_bemba_fds` is a Bemba (Zambia) model originally trained by csikasote.

NOTE: This model only works on a GPU, if you need to use this model on a CPU device please use asr_wav2vec2_large_xls_r_1b_bemba_fds

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_wav2vec2_large_xls_r_1b_bemba_fds_gpu_bem_4.2.0_3.0_1664193530490.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

audio_assembler = AudioAssembler() \
    .setInputCol("audio_content") \
    .setOutputCol("audio_assembler")

speech_to_text = Wav2Vec2ForCTC \
    .pretrained("asr_wav2vec2_large_xls_r_1b_bemba_fds_gpu", "bem")\
    .setInputCols("audio_assembler") \
    .setOutputCol("text")

pipeline = Pipeline(stages=[
  audio_assembler,
  speech_to_text,
])

pipelineModel = pipeline.fit(audioDf)

pipelineDF = pipelineModel.transform(audioDf)
```
```scala

val audioAssembler = new AudioAssembler()
    .setInputCol("audio_content") 
    .setOutputCol("audio_assembler")

val speechToText = Wav2Vec2ForCTC
    .pretrained("asr_wav2vec2_large_xls_r_1b_bemba_fds_gpu", "bem")
    .setInputCols("audio_assembler") 
    .setOutputCol("text") 

val pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))

val pipelineModel = pipeline.fit(audioDf)

val pipelineDF = pipelineModel.transform(audioDf)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|asr_wav2vec2_large_xls_r_1b_bemba_fds_gpu|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|bem|
|Size:|3.6 GB|