---
layout: model
title: Chinese whisper_tiny_chinese_zhihcheng_pipeline pipeline WhisperForCTC from ZhihCheng
author: John Snow Labs
name: whisper_tiny_chinese_zhihcheng_pipeline
date: 2024-09-17
tags: [zh, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: zh
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_tiny_chinese_zhihcheng_pipeline` is a Chinese model originally trained by ZhihCheng.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_tiny_chinese_zhihcheng_pipeline_zh_5.5.0_3.0_1726551468653.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_tiny_chinese_zhihcheng_pipeline_zh_5.5.0_3.0_1726551468653.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_tiny_chinese_zhihcheng_pipeline", lang = "zh")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_tiny_chinese_zhihcheng_pipeline", lang = "zh")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_tiny_chinese_zhihcheng_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|zh|
|Size:|389.6 MB|

## References

https://huggingface.co/ZhihCheng/whisper-tiny-zh

## Included Models

- AudioAssembler
- WhisperForCTC