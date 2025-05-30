---
layout: model
title: Dutch, Flemish gpt_neo_125m_dutch_pipeline pipeline GPT2Transformer from yhavinga
author: John Snow Labs
name: gpt_neo_125m_dutch_pipeline
date: 2025-01-24
tags: [nl, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: nl
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gpt_neo_125m_dutch_pipeline` is a Dutch, Flemish model originally trained by yhavinga.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gpt_neo_125m_dutch_pipeline_nl_5.5.1_3.0_1737716339099.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gpt_neo_125m_dutch_pipeline_nl_5.5.1_3.0_1737716339099.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gpt_neo_125m_dutch_pipeline", lang = "nl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gpt_neo_125m_dutch_pipeline", lang = "nl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gpt_neo_125m_dutch_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|nl|
|Size:|470.5 MB|

## References

https://huggingface.co/yhavinga/gpt-neo-125M-dutch

## Included Models

- DocumentAssembler
- GPT2Transformer