---
layout: model
title: Tagalog roberta_tagalog_base_ft_udpos213_slovene_pipeline pipeline RoBertaForTokenClassification from iceman2434
author: John Snow Labs
name: roberta_tagalog_base_ft_udpos213_slovene_pipeline
date: 2025-02-03
tags: [tl, open_source, pipeline, onnx]
task: Named Entity Recognition
language: tl
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_tagalog_base_ft_udpos213_slovene_pipeline` is a Tagalog model originally trained by iceman2434.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_tagalog_base_ft_udpos213_slovene_pipeline_tl_5.5.1_3.0_1738563992732.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_tagalog_base_ft_udpos213_slovene_pipeline_tl_5.5.1_3.0_1738563992732.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_tagalog_base_ft_udpos213_slovene_pipeline", lang = "tl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_tagalog_base_ft_udpos213_slovene_pipeline", lang = "tl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_tagalog_base_ft_udpos213_slovene_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|tl|
|Size:|407.2 MB|

## References

https://huggingface.co/iceman2434/roberta-tagalog-base-ft-udpos213-sl

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification