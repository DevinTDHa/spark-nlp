---
layout: model
title: Thai thainer_corpus_v2_base_model_pipeline pipeline CamemBertForTokenClassification from pythainlp
author: John Snow Labs
name: thainer_corpus_v2_base_model_pipeline
date: 2025-05-27
tags: [th, open_source, pipeline, onnx]
task: Named Entity Recognition
language: th
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`thainer_corpus_v2_base_model_pipeline` is a Thai model originally trained by pythainlp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/thainer_corpus_v2_base_model_pipeline_th_5.5.1_3.0_1748370237828.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/thainer_corpus_v2_base_model_pipeline_th_5.5.1_3.0_1748370237828.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("thainer_corpus_v2_base_model_pipeline", lang = "th")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("thainer_corpus_v2_base_model_pipeline", lang = "th")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|thainer_corpus_v2_base_model_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|th|
|Size:|392.3 MB|

## References

https://huggingface.co/pythainlp/thainer-corpus-v2-base-model

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertForTokenClassification