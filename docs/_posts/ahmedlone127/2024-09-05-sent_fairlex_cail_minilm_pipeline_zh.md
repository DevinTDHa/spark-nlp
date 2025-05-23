---
layout: model
title: Chinese sent_fairlex_cail_minilm_pipeline pipeline XlmRoBertaSentenceEmbeddings from coastalcph
author: John Snow Labs
name: sent_fairlex_cail_minilm_pipeline
date: 2024-09-05
tags: [zh, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained XlmRoBertaSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_fairlex_cail_minilm_pipeline` is a Chinese model originally trained by coastalcph.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_fairlex_cail_minilm_pipeline_zh_5.5.0_3.0_1725504465975.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_fairlex_cail_minilm_pipeline_zh_5.5.0_3.0_1725504465975.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_fairlex_cail_minilm_pipeline", lang = "zh")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_fairlex_cail_minilm_pipeline", lang = "zh")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_fairlex_cail_minilm_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|zh|
|Size:|403.5 MB|

## References

https://huggingface.co/coastalcph/fairlex-cail-minilm

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- XlmRoBertaSentenceEmbeddings