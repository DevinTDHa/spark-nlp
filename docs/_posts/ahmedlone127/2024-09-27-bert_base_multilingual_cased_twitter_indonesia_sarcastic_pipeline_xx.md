---
layout: model
title: Multilingual bert_base_multilingual_cased_twitter_indonesia_sarcastic_pipeline pipeline BertForSequenceClassification from w11wo
author: John Snow Labs
name: bert_base_multilingual_cased_twitter_indonesia_sarcastic_pipeline
date: 2024-09-27
tags: [xx, open_source, pipeline, onnx]
task: Text Classification
language: xx
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_base_multilingual_cased_twitter_indonesia_sarcastic_pipeline` is a Multilingual model originally trained by w11wo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_multilingual_cased_twitter_indonesia_sarcastic_pipeline_xx_5.5.0_3.0_1727417827404.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_multilingual_cased_twitter_indonesia_sarcastic_pipeline_xx_5.5.0_3.0_1727417827404.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_base_multilingual_cased_twitter_indonesia_sarcastic_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_base_multilingual_cased_twitter_indonesia_sarcastic_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_multilingual_cased_twitter_indonesia_sarcastic_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|667.3 MB|

## References

https://huggingface.co/w11wo/bert-base-multilingual-cased-twitter-indonesia-sarcastic

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification