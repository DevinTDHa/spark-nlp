---
layout: model
title: English chnsenticorp_xlm_r_pipeline pipeline XlmRoBertaForSequenceClassification from Cincin-nvp
author: John Snow Labs
name: chnsenticorp_xlm_r_pipeline
date: 2024-09-09
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`chnsenticorp_xlm_r_pipeline` is a English model originally trained by Cincin-nvp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/chnsenticorp_xlm_r_pipeline_en_5.5.0_3.0_1725908071457.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/chnsenticorp_xlm_r_pipeline_en_5.5.0_3.0_1725908071457.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("chnsenticorp_xlm_r_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("chnsenticorp_xlm_r_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chnsenticorp_xlm_r_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|850.9 MB|

## References

https://huggingface.co/Cincin-nvp/ChnSentiCorp_XLM-R

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification