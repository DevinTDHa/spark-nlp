---
layout: model
title: English fakenewsdetection_charliebedo_pipeline pipeline RoBertaForSequenceClassification from sgonzalezsilot
author: John Snow Labs
name: fakenewsdetection_charliebedo_pipeline
date: 2025-04-02
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fakenewsdetection_charliebedo_pipeline` is a English model originally trained by sgonzalezsilot.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fakenewsdetection_charliebedo_pipeline_en_5.5.1_3.0_1743613339407.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fakenewsdetection_charliebedo_pipeline_en_5.5.1_3.0_1743613339407.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("fakenewsdetection_charliebedo_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("fakenewsdetection_charliebedo_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|fakenewsdetection_charliebedo_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|458.5 MB|

## References

https://huggingface.co/sgonzalezsilot/FakeNewsDetection_charliebedo

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification