---
layout: model
title: English distillbert_for_rhetorical_rolelabeling_pipeline pipeline DistilBertForSequenceClassification from engineersaloni159
author: John Snow Labs
name: distillbert_for_rhetorical_rolelabeling_pipeline
date: 2024-09-16
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distillbert_for_rhetorical_rolelabeling_pipeline` is a English model originally trained by engineersaloni159.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distillbert_for_rhetorical_rolelabeling_pipeline_en_5.5.0_3.0_1726506184738.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distillbert_for_rhetorical_rolelabeling_pipeline_en_5.5.0_3.0_1726506184738.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distillbert_for_rhetorical_rolelabeling_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distillbert_for_rhetorical_rolelabeling_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distillbert_for_rhetorical_rolelabeling_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/engineersaloni159/DistillBERT_for_rhetorical_rolelabeling

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification