---
layout: model
title: English distilbert_token_classifier_autotrain_final_784824218_pipeline pipeline DistilBertForTokenClassification from Lucifermorningstar011
author: John Snow Labs
name: distilbert_token_classifier_autotrain_final_784824218_pipeline
date: 2024-09-02
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained DistilBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_token_classifier_autotrain_final_784824218_pipeline` is a English model originally trained by Lucifermorningstar011.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_autotrain_final_784824218_pipeline_en_5.5.0_3.0_1725268054317.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_autotrain_final_784824218_pipeline_en_5.5.0_3.0_1725268054317.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilbert_token_classifier_autotrain_final_784824218_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilbert_token_classifier_autotrain_final_784824218_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_token_classifier_autotrain_final_784824218_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.2 MB|

## References

https://huggingface.co/Lucifermorningstar011/autotrain-final-784824218

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification