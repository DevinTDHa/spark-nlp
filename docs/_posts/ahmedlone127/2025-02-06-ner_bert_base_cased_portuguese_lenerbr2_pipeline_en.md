---
layout: model
title: English ner_bert_base_cased_portuguese_lenerbr2_pipeline pipeline BertForTokenClassification from luciolrv
author: John Snow Labs
name: ner_bert_base_cased_portuguese_lenerbr2_pipeline
date: 2025-02-06
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ner_bert_base_cased_portuguese_lenerbr2_pipeline` is a English model originally trained by luciolrv.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_bert_base_cased_portuguese_lenerbr2_pipeline_en_5.5.1_3.0_1738842842971.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_bert_base_cased_portuguese_lenerbr2_pipeline_en_5.5.1_3.0_1738842842971.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ner_bert_base_cased_portuguese_lenerbr2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ner_bert_base_cased_portuguese_lenerbr2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_bert_base_cased_portuguese_lenerbr2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|406.0 MB|

## References

https://huggingface.co/luciolrv/ner-bert-base-cased-pt-lenerbr2

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification