---
layout: model
title: Multilingual roberta_ner_roberta_base_biomedical_clinical_spanish_finetuned_ner_craft_augmentedtransfer_english_pipeline pipeline RoBertaForTokenClassification from StivenLancheros
author: John Snow Labs
name: roberta_ner_roberta_base_biomedical_clinical_spanish_finetuned_ner_craft_augmentedtransfer_english_pipeline
date: 2024-09-02
tags: [xx, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_ner_roberta_base_biomedical_clinical_spanish_finetuned_ner_craft_augmentedtransfer_english_pipeline` is a Multilingual model originally trained by StivenLancheros.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_ner_roberta_base_biomedical_clinical_spanish_finetuned_ner_craft_augmentedtransfer_english_pipeline_xx_5.5.0_3.0_1725311077665.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_ner_roberta_base_biomedical_clinical_spanish_finetuned_ner_craft_augmentedtransfer_english_pipeline_xx_5.5.0_3.0_1725311077665.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_ner_roberta_base_biomedical_clinical_spanish_finetuned_ner_craft_augmentedtransfer_english_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_ner_roberta_base_biomedical_clinical_spanish_finetuned_ner_craft_augmentedtransfer_english_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_ner_roberta_base_biomedical_clinical_spanish_finetuned_ner_craft_augmentedtransfer_english_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|438.8 MB|

## References

https://huggingface.co/StivenLancheros/roberta-base-biomedical-clinical-es-finetuned-ner-CRAFT_AugmentedTransfer_EN

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification