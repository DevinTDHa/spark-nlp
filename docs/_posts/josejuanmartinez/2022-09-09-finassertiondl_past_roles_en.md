---
layout: model
title: Identify Job Experiences in the Past
author: John Snow Labs
name: finassertiondl_past_roles
date: 2022-09-09
tags: [en, finance, assertion, status, job, experiences, past, licensed]
task: Assertion Status
language: en
edition: Spark NLP for Finance 1.0.0
spark_version: 3.2
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is aimed to detect if any Role, Job Title, Person, Organization, Date, etc. entity, extracted with NER, is expressed as a Past Experience.

## Predicted Entities

`NO_PAST`, `PAST`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/ASSERTIONDL_PAST_ROLES/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finassertiondl_past_roles_en_1.0.0_3.2_1662762393161.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

# Tokenizer splits words in a relevant format for NLP
tokenizer = Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")

embeddings = BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
# Add as many NER as you wish here. We have added 2 as an example.
# ===============
tokenClassifier = FinanceBertForTokenClassifier.pretrained("finner_bert_roles", "en", "finance/models")\
  .setInputCols("token", "document")\
  .setOutputCol("label")

ner = FinanceNerModel.pretrained("finner_org_per_role", "en", "finance/models")\
  .setInputCols("document", "token", "embeddings")\
  .setOutputCol("label2")

ner_converter = NerConverterInternal() \
    .setInputCols(["document", "token", "label"]) \
    .setOutputCol("ner_chunk")

ner_converter2 = NerConverterInternal() \
    .setInputCols(["document", "token", "label2"]) \
    .setOutputCol("ner_chunk2")

merger =  ChunkMergeApproach()\
    .setInputCols(["ner_chunk", "ner_chunk2"])\
    .setOutputCol("merged_chunk")
# ===============

assertion = AssertionDLModel.pretrained("finassertiondl_past_roles", "en", "finance/models")\
    .setInputCols(["document", "merged_chunk", "embeddings"]) \
    .setOutputCol("assertion")
    
nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    tokenizer,
    embeddings,
    tokenClassifier,
    ner,
    ner_converter,
    ner_converter2,
    merger,
    assertion
    ])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)
lp = LightPipeline(model)
r = lp.fullAnnotate("Mrs. Charles was before Managing Director at Liberty, LC")
```

</div>

## Results

```bash
chunk,begin,end,entity_type,assertion
Mrs. Charles,0,11,PERSON,PAST
Managing Director,24,40,ROLE,PAST
Liberty, LC,45,55,ORG,PAST
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finassertiondl_past_roles|
|Type:|finance|
|Compatibility:|Spark NLP for Finance 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, doc_chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|
|Size:|2.2 MB|

## References

In-house annotations from 10K Filings and Wikidata

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
NO_PAST	 362	 6	 13	 0.9836956	 0.96533334	 0.974428
PAST	 196	 13	 6	 0.93779904	 0.97029704	 0.9537713
tp: 558 fp: 19 fn: 19 labels: 2
Macro-average	 prec: 0.96074736, rec: 0.96781516, f1: 0.96426827
Micro-average	 prec: 0.96707106, rec: 0.96707106, f1: 0.96707106
```