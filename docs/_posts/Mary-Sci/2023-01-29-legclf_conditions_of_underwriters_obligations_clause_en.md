---
layout: model
title: Legal Conditions Of Underwriters Obligations Clause Binary Classifier
author: John Snow Labs
name: legclf_conditions_of_underwriters_obligations_clause
date: 2023-01-29
tags: [en, legal, classification, conditions, underwriters, obligations, clauses, conditions_of_underwriters_obligations, licensed, tensorflow]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: LegalClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Binary Classifier (True, False) for the `conditions-of-underwriters-obligations` clause type. To use this model, make sure you provide enough context as an input. Adding Sentence Splitters to the pipeline will make the model see only sentences, not the whole text, so it's better to skip it, unless you want to do Binary Classification as sentence level.

If you have big legal documents, and you want to look for clauses, we recommend you to split the documents using any of the techniques available in our Legal NLP Workshop Tokenization & Splitting Tutorial (link [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Legal/1.Tokenization_Splitting.ipynb)), namely:
- Paragraph splitting (by multiline);
- Splitting by headers / subheaders;
- etc.

Take into consideration the embeddings of this model allows up to 512 tokens. If you have more than that, consider splitting in smaller pieces (you can also check the same tutorial link provided above).

This model can be combined with any of the other 200+ Legal Clauses Classifiers you will find in Models Hub, getting as an output a series of True/False values for each of the legal clause model you have added.

## Predicted Entities

`conditions-of-underwriters-obligations`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_conditions_of_underwriters_obligations_clause_en_1.0.0_3.0_1675005111493.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_conditions_of_underwriters_obligations_clause_en_1.0.0_3.0_1675005111493.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python

document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
  
embeddings = nlp.BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en")\
    .setInputCols("document")\
    .setOutputCol("sentence_embeddings")
    
doc_classifier = legal.ClassifierDLModel.pretrained("legclf_conditions_of_underwriters_obligations_clause", "en", "legal/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("category")
    
nlpPipeline = nlp.Pipeline(stages=[
    document_assembler, 
    embeddings,
    doc_classifier])
 
df = spark.createDataFrame([["YOUR TEXT HERE"]]).toDF("text")

model = nlpPipeline.fit(df)

result = model.transform(df)

```

</div>

## Results

```bash

+-------+
|result|
+-------+
|[conditions-of-underwriters-obligations]|
|[other]|
|[other]|
|[conditions-of-underwriters-obligations]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_conditions_of_underwriters_obligations_clause|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.8 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house

## Benchmarking

```bash
                                 label  precision    recall  f1-score   support
conditions-of-underwriters-obligations       0.98      0.98      0.98        55
                                 other       0.99      0.99      0.99       104
                              accuracy          -         -      0.99       159
                             macro-avg       0.99      0.99      0.99       159
                          weighted-avg       0.99      0.99      0.99       159
```