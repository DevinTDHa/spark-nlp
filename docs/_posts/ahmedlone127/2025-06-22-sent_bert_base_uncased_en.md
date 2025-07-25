---
layout: model
title: BERT Sentence Embeddings (Base Uncased)
author: John Snow Labs
name: sent_bert_base_uncased
date: 2025-06-22
tags: [open_source, embeddings, en, openvino]
task: Embeddings
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: openvino
annotator: BertSentenceEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model contains a deep bidirectional transformer trained on Wikipedia and the BookCorpus. The details are described in the paper "[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)".

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_base_uncased_en_5.5.1_3.0_1750615779017.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_base_uncased_en_5.5.1_3.0_1750615779017.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
embeddings = BertSentenceEmbeddings.pretrained("sent_bert_base_uncased", "en") \
.setInputCols("sentence") \
.setOutputCol("sentence_embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I hate cancer', "Antibiotics aren't painkiller"]], ["text"]))
```
```scala
...
val embeddings = BertSentenceEmbeddings.pretrained("sent_bert_base_uncased", "en")
.setInputCols("sentence")
.setOutputCol("sentence_embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, embeddings))
val data = Seq("I hate cancer, "Antibiotics aren't painkiller").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["I hate cancer", "Antibiotics aren't painkiller"]
embeddings_df = nlu.load('en.embed_sentence.bert_base_uncased').predict(text, output_level='sentence')
embeddings_df
```
</div>

## Results

```bash

	en_embed_sentence_bert_base_uncased_embeddings 	      sentence
		
	[0.48797768354415894, 0.250575453042984, -0.24... 	I hate cancer
	[0.06506041437387466, 0.06032593920826912, -0.... 	Antibiotics aren't painkiller
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_base_uncased|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|407.2 MB|