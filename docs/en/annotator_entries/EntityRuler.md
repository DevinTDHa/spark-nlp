{%- capture title -%}
EntityRuler
{%- endcapture -%}

{%- capture model_description -%}
Instantiated model of the EntityRulerApproach.
For usage and examples see the documentation of the main class.
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_api_link -%}
[EntityRulerModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/er/EntityRulerModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[EntityRulerModel](https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.EntityRulerModel.html)
{%- endcapture -%}

{%- capture model_source_link -%}
[EntityRulerModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/er/EntityRulerModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Fits an Annotator to match exact strings or regex patterns provided in a file against a token and assigns them a
named entity. The definitions can contain any number of named entities.

There are multiple ways and formats to set the extraction resource. It is possible to set it either as a "JSON",
"JSONL" or "CSV" file. A path to the file needs to be provided to `setPatternsResource`. The file format needs to be
set as the "format" field in the `option` parameter map and depending on the file type, additional parameters might
need to be set.

To enable regex extraction, `setEnablePatternRegex(true)` needs to be called.

If the file is in a JSON format, then the rule definitions need to be given in a list with the fields "id", "label"
and "patterns":
```
[
  {
    "label": "PERSON",
    "patterns": ["Jon", "John", "John Snow"]
  },
  {
    "label": "PERSON",
    "patterns": ["Stark", "Snow"]
  },
  {
    "label": "PERSON",
    "patterns": ["Eddard", "Eddard Stark"]
  },
  {
    "label": "LOCATION",
    "patterns": ["Winterfell"]
  }
]
```

The same fields also apply to a file in the JSONL format:
```
{"id": "names-with-j", "label": "PERSON", "patterns": ["Jon", "John", "John Snow"]}
{"id": "names-with-s", "label": "PERSON", "patterns": ["Stark", "Snow"]}
{"id": "names-with-e", "label": "PERSON", "patterns": ["Eddard", "Eddard Stark"]}
{"id": "locations", "label": "LOCATION", "patterns": ["Winterfell"]}
```


In order to use a CSV file, an additional parameter "delimiter" needs to be set. In this case, the delimiter might be
set by using `.setPatternsResource("patterns.csv", ReadAs.TEXT, Map("format"->"csv", "delimiter" -> "\\|"))`
```
PERSON|Jon
PERSON|John
PERSON|John Snow
LOCATION|Winterfell
```

**Note:**

As this annotator is operating on a token level, it will not extract entities on the whole document.
To extract entities that are split up by the tokenizer (e.g. expressions containing spaces), it is necessary to set
an exception for these in the Tokenizer.
For example, if we want to extract "Jon Snow" and "Eddard Stark", then we need to define tokenizer like so:

```
val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")
  .setExceptions(Array("Jon Snow", "Eddard Stark"))
```

Then we can define a pattern which includes spaces:
```json
[
  {
    "id": "person-regex",
    "label": "PERSON",
    "patterns": ["\\w+\\s\\w+", "\\w+-\\w+"]
  }
]
```
which will result in the names being extracted:
```
+----------------------------------------------------------------------------------------------------------------------------------------+
|entity                                                                                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
|[[chunk, 5, 16, Eddard Stark, [entity -> PERSON, sentence -> 0], []], [chunk, 47, 55, Jon Snow, [entity -> PERSON, sentence -> 1], []]]|
+----------------------------------------------------------------------------------------------------------------------------------------+
```
{%- endcapture -%}

{%- capture approach_input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture approach_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture approach_python_example -%}
# In this example, the entities file as the form of
#
# PERSON|Jon
# PERSON|John
# PERSON|John Snow
# LOCATION|Winterfell
#
# where each line represents an entity and the associated string delimited by "|".

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.common import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")
entityRuler = EntityRulerApproach() \
    .setInputCols(["document", "token"]) \
    .setOutputCol("entities") \
    .setPatternsResource(
      "patterns.csv",
      ReadAs.TEXT,
      {"format": "csv", "delimiter": "\\|"}
    ) \
    .setEnablePatternRegex(True)
pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    entityRuler
])
data = spark.createDataFrame([["Jon Snow wants to be lord of Winterfell."]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.selectExpr("explode(entities)").show(truncate=False)
+--------------------------------------------------------------------+
|col                                                                 |
+--------------------------------------------------------------------+
|[chunk, 0, 2, Jon, [entity -> PERSON, sentence -> 0], []]           |
|[chunk, 29, 38, Winterfell, [entity -> LOCATION, sentence -> 0], []]|
+--------------------------------------------------------------------+

{%- endcapture -%}

{%- capture approach_scala_example -%}
// In this example, the entities file as the form of
//
// PERSON|Jon
// PERSON|John
// PERSON|John Snow
// LOCATION|Winterfell
//
// where each line represents an entity and the associated string delimited by "|".

import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.er.EntityRulerApproach
import com.johnsnowlabs.nlp.util.io.ReadAs

import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val entityRuler = new EntityRulerApproach()
  .setInputCols("document", "token")
  .setOutputCol("entities")
  .setPatternsResource(
    "src/test/resources/entity-ruler/patterns.csv",
    ReadAs.TEXT,
    {"format": "csv", "delimiter": "|")}
  )
  .setEnablePatternRegex(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  entityRuler
))

val data = Seq("Jon Snow wants to be lord of Winterfell.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(entities)").show(false)
+--------------------------------------------------------------------+
|col                                                                 |
+--------------------------------------------------------------------+
|[chunk, 0, 2, Jon, [entity -> PERSON, sentence -> 0], []]           |
|[chunk, 29, 38, Winterfell, [entity -> LOCATION, sentence -> 0], []]|
+--------------------------------------------------------------------+

{%- endcapture -%}

{%- capture approach_api_link -%}
[EntityRulerApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/er/EntityRulerApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[EntityRulerApproach](https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.EntityRulerApproach.html)
{%- endcapture -%}

{%- capture approach_source_link -%}
[EntityRulerApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/er/EntityRulerApproach.scala)
{%- endcapture -%}


{% include templates/approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_api_link=model_python_api_link
model_api_link=model_api_link
model_source_link=model_source_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_python_api_link=approach_python_api_link
approach_api_link=approach_api_link
approach_source_link=approach_source_link
%}
