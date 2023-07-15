import org.scalatest.flatspec.AnyFlatSpec

class DocExamplesTest extends AnyFlatSpec {

  import com.johnsnowlabs.nlp.util.io.ResourceHelper
  import org.apache.spark.sql.SparkSession
  import com.johnsnowlabs.nlp.{Annotation, SparkNLP}
  import com.johnsnowlabs.tags.SlowTest

  //  val spark: SparkSession = ResourceHelper.getActiveSparkSession
  val spark: SparkSession = SparkNLP.start()

  behavior of "DocExamples"

  it should "MultiDocumentAssembler" ignore {
    import spark.implicits._
    import com.johnsnowlabs.nlp.MultiDocumentAssembler
    val data = Seq(
      (
        "Spark NLP is an open-source text processing library., ",
        "Spark NLP ist eine Open-Source-Bibliothek für Textverarbeitung.",
        "Spark NLP est une bibliothèque de traitement de texte à code source ouvert.")).toDF(
      "textEN",
      "textDE",
      "textFR")

    val multiDocumentAssembler =
      new MultiDocumentAssembler()
        .setInputCols("textEN", "textDE", "textFR")
        .setOutputCols("documentEN", "documentDE", "documentFR")

    val result = multiDocumentAssembler.transform(data)
    result.select("documentEN", "documentDE", "documentFR").show(1, truncate = 29)
  }

  it should "SwinForImageC" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.base._
    import com.johnsnowlabs.nlp.annotator._
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.sql.DataFrame

    val imageDF: DataFrame = ResourceHelper.spark.read
      .format("image")
      .option("dropInvalid", value = true)
      .load("src/test/resources/image/")

    val imageAssembler: ImageAssembler = new ImageAssembler()
      .setInputCol("image")
      .setOutputCol("image_assembler")

    val imageClassifier: SwinForImageClassification = SwinForImageClassification
      .pretrained(
        "image_classifier_swin_base_patch_4_window_7_224",
        "en",
        "https://devin-sparknlp-test.s3.eu-central-1.amazonaws.com/image_classifier_swin_base_patch_4_window_7_224.zip")
      .setInputCols("image_assembler")
      .setOutputCol("class")

    val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))
    val pipelineDF = pipeline.fit(imageDF).transform(imageDF)

  }

  it should "RobertaForSeq" taggedAs SlowTest ignore {
    import spark.implicits._
    import com.johnsnowlabs.nlp.base._
    import com.johnsnowlabs.nlp.annotator._
    import org.apache.spark.ml.Pipeline

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val sequenceClassifier = RoBertaForSequenceClassification
      .pretrained()
      .setInputCols("token", "document")
      .setOutputCol("label")
      .setCaseSensitive(true)

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))

    val data = Seq("I loved this movie when I was a child.", "It was pretty boring.")
      .toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("label.result").show(false)
  }

  it should "spanbert" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator._
    import com.johnsnowlabs.nlp.base._
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")
    val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")
    val corefResolution = SpanBertCorefModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("corefs")
    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, sentence, tokenizer, corefResolution))
    val data = Seq("John told Mary he would like to borrow a book from her.").toDF("text")
    val result = pipeline.fit(data).transform(data)
    result
      .selectExpr("explode (corefs) AS coref")
      .selectExpr("coref.result as token", "coref.metadata")
      .show(truncate = false)
  }

  it should "wav2vec" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.audio.Wav2Vec2ForCTC
    import com.johnsnowlabs.nlp.base._
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val audioAssembler: AudioAssembler = new AudioAssembler()
      .setInputCol("audio_content")
      .setOutputCol("audio_assembler")

    val speechToText: Wav2Vec2ForCTC = Wav2Vec2ForCTC
      .pretrained()
      .setInputCols("audio_assembler")
      .setOutputCol("text")

    val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))

    val bufferedSource =
      scala.io.Source.fromFile("src/test/resources/audio/csv/audi_floats.csv")

    val rawFloats = bufferedSource
      .getLines()
      .map(_.split(",").head.trim.toFloat)
      .toArray
    bufferedSource.close

    val processedAudioFloats = Seq(rawFloats).toDF("audio_content")
    processedAudioFloats.printSchema()

    val result = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)
    result.select("text.result").show(truncate = false)
  }

  it should "camembertembeddings" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.EmbeddingsFinisher
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.CamemBertEmbeddings
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val embeddings = CamemBertEmbeddings
      .pretrained()
      .setInputCols("token", "document")
      .setOutputCol("camembert_embeddings")

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("camembert_embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, tokenizer, embeddings, embeddingsFinisher))

    val data = Seq("C'est une phrase.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(finished_embeddings.result) as result").show(5, 80)
  }

  it should "USE multi" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator._
    import com.johnsnowlabs.nlp.base._
    import org.apache.spark.ml.Pipeline

    val smallCorpus = ResourceHelper.spark.read
      .option("header", "true")
      .csv("src/test/resources/embeddings/sentence_embeddings_use.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val useEmbeddings = UniversalSentenceEncoder
      .pretrained("tfhub_use_multi", "xx")
      .setInputCols("sentence")
      .setOutputCol("sentence_embeddings")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, useEmbeddings))
  }

  it should "nerdlpipe" taggedAs SlowTest in {
    import com.johnsnowlabs.nlp.annotator._
    import com.johnsnowlabs.nlp.base._
    import org.apache.spark.ml.Pipeline
    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val token = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    val posTagger = PerceptronModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("pos")

    val wordEmbeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("word_embeddings")

    val ner = NerDLModel
      .pretrained("ner_dl", "en")
      .setInputCols("token", "sentence", "word_embeddings")
      .setOutputCol("ner")

    val nerConverter = new NerConverter()
      .setInputCols("sentence", "token", "ner")
      .setOutputCol("ner_converter")

    val finisher = new Finisher()
      .setInputCols("ner", "ner_converter")
      .setCleanAnnotations(false)

    val pipeline = new Pipeline().setStages(
      Array(
        document,
        sentenceDetector,
        token,
        posTagger,
        wordEmbeddings,
        ner,
        nerConverter,
        finisher))

    val testData = spark
      .createDataFrame(
        Seq(
          (1, "Google has announced the release of a beta version of the popular TensorFlow machine learning library"),
          (2, "The Paris metro will soon enter the 21st century, ditching single-use paper tickets for rechargeable electronic cards.")))
      .toDF("id", "text")

    val predicion = pipeline.fit(testData).transform(testData)
    predicion.select("ner_converter.result").show(false)
    predicion.select("pos.result").show(false)
  }
  it should "AlbertTamil" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.EmbeddingsFinisher
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.AlbertEmbeddings
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val embeddings = AlbertEmbeddings
      .pretrained("albert_indic", "xx")
      .setInputCols("token", "document")
      .setOutputCol("embeddings")

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, tokenizer, embeddings, embeddingsFinisher))

    val data = Seq(
      "கர்நாடக சட்டப் பேரவையில் வெற்றி பெற்ற எம்எல்ஏக்கள் இன்று பதவியேற்றுக் கொண்ட நிலையில் , காங்கிரஸ் எம்எல்ஏ ஆனந்த் சிங் க்கள் ஆப்சென்ட் ஆகி அதிர்ச்சியை ஏற்படுத்தியுள்ளார் . உச்சநீதிமன்ற உத்தரவுப்படி இன்று மாலை முதலமைச்சர் எடியூரப்பா இன்று நம்பிக்கை வாக்கெடுப்பு நடத்தி பெரும்பான்மையை நிரூபிக்க உச்சநீதிமன்றம் உத்தரவிட்டது .")
      .toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  }

  it should "XlmRoberta" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.EmbeddingsFinisher
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.XlmRoBertaEmbeddings
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = XlmRoBertaEmbeddings
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(true)

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, embeddingsFinisher))

    val data = Seq("This is a sentence.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  }

  it should "XlmRobertaForTok" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator._
    import com.johnsnowlabs.nlp.base._
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val tokenClassifier = XlmRoBertaForTokenClassification
      .pretrained()
      .setInputCols("token", "document")
      .setOutputCol("label")
      .setCaseSensitive(true)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))

    val data = Seq(
      "John Lenon was born in London and lived in Paris. My name is Sarah and I live in London")
      .toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("label.result").show(false)
  }

  it should "XlmRobertaSentence" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.EmbeddingsFinisher
    import com.johnsnowlabs.nlp.annotator._
    import com.johnsnowlabs.nlp.base._
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val sentenceEmbeddings = XlmRoBertaSentenceEmbeddings
      .pretrained()
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")
      .setCaseSensitive(true)

    // you can either use the output to train ClassifierDL, SentimentDL, or MultiClassifierDL
    // or you can use EmbeddingsFinisher to prepare the results for Spark ML functions

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("sentence_embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, sentenceEmbeddings, embeddingsFinisher))

    val data = Seq("This is a sentence.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  }

  it should "XLNet" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.EmbeddingsFinisher
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.XlnetEmbeddings
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val embeddings = XlnetEmbeddings
      .pretrained()
      .setInputCols("token", "document")
      .setOutputCol("embeddings")

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, tokenizer, embeddings, embeddingsFinisher))

    val data = Seq("This is a sentence.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  }

  it should "XlNetForTOk" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator._
    import com.johnsnowlabs.nlp.base._
    import org.apache.spark.ml.Pipeline
    import spark.implicits._
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
    val tokenClassifier = XlnetForTokenClassification
      .pretrained()
      .setInputCols("token", "document")
      .setOutputCol("label")
      .setCaseSensitive(true)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))
    val data = Seq(
      "John Lenon was born in London and lived in Paris. My name is Sarah and I live in London")
      .toDF("text")
    val result = pipeline.fit(data).transform(data)
    result.select("label.result").show(false)
  }

  it should "Albert" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.EmbeddingsFinisher
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.AlbertEmbeddings
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val embeddings = AlbertEmbeddings
      .pretrained()
      .setInputCols("token", "document")
      .setOutputCol("embeddings")

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, tokenizer, embeddings, embeddingsFinisher))

    val data = Seq("This is a sentence.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  }

  it should "AlbertForTok" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator._
    import com.johnsnowlabs.nlp.base._
    import org.apache.spark.ml.Pipeline
    import spark.implicits._
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
    val tokenClassifier = AlbertForTokenClassification
      .pretrained()
      .setInputCols("token", "document")
      .setOutputCol("label")
      .setCaseSensitive(true)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))
    val data = Seq(
      "John Lenon was born in London and lived in Paris. My name is Sarah and I live in London")
      .toDF("text")
    val result = pipeline.fit(data).transform(data)
    result.select("label.result").show(false)
  }

  it should "T5" ignore {
    import com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val t5 = T5Transformer
      .pretrained("t5_small")
      .setTask("summarize:")
      .setInputCols(Array("documents"))
      .setMaxOutputLength(200)
      .setOutputCol("summaries")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

    val data = Seq(
      "Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a " +
        "downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness" +
        " of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this " +
        "paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework " +
        "that converts all text-based language problems into a text-to-text format. Our systematic study compares " +
        "pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens " +
        "of language understanding tasks. By combining the insights from our exploration with scale and our new " +
        "Colossal Clean Crawled Corpus, we achieve state-of-the-art results on many benchmarks covering " +
        "summarization, question answering, text classification, and more. To facilitate future work on transfer " +
        "learning for NLP, we release our data set, pre-trained models, and code.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("summaries.result").show(false)
  }

  it should "Marian" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator.SentenceDetectorDLModel
    import com.johnsnowlabs.nlp.annotators.seq2seq.MarianTransformer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = SentenceDetectorDLModel
      .pretrained("sentence_detector_dl", "xx")
      .setInputCols("document")
      .setOutputCol("sentence")

    val marian = MarianTransformer
      .pretrained()
      .setInputCols("sentence")
      .setOutputCol("translation")
      .setMaxInputLength(30)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, marian))

    val data = Seq("What is the capital of France? We should know this in french.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(translation.result) as result").show(false)
  }

  it should "T5 Style Transfer" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val t5 = T5Transformer
      .load("/home/ducha/Workspace/Training/T5/passive_to_active_styletransfer-spark-nlp")
      .setTask("transfer Passive to Active:")
      .setMaxOutputLength(200)
      .setInputCols("documents")
      .setOutputCol("transfer")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

    val data = Seq("A letter was sent to you.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("transfer.result").show(false)
  }

  it should "T5 SQL" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer
    import com.johnsnowlabs.nlp.base.DocumentAssembler

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("documents")

    val t5 = T5Transformer
      .load("/home/ducha/Workspace/Training/T5/t5-small-finetuned-wikiSQL-spark-nlp")
      .setInputCols("documents")
      .setOutputCol("sql")

    t5.save("/home/ducha/Workspace/Training/T5/t5-small-finetuned-wikiSQL-spark-nlp2")

    //    val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))
    //
    //    val data = Seq("How many customers have ordered more than 2 items?").toDF("text")
    //    val result = pipeline.fit(data).transform(data)
    //
    //    result.select("sql.result").show(false)
  }

  it should "EntityRulerApproachExceptions" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator.{EntityRulerApproach, Tokenizer}
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.util.io.ReadAs
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")
      .setExceptions(Array("Jon Snow", "Eddard Stark"))

    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource(
        path = "src/test/resources/entity-ruler/regex_patterns.json",
        readAs = ReadAs.TEXT)
//      .setEnablePatternRegex(true)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))

    val data = Seq("Jon Snow wants to be lord of Winterfell. Eddard Stark is dead.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(entities)").show(false)
  }

  it should "TransformerLoading" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator._
    import com.johnsnowlabs.nlp.base._
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val ModelName = "finiteautomata/beto-sentiment-analysis"
    val tokenClassifier = BertForSequenceClassification
      .loadSavedModel(
        s"/home/ducha/Workspace/scala/spark-nlp/python/notebooks/$ModelName/saved_model/1",
        spark)
      .setInputCols("document", "token")
      .setOutputCol("ner")
      .setCaseSensitive(false)
      .setMaxSentenceLength(128)

    // Optionally the classifier can be saved to load it more conveniently into Spark NLP
    tokenClassifier.write
      .overwrite()
      .save(s"/home/ducha/Workspace/scala/spark-nlp/python/notebooks/${ModelName}_spark_nlp")

    val tokenClassifierLoaded = BertForSequenceClassification
      .load(s"/home/ducha/Workspace/scala/spark-nlp/python/notebooks/${ModelName}_spark_nlp")
      .setInputCols("document", "token")
      .setOutputCol("ner")

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifierLoaded))

    val data = Seq("¡La película fue genial!").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("ner.result").show(truncate = false)
  }

  //  it should "Doc2VecModel" taggedAs SlowTest ignore {
  //    import spark.implicits._
  //    import com.johnsnowlabs.nlp.base.DocumentAssembler
  //    import com.johnsnowlabs.nlp.annotator.{Tokenizer}
  //    import com.johnsnowlabs.nlp.EmbeddingsFinisher
  //
  //    import org.apache.spark.ml.Pipeline
  //
  //    val documentAssembler = new DocumentAssembler()
  //      .setInputCol("text")
  //      .setOutputCol("document")
  //
  //    val tokenizer = new Tokenizer()
  //      .setInputCols(Array("document"))
  //      .setOutputCol("token")
  //
  //    val embeddings = Doc2VecModel.pretrained()
  //      .setInputCols("token")
  //      .setOutputCol("embeddings")
  //
  //    val embeddingsFinisher = new EmbeddingsFinisher()
  //      .setInputCols("embeddings")
  //      .setOutputCols("finished_embeddings")
  //      .setOutputAsVector(true)
  //
  //    val pipeline = new Pipeline().setStages(Array(
  //      documentAssembler,
  //      tokenizer,
  //      embeddings,
  //      embeddingsFinisher
  //    ))
  //
  //    val data = Seq("This is a sentence.").toDF("text")
  //    val result = pipeline.fit(data).transform(data)
  //
  //    result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  //  }
  //
  //  it should "Doc2VecApproach" taggedAs SlowTest ignore {
  //    import spark.implicits._
  //    import com.johnsnowlabs.nlp.annotator.{Tokenizer, Doc2VecApproach}
  //    import com.johnsnowlabs.nlp.base.DocumentAssembler
  //    import org.apache.spark.ml.Pipeline
  //    import com.johnsnowlabs.nlp.EmbeddingsFinisher
  //
  //    val documentAssembler = new DocumentAssembler()
  //      .setInputCol("text")
  //      .setOutputCol("document")
  //
  //    val tokenizer = new Tokenizer()
  //      .setInputCols(Array("document"))
  //      .setOutputCol("token")
  //
  //    val embeddings = new Doc2VecApproach()
  //      .setInputCols("token")
  //      .setOutputCol("embeddings")
  //
  //    val embeddingsFinisher = new EmbeddingsFinisher()
  //      .setInputCols("embeddings")
  //      .setOutputCols("finished_embeddings")
  //      .setOutputAsVector(true)
  //
  //    val pipeline = new Pipeline()
  //      .setStages(Array(
  //        documentAssembler,
  //        tokenizer,
  //        embeddings,
  //        embeddingsFinisher
  //      ))
  //
  //    val path = "src/test/resources/spell/sherlockholmes.txt"
  //    val dataset = spark.sparkContext.textFile(path)
  //      .toDF("text")
  //    val pipelineModel = pipeline.fit(dataset)
  //
  //    val data = Seq("This is a document.").toDF("text")
  //    val result = pipelineModel.transform(data)
  //
  //    result.selectExpr("explode(finished_embeddings) as result").show()
  //  }

  it should "CamembertRoberta" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.EmbeddingsFinisher
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.XlmRoBertaEmbeddings
    import org.apache.spark.ml.Pipeline
    import spark.implicits._
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")
    val embeddings = XlmRoBertaEmbeddings
      .load("/home/ducha/spark-nlp/python/notebooks/camembert-base_spark_nlp")
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(true)
    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)
    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, embeddingsFinisher))
    val data = Seq("This is a sentence.").toDF("text")
    val result = pipeline.fit(data).transform(data)
    result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  }

  it should "Show Public models pipelines" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.pretrained.ResourceDownloader

    ResourceDownloader.showPublicPipelines(lang = "en")
  }

  it should "japaneseModels2" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.DocumentAssembler
    import com.johnsnowlabs.nlp.annotator.{SentenceDetector, WordSegmenterModel}
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val word_segmenter = WordSegmenterModel
      .pretrained("wordseg_gsd_ud", "ja")
      .setInputCols("sentence")
      .setOutputCol("token")

//    val embeddings = BertEmbeddings
//      .pretrained("bert_base_japanese", "ja")
//      .setInputCols("sentence", "token")
//      .setOutputCol("embeddings")
//
//    val nerTagger = NerDLModel
//      .pretrained("ner_ud_gsd_bert_base_ja", "ja")
//      .setInputCols("sentence", "token")
//      .setOutputCol("ner")

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, sentence, word_segmenter)
    ) // , embeddings, nerTagger))

    val data = Seq("宮本茂氏は、日本の任天堂のゲームプロデューサーです。").toDF("text")
    val model = pipeline.fit(data)
    val result = model.transform(data)

//    result.selectExpr("explode(arrays_zip(token.result, ner.result))").show()
    result.selectExpr("token.result").show(truncate = false)
  }
  it should "japaneseModel" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.DocumentAssembler
    import com.johnsnowlabs.nlp.annotator.{
      BertForTokenClassification,
      SentenceDetector,
      WordSegmenterModel
    }
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val word_segmenter = WordSegmenterModel
      .pretrained("wordseg_gsd_ud", "ja")
      .setInputCols("sentence")
      .setOutputCol("token")

    val nerTagger = BertForTokenClassification
      .pretrained("./bert-base-japanese-char_ner_spark_nlp", "ja")
      .setInputCols("sentence", "token")
      .setOutputCol("ner")

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, sentence, word_segmenter, nerTagger))

    val data = Seq("宮本茂氏は、日本の任天堂のゲームプロデューサーです。").toDF("text")
    val model = pipeline.fit(data)
    val result = model.transform(data)

    result
      .selectExpr("explode(arrays_zip(token.result, ner.result))")
      .selectExpr("col'0' as token", "col'1' as ner")
      .show()
  }
  it should "externalTraining" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator._
    import com.johnsnowlabs.nlp.base._
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val modelName = "elastic/distilbert-base-cased-finetuned-conll03-english"
    var tokenClassifier = DistilBertForTokenClassification
      .loadSavedModel(s"$modelName/saved_model/1", spark)
      .setInputCols("document", "token")
      .setOutputCol("label")
      .setCaseSensitive(true)
      .setMaxSentenceLength(128)

    // Optionally the classifier can be saved to load it more conveniently into Spark NLP
    tokenClassifier.write.overwrite.save(s"${modelName}_spark_nlp")

    tokenClassifier = DistilBertForTokenClassification
      .load(s"${modelName}_spark_nlp")
      .setInputCols("token", "document")
      .setOutputCol("label")
      .setCaseSensitive(true)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))

    val data = Seq(
      "John Lenon was born in London and lived in Paris. My name is Sarah and I live in London")
      .toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("label.result").show(false)
  }

  it should "transformerdlmodel" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.AlbertEmbeddings
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    // First extract the prerequisites for the NerDLModel
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    // Use the transformer embeddings
    val embeddings = AlbertEmbeddings
      .pretrained("albert_base_uncased", "en")
      .setInputCols("token", "document")
      .setOutputCol("embeddings")

    // This pretrained model requires those specific transformer embeddings
    val nerModel = NerDLModel
      .pretrained("albert_base_uncased", "en")
      .setInputCols("document", "token", "embeddings")
      .setOutputCol("ner")

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings, nerModel))

    val data = Seq("U.N. official Ekeus heads for Baghdad.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("ner.result").show(false)
  }

  it should "transformerSentimentDL" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLModel
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = UniversalSentenceEncoder
      .pretrained("tfhub_use", lang = "en")
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    // This pretrained model requires those specific transformer embeddings
    val classifier = SentimentDLModel
      .pretrained("sentimentdl_use_imdb")
      .setInputCols("sentence_embeddings")
      .setOutputCol("sentiment")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, embeddings, classifier))

    val data = Seq("That was a fantastic movie!").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("sentiment.result").show(false)
  }

  it should "transformerclassifierApproach" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.XlmRoBertaSentenceEmbeddings
    import org.apache.spark.ml.Pipeline

    val smallCorpus =
      spark.read.option("header", "true").csv("src/test/resources/classifier/sentiment.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = XlmRoBertaSentenceEmbeddings
      .pretrained()
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    // Then the training can start with the transformer embeddings
    val docClassifier = new ClassifierDLApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol("category")
      .setLabelColumn("label")
      .setBatchSize(64)
      .setMaxEpochs(20)
      .setLr(5e-3f)
      .setDropout(0.5f)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, embeddings, docClassifier))

    val pipelineModel = pipeline.fit(smallCorpus)
  }
  it should "transfomerClassifierDL" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator.ClassifierDLModel
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    // First extract the prerequisites for the NerDLModel
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    // Use the transformer embeddings
    val embeddings = BertSentenceEmbeddings
      .pretrained("sent_bert_multi_cased", "xx")
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    // This pretrained model requires those specific transformer embeddings
    val document_classifier = ClassifierDLModel
      .pretrained("classifierdl_bert_news", "de")
      .setInputCols(Array("document", "sentence_embeddings"))
      .setOutputCol("class")

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, sentence, tokenizer, embeddings, document_classifier))

    val data = Seq(
      "The Grand Budapest Hotel is a 2014 comedy-drama film written and directed by Wes Anderson")
      .toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("ner.result").show(false)
  }
  it should "transformerModel" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
    import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.ElmoEmbeddings
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    // First extract the prerequisites for the NerDLModel
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    // Use the transformer embeddings
    val embeddings = ElmoEmbeddings
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")

    // This pretrained model requires those specific transformer embeddings
    val nerModel = NerDLModel
      .pretrained("ner_conll_elmo", "en")
      .setInputCols("document", "token", "embeddings")
      .setOutputCol("ner")

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, sentence, tokenizer, embeddings, nerModel))

    val data = Seq("U.N. official Ekeus heads for Baghdad.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("ner.result").show(false)
  }

  it should "transformerNerDLApproach" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator._
    import com.johnsnowlabs.nlp.base._
    import com.johnsnowlabs.nlp.training.CoNLL
    import org.apache.spark.ml.Pipeline
    import scala.language.existentials

    // First extract the prerequisites for the NerDLApproach
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")

    val preProcessingPipeline = new Pipeline().setStages(Array(documentAssembler, embeddings))

    // We use the text and labels from the CoNLL dataset
    val conll = CoNLL()
    val Array(trainData, testData) = conll
      .readDataset(spark, "src/test/resources/conll2003/eng.train")
      .limit(5)
      .randomSplit(Array(0.8, 0.2))

    preProcessingPipeline
      .fit(testData)
      .transform(testData)
      .write
      .mode("overwrite")
      .parquet("test_data")

//    val trainData = conll
//      .readDataset(spark = spark, path = "src/test/resources/conll2003/eng.train")
//      .limit(1000)
//    val testData = conll
//      .readDataset(spark = spark, path = "src/test/resources/conll2003/eng.train")
//      .limit(1000)

//    println(s"TEST DATASET COUNT: ${testData.count}")
//    testData.printSchema()

    // Then the training can start with the transformer embeddings
    val nerTagger = new NerDLApproach()
      .setInputCols("document", "token", "embeddings")
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(1)
      .setRandomSeed(0)
      .setVerbose(0)
      .setTestDataset("test_data")

    val pipeline = new Pipeline().setStages(Array(preProcessingPipeline, nerTagger))

    val pipelineModel = pipeline.fit(trainData)
  }

  it should "GraphExtraction" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.GraphFinisher
    import com.johnsnowlabs.nlp.annotators.{GraphExtraction, Tokenizer}
    import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
    import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
    import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
    import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
    import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    val embeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

    val nerTagger = NerDLModel
      .pretrained()
      .setInputCols("sentence", "token", "embeddings")
      .setOutputCol("ner")

    val posTagger = PerceptronModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("pos")

    val dependencyParser = DependencyParserModel
      .pretrained()
      .setInputCols("sentence", "pos", "token")
      .setOutputCol("dependency")

    val typedDependencyParser = TypedDependencyParserModel
      .pretrained()
      .setInputCols("dependency", "pos", "token")
      .setOutputCol("dependency_type")

    val graph_extraction = new GraphExtraction()
      .setInputCols("document", "token", "ner")
      .setOutputCol("graph")
      .setRelationshipTypes(Array("prefer-LOC"))
      .setMergeEntities(true)
    //      .setDependencyParserModel(Array("dependency_conllu", "en",  "public/models"))
    //      .setTypedDependencyParserModel(Array("dependency_typed_conllu", "en",  "public/models"))

    val pipeline = new Pipeline().setStages(
      Array(
        documentAssembler,
        sentence,
        tokenizer,
        embeddings,
        nerTagger,
        posTagger,
        //      dependencyParser,
        //      typedDependencyParser,
        graph_extraction))

    val data = Seq("You and John prefer the morning flight through Denver").toDF("text")
    val result = pipeline.fit(data).transform(data)

    //    result.select("graph").show(false)

    val graphFinisher = new GraphFinisher()
      .setInputCol("graph")
      .setOutputCol("graph_finished")
      .setOutputAsArray(false)

    val finishedResult = graphFinisher.transform(result)
    finishedResult.select("text", "graph_finished").show(false)
  }
  // it should "NerOverwriter" taggedAs SlowTest ignore {
  //   import spark.implicits._
  //   import com.johnsnowlabs.nlp.base.DocumentAssembler
  //   import com.johnsnowlabs.nlp.annotators.Tokenizer
  //   import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
  //   import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
  //   import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
  //   import com.johnsnowlabs.nlp.annotators.ner.NerOverwriter
  //   import org.apache.spark.ml.Pipeline

  //   // First extract the prerequisite Entities
  //   val documentAssembler = new DocumentAssembler()
  //     .setInputCol("text")
  //     .setOutputCol("document")

  //   val sentence = new SentenceDetector()
  //     .setInputCols("document")
  //     .setOutputCol("sentence")

  //   val tokenizer = new Tokenizer()
  //     .setInputCols("sentence")
  //     .setOutputCol("token")

  //   val embeddings = WordEmbeddingsModel
  //     .pretrained()
  //     .setInputCols("sentence", "token")
  //     .setOutputCol("bert")

  //   val nerTagger = NerDLModel
  //     .pretrained()
  //     .setInputCols("sentence", "token", "bert")
  //     .setOutputCol("ner")

  //   val pipeline = new Pipeline().setStages(
  //     Array(documentAssembler, sentence, tokenizer, embeddings, nerTagger))

  //   val data =
  //     Seq("Spark NLP Crosses Five Million Downloads, John Snow Labs Announces.").toDF("text")
  //   val result = pipeline.fit(data).transform(data)

  //   result.selectExpr("explode(ner)").show(false)

  //   // The recognized entity can then be overwritten
  //   val nerOverwriter = new NerOverwriter()
  //     .setInputCols("ner")
  //     .setOutputCol("ner_overwritten")
  //     .setStopWords(Array("Million"))
  //     .setNewResult("B-CARDINAL")

  //   nerOverwriter.transform(result).selectExpr("explode(ner_overwritten)").show(false)
  // }

  it should "Token2Chunk" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.{Token2Chunk, Tokenizer}
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val token2chunk = new Token2Chunk()
      .setInputCols("token")
      .setOutputCol("chunk")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, token2chunk))

    val data = Seq("One Two Three Four").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(chunk) as result").show(false)
  }

  it should "RecursiveTokenizer" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.RecursiveTokenizer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new RecursiveTokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer))

    val data = Seq("One, after the Other, (and) again. PO, QAM,").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("token.result").show(false)
  }

  it should "ChunkTokenizer" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.DocumentAssembler
    import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
    import com.johnsnowlabs.nlp.annotators.{ChunkTokenizer, TextMatcher, Tokenizer}
    import com.johnsnowlabs.nlp.util.io.ReadAs
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val entityExtractor = new TextMatcher()
      .setInputCols("sentence", "token")
      .setEntities("src/test/resources/entity-extractor/test-chunks.txt", ReadAs.TEXT)
      .setOutputCol("entity")

    val chunkTokenizer = new ChunkTokenizer()
      .setInputCols("entity")
      .setOutputCol("chunk_token")

    val pipeline = new Pipeline()
      .setStages(
        Array(documentAssembler, sentenceDetector, tokenizer, entityExtractor, chunkTokenizer))

    val data = Seq(
      "Hello world, my name is Michael, I am an artist and I work at Benezar",
      "Robert, an engineer from Farendell, graduated last year. The other one, Lucas, graduated last week.")
      .toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("entity.result as entity", "chunk_token.result as chunk_token").show(false)
  }

  it should "WordSegApproach" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.ws.WordSegmenterApproach
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.training.POS
    import org.apache.spark.ml.Pipeline

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val wordSegmenter = new WordSegmenterApproach()
      .setInputCols("document")
      .setOutputCol("token")
      .setPosColumn("tags")
      .setNIterations(5)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, wordSegmenter))

    val trainingDataSet = POS().readDataset(
      ResourceHelper.spark,
      "src/test/resources/word-segmenter/chinese_train.utf8")

    val pipelineModel = pipeline.fit(trainingDataSet)
  }

  it should "WordSegModel" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator.WordSegmenterModel
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val wordSegmenter = WordSegmenterModel
      .pretrained()
      .setInputCols("document")
      .setOutputCol("words_segmented")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, wordSegmenter))

    val data = Seq("然而，這樣的處理也衍生了一些問題。").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("words_segmented.result").show(false)
  }

  it should "RegexTokenizer" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.RegexTokenizer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val regexTokenizer = new RegexTokenizer()
      .setInputCols("document")
      .setOutputCol("regexToken")
      .setToLowercase(true)
      .setPattern(raw"\s+")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, regexTokenizer))

    val data = Seq("This is   my\tfirst sentence.\nThis is my second.").toDF("text")
    val result = pipeline.fit(data).transform(data)
    result.selectExpr("regexToken.result").show(false)
  }

  it should "typeddepModel" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
    import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
    import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
    import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    val posTagger = PerceptronModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("pos")

    val dependencyParser = DependencyParserModel
      .pretrained()
      .setInputCols("sentence", "pos", "token")
      .setOutputCol("dependency")

    val typedDependencyParser = TypedDependencyParserModel
      .pretrained()
      .setInputCols("dependency", "pos", "token")
      .setOutputCol("dependency_type")

    val pipeline = new Pipeline().setStages(
      Array(
        documentAssembler,
        sentence,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParser))

    val data = Seq(
      "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent " +
        "firm Federal Mogul.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result
      .selectExpr(
        "explode(arrays_zip(token.result, dependency.result, dependency_type.result)) as cols")
      .selectExpr("cols['0'] as token", "cols['1'] as dependency", "cols['2'] as dependency_type")
      .show(8, truncate = false)
  }

  it should "TypedDependencyParserApproach" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
    import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserApproach
    import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
    import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    val posTagger = PerceptronModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("pos")

    val dependencyParser = DependencyParserModel
      .pretrained()
      .setInputCols("sentence", "pos", "token")
      .setOutputCol("dependency")

    val typedDependencyParser = new TypedDependencyParserApproach()
      .setInputCols("dependency", "pos", "token")
      .setOutputCol("dependency_type")
      .setConllU("src/test/resources/parser/labeled/train_small.conllu.txt")
      .setNumberOfIterations(1)

    val pipeline = new Pipeline().setStages(
      Array(
        documentAssembler,
        sentence,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParser))

    // Additional training data is not needed, the dependency parser relies on CoNLL-U only.
    val emptyDataSet = Seq.empty[String].toDF("text")
    val pipelineModel = pipeline.fit(emptyDataSet)
  }

  it should "depModel" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
    import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
    import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    val posTagger = PerceptronModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("pos")

    val dependencyParserApproach = DependencyParserModel
      .pretrained()
      .setInputCols("sentence", "pos", "token")
      .setOutputCol("dependency")

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, sentence, tokenizer, posTagger, dependencyParserApproach))

    val data = Seq(
      "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent " +
        "firm Federal Mogul.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result
      .selectExpr("explode(arrays_zip(token.result, dependency.result)) as cols")
      .selectExpr("cols['0'] as token", "cols['1'] as dependency")
      .show(8, truncate = false)
    succeed
  }

  it should "DependencyParserApproach" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserApproach
    import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
    import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    val posTagger = PerceptronModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("pos")

    val dependencyParserApproach = new DependencyParserApproach()
      .setInputCols("sentence", "pos", "token")
      .setOutputCol("dependency")
      .setDependencyTreeBank("src/test/resources/parser/unlabeled/dependency_treebank")
      .setNumberOfIterations(1)

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, sentence, tokenizer, posTagger, dependencyParserApproach))

    // Additional training data is not needed, the dependency parser relies on the dependency tree bank / CoNLL-U only.
    val emptyDataSet = Seq.empty[String].toDF("text")
    val pipelineModel = pipeline.fit(emptyDataSet)
  }

  it should "ContextSpellModel" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.DocumentAssembler
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerModel
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("doc")

    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCols(Array("doc"))
      .setOutputCol("token")

    val spellChecker = ContextSpellCheckerModel
      .pretrained()
      .setTradeOff(12.0f)
      .setInputCols("token")
      .setOutputCol("checked")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, spellChecker))

    val data =
      Seq("It was a cold , dreary day and the country was white with smow .").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("checked.result").show(false)
  }

  it should "ContextSpell" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerApproach
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val spellChecker = new ContextSpellCheckerApproach()
      .setInputCols("token")
      .setOutputCol("corrected")
      .setWordMaxDistance(3)
      .setBatchSize(24)
      .setEpochs(8)
      .setLanguageModelClasses(1650) // dependant on vocabulary size
    //      .addVocabClass("_NAME_", names) // Extra classes for correction could be added like this

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, spellChecker))

    val path = "src/test/resources/spell/sherlockholmes.txt"
    val dataset = spark.sparkContext
      .textFile(path)
      .toDF("text")
    val pipelineModel = pipeline.fit(dataset)

  }

  it should "symmetricModel" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteModel
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val spellChecker = SymmetricDeleteModel
      .pretrained()
      .setInputCols("token")
      .setOutputCol("spell")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, spellChecker))

    val data = Seq("spmetimes i wrrite wordz erong.").toDF("text")
    val result = pipeline.fit(data).transform(data)
    result.select("spell.result").show(false)
  }

  it should "symmetric" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val spellChecker = new SymmetricDeleteApproach()
      .setInputCols("token")
      .setOutputCol("spell")
      .setDictionary("src/test/resources/spell/words.txt")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, spellChecker))

    //    val pipelineModel = pipeline.fit(trainingData)
  }

  it should "norvigmodel" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    println(spark.version)
    println(SparkNLP.version())

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val spellChecker: NorvigSweetingModel = NorvigSweetingModel
      .pretrained()
      .setInputCols("token")
      .setOutputCol("spell")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, spellChecker))

    val data = Seq("spmetimes i wrrite wordz erong.").toDF("text")
    val result = pipeline.fit(data).transform(data)
    result.select("spell.result").show(false)
  }

  it should "norvig" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val spellChecker = new NorvigSweetingApproach()
      .setInputCols("token")
      .setOutputCol("spell")
      .setDictionary("src/test/resources/spell/words.txt")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, spellChecker))

    val data = Seq("Sumtimes i wrrite wordswrong.").toDF("text")
    val result = pipeline.fit(data).transform(data)
    result.select("spell.result").show(false)
  }

  it should "NerDLModel" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.annotators.ner.NerConverter
    import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
    import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    // First extract the prerequisites for the NerCrfModel
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    val embeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("bert")

    // Then NER can be extracted
    val nerTagger = NerDLModel
      .pretrained()
      .setInputCols("sentence", "token", "bert")
      .setOutputCol("ner")

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, sentence, tokenizer, embeddings, nerTagger))

    val data = Seq("U.N. official Ekeus heads for Baghdad.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    //    result.select("ner.result").show(false)

    val selected_results = Annotation.collect(result, "sentence", "token", "ner")

    println(selected_results.mkString("Row(", ", ", ")"))

    val converter = new NerConverter()
      .setInputCols("sentence", "token", "ner")
      .setOutputCol("entities")
      .setPreservePosition(false)
      .setWhiteList("ORG", "LOC")

    converter.transform(result).selectExpr("explode(entities)").show(false)
  }

  it should "NerDLApproach" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
    import com.johnsnowlabs.nlp.embeddings.BertEmbeddings
    import com.johnsnowlabs.nlp.training.CoNLL
    import org.apache.spark.ml.Pipeline

    val embeddings = BertEmbeddings
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

    // Then the training can start
    val nerTagger = new NerDLApproach()
      .setInputCols("sentence", "token", "embeddings")
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(1)
      .setRandomSeed(0)
      .setVerbose(0)

    val pipeline = new Pipeline().setStages(Array(embeddings, nerTagger))

    // We use the text and labels from the CoNLL dataset
    val conll = CoNLL()
    val trainingData =
      conll.readDataset(spark, "src/test/resources/conll2003/eng.train").limit(10)

    val pipelineModel = pipeline.fit(trainingData)
  }

  it should "NerCrfModel" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator.{SentenceDetector, Tokenizer}
    import com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel
    import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    // First extract the prerequisites for the NerCrfModel
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    val embeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("word_embeddings")

    val posTagger = PerceptronModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("pos")

    // Then NER can be extracted
    val nerTagger = NerCrfModel
      .load("/home/ducha/Workspace/Training/NerCrfApproach/ner")
      .setInputCols("sentence", "token", "word_embeddings", "pos")
      .setOutputCol("ner")

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, sentence, tokenizer, embeddings, posTagger, nerTagger))

    val data = Seq("U.N. official Ekeus heads for Baghdad.").toDF("text")
    val result = pipeline
      .fit(data)
      .transform(data)

    result.select("ner.result").show(false)
  }

  it should "NerCrfApproach" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator.NerCrfApproach
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
    import com.johnsnowlabs.nlp.training.CoNLL
    import org.apache.spark.ml.Pipeline

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val nerTagger = new NerCrfApproach()
      .setInputCols("sentence", "token", "pos", "embeddings")
      .setLabelColumn("label")
      .setMinEpochs(1)
      .setMaxEpochs(3)
      .setOutputCol("ner")

    val pipeline = new Pipeline().setStages(
      Array(
        //      documentAssembler,
        embeddings,
        nerTagger))

    // This CoNLL dataset already includes the sentence, token, pos and label column.
    // If a custom dataset is used, these will need to be extracted.
    val conll = CoNLL()
    val trainingData =
      conll.readDataset(spark, "src/test/resources/conll2003/eng.train").limit(10)
    //      .withColumn("id", monotonically_increasing_id)

    val pipelineModel = pipeline.fit(trainingData)
    pipelineModel.transform(trainingData.limit(3)).select("ner.result").show(truncate = false)

    //    for (i <- 0 to 100) {
    //      val td = trainingData.filter(s"id == $i")
    //      val pipelineModel = pipeline.fit(td)
    //      val res = pipelineModel.transform(trainingData).select("ner.result").collect()
    //      if (res.forall((r: org.apache.spark.sql.Row) => r.getString(0).contains("Start")))
    //        println("error at index", i)
    //    }
  }

  //  it should "yake" taggedAs SlowTest ignore {
  //    import spark.implicits._
  //    import com.johnsnowlabs.nlp.base.DocumentAssembler
  //    import com.johnsnowlabs.nlp.annotator.{SentenceDetector, Tokenizer}
  //    import com.johnsnowlabs.nlp.annotators.keyword.yake.YakeModel
  //    import org.apache.spark.ml.Pipeline
  //
  //    val documentAssembler = new DocumentAssembler()
  //      .setInputCol("text")
  //      .setOutputCol("document")
  //
  //    val sentenceDetector = new SentenceDetector()
  //      .setInputCols("document")
  //      .setOutputCol("sentence")
  //
  //    val token = new Tokenizer()
  //      .setInputCols("sentence")
  //      .setOutputCol("token")
  //      .setContextChars(Array("(", ")", "?", "!", ".", ","))
  //
  //    val keywords = new YakeModel()
  //      .setInputCols("token")
  //      .setOutputCol("keywords")
  //      .setThreshold(0.6f)
  //      .setMinNGrams(2)
  //      .setNKeywords(10)
  //
  //    val pipeline = new Pipeline().setStages(Array(
  //      documentAssembler,
  //      sentenceDetector,
  //      token,
  //      keywords
  //    ))
  //
  //    val data = Seq(
  //      "Sources tell us that Google is acquiring Kaggle, a platform that hosts data science and machine learning competitions. Details about the transaction remain somewhat vague, but given that Google is hosting its Cloud Next conference in San Francisco this week, the official announcement could come as early as tomorrow. Reached by phone, Kaggle co-founder CEO Anthony Goldbloom declined to deny that the acquisition is happening. Google itself declined 'to comment on rumors'. Kaggle, which has about half a million data scientists on its platform, was founded by Goldbloom  and Ben Hamner in 2010. The service got an early start and even though it has a few competitors like DrivenData, TopCoder and HackerRank, it has managed to stay well ahead of them by focusing on its specific niche. The service is basically the de facto home for running data science and machine learning competitions. With Kaggle, Google is buying one of the largest and most active communities for data scientists - and with that, it will get increased mindshare in this community, too (though it already has plenty of that thanks to Tensorflow and other projects). Kaggle has a bit of a history with Google, too, but that's pretty recent. Earlier this month, Google and Kaggle teamed up to host a $100,000 machine learning competition around classifying YouTube videos. That competition had some deep integrations with the Google Cloud Platform, too. Our understanding is that Google will keep the service running - likely under its current name. While the acquisition is probably more about Kaggle's community than technology, Kaggle did build some interesting tools for hosting its competition and 'kernels', too. On Kaggle, kernels are basically the source code for analyzing data sets and developers can share this code on the platform (the company previously called them 'scripts'). Like similar competition-centric sites, Kaggle also runs a job board, too. It's unclear what Google will do with that part of the service. According to Crunchbase, Kaggle raised $12.5 million (though PitchBook says it's $12.75) since its   launch in 2010. Investors in Kaggle include Index Ventures, SV Angel, Max Levchin, Naval Ravikant, Google chief economist Hal Varian, Khosla Ventures and Yuri Milner"
  //    ).toDF("text")
  //    val result = pipeline.fit(data).transform(data)
  //
  //    // combine the result and score (contained in keywords.metadata)
  //    val scores = result
  //      .selectExpr("explode(arrays_zip(keywords.result, keywords.metadata)) as resultTuples")
  //      .select($"resultTuples.0" as "keyword", $"resultTuples.1.score")
  //
  //    // Order ascending, as lower scores means higher importance
  //    scores.orderBy("score").show(5, truncate = false)
  //  }

  it should "LanguageDetect" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.ld.dl.LanguageDetectorDL
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val languageDetector = LanguageDetectorDL
      .pretrained()
      .setInputCols("document")
      .setOutputCol("language")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, languageDetector))

    val data = Seq(
      "Spark NLP is an open-source text processing library for advanced natural language processing for the Python, Java and Scala programming languages.",
      "Spark NLP est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python, Java et Scala.",
      "Spark NLP ist eine Open-Source-Textverarbeitungsbibliothek für fortgeschrittene natürliche Sprachverarbeitung für die Programmiersprachen Python, Java und Scala.")
      .toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("language.result").show(false)
  }

  it should "SentimentDLModel" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator.UniversalSentenceEncoder
    import com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLModel
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val useEmbeddings = UniversalSentenceEncoder
      .pretrained()
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    val sentiment = SentimentDLModel
      .pretrained("sentimentdl_use_twitter")
      .setInputCols("sentence_embeddings")
      .setThreshold(0.7f)
      .setThresholdLabel("neutral")
      .setOutputCol("sentiment")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, useEmbeddings, sentiment))

    val data =
      Seq("Wow, the new video is awesome!", "bruh this sucks what a damn waste of time").toDF(
        "text")
    val result = pipeline.fit(data).transform(data)

    result.select("text", "sentiment.result").show(false)
  }

  it should "SentimentApproach" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator.UniversalSentenceEncoder
    import com.johnsnowlabs.nlp.annotators.classifier.dl.{SentimentDLApproach, SentimentDLModel}
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline

    val smallCorpus =
      spark.read.option("header", "true").csv("src/test/resources/classifier/sentiment.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val useEmbeddings = UniversalSentenceEncoder
      .pretrained()
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    val docClassifier = new SentimentDLApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol("sentiment")
      .setLabelColumn("label")
      .setBatchSize(32)
      .setMaxEpochs(1)
      .setLr(5e-3f)
      .setDropout(0.5f)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, useEmbeddings, docClassifier))

    val pipelineModel = pipeline.fit(smallCorpus)
    pipelineModel.stages.last
      .asInstanceOf[SentimentDLModel]
      .write
      .overwrite()
      .save("./tmp_sentimentDL_model")

    val pipelineDF = pipelineModel.transform(smallCorpus)
  }

  it should "MultiClassDLModel" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLModel
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val useEmbeddings = UniversalSentenceEncoder
      .pretrained()
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    val multiClassifierDl = MultiClassifierDLModel
      .pretrained("multiclassifierdl_use_toxic")
      .setInputCols("sentence_embeddings")
      .setOutputCol("classifications")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, useEmbeddings, multiClassifierDl))

    val data = Seq("This is pretty good stuff!", "Wtf kind of crap is this").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.select("text", "classifications.result").show(false)
  }

  it should "MultiClassDLApproach" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLApproach
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.sql.functions.{col, udf}

    // Process training data to create text with arrays of labels
    def splitAndTrim = udf { labels: String =>
      labels.split(", ").map(x => x.trim)
    }

    val smallCorpus = spark.read
      .option("header", value = true)
      .option("inferSchema", value = true)
      .option("mode", "DROPMALFORMED")
      .csv("src/test/resources/classifier/e2e.csv")
      .withColumn("labels", splitAndTrim(col("mr")))
      .withColumn("text", col("ref"))
      .drop("mr")

    // Then create pipeline for training
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
      .setCleanupMode("shrink")

    val embeddings = UniversalSentenceEncoder
      .pretrained()
      .setInputCols("document")
      .setOutputCol("embeddings")

    val docClassifier = new MultiClassifierDLApproach()
      .setInputCols("embeddings")
      .setOutputCol("category")
      .setLabelColumn("labels")
      .setBatchSize(128)
      .setMaxEpochs(10)
      .setLr(1e-3f)
      .setThreshold(0.5f)
      .setValidationSplit(0.1f)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, embeddings, docClassifier))

    val pipelineModel = pipeline.fit(smallCorpus)
  }

  it should "ClasisfierDLModel" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator.SentenceDetector
    import com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLModel
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val useEmbeddings = UniversalSentenceEncoder
      .pretrained()
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    val sarcasmDL = ClassifierDLModel
      .pretrained("classifierdl_use_sarcasm")
      .setInputCols("sentence_embeddings")
      .setOutputCol("sarcasm")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, useEmbeddings, sarcasmDL))

    val data = Seq(
      "I'm ready!",
      "If I could put into words how much I love waking up at 6 am on Mondays I would.").toDF(
      "text")
    val result = pipeline.fit(data).transform(data)

    result
      .selectExpr("explode(arrays_zip(sentence, sarcasm)) as result")
      .selectExpr("result.sentence.result as sentence", "result.sarcasm.result as sarcasm")
      .show(false)
  }

  it should "ClassifierDLApproach" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
    import org.apache.spark.ml.Pipeline

    val smallCorpus = ResourceHelper.spark.read
      .option("header", "true")
      .csv("src/test/resources/classifier/sentiment.csv")

    println("count of training dataset: ", smallCorpus.count)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val useEmbeddings = UniversalSentenceEncoder
      .pretrained()
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    val docClassifier = new ClassifierDLApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol("category")
      .setLabelColumn("label")
      .setBatchSize(64)
      .setMaxEpochs(20)
      .setLr(5e-3f)
      .setDropout(0.5f)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, useEmbeddings, docClassifier))

    val pipelineModel = pipeline.fit(smallCorpus)
  }

  it should "ChunkEmbeddings" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
    import com.johnsnowlabs.nlp.annotators.{NGramGenerator, Tokenizer}
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.{ChunkEmbeddings, WordEmbeddingsModel}
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    // Extract the Embeddings from the NGrams
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val nGrams = new NGramGenerator()
      .setInputCols("token")
      .setOutputCol("chunk")
      .setN(2)

    val embeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    // Convert the NGram chunks into Word Embeddings
    val chunkEmbeddings = new ChunkEmbeddings()
      .setInputCols("chunk", "embeddings")
      .setOutputCol("chunk_embeddings")
      .setPoolingStrategy("AVERAGE")

    val pipeline = new Pipeline()
      .setStages(
        Array(documentAssembler, sentence, tokenizer, nGrams, embeddings, chunkEmbeddings))

    val data = Seq("This is a sentence.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result
      .selectExpr("explode(chunk_embeddings) as result")
      .selectExpr("result.annotatorType", "result.result", "result.embeddings")
      .show(5, 80)
  }

  it should "Use" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.EmbeddingsFinisher
    import com.johnsnowlabs.nlp.annotator.SentenceDetector
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val embeddings = UniversalSentenceEncoder
      .pretrained()
      .setInputCols("sentence")
      .setOutputCol("sentence_embeddings")

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, embeddings, embeddingsFinisher))

    val data = Seq("This is a sentence.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  }

  it should "Roberta" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.EmbeddingsFinisher
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.RoBertaEmbeddings
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val embeddings = RoBertaEmbeddings
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(true)

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, embeddingsFinisher))

    val data =
      Seq((1, "Where was John Lennon born?"), (2, "Where was ... born?"), (3, "Or was he?"))
        .toDF("id", "text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("finished_embeddings").show(5, truncate = false)
    result.selectExpr("explode(document)").show(truncate = false)
  }

  it should "DistilBert" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.EmbeddingsFinisher
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.DistilBertEmbeddings
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = DistilBertEmbeddings
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(true)

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, embeddingsFinisher))

    val data = Seq("This is a sentence.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  }

  it should "Elmo" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.EmbeddingsFinisher
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.ElmoEmbeddings
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val embeddings = ElmoEmbeddings
      .pretrained()
      .setPoolingLayer("word_emb")
      .setInputCols("token", "document")
      .setOutputCol("embeddings")

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, tokenizer, embeddings, embeddingsFinisher))

    val data = Seq("This is a sentence.").toDF("text")
    val result = pipeline.fit(data).transform(data)
    result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  }

  it should "SentenceEmbeddings" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.EmbeddingsFinisher
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.{SentenceEmbeddings, WordEmbeddingsModel}
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")

    val embeddingsSentence = new SentenceEmbeddings()
      .setInputCols(Array("document", "embeddings"))
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("sentence_embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(
        Array(documentAssembler, tokenizer, embeddings, embeddingsSentence, embeddingsFinisher))

    val data = Seq("This is a sentence.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("finished_embeddings").show(5, 80)
  }

  it should "SentenceBert" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.EmbeddingsFinisher
    import com.johnsnowlabs.nlp.annotator.SentenceDetector
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val embeddings = BertSentenceEmbeddings
      .pretrained("sent_small_bert_L2_128")
      .setInputCols("sentence")
      .setOutputCol("sentence_bert_embeddings")
      .setMaxSentenceLength(32)

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("sentence_bert_embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, sentence, embeddings, embeddingsFinisher))

    val data = Seq("John loves apples. Mary loves oranges. John loves Mary.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  }

  it should "BertEmbeddings" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.EmbeddingsFinisher
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.BertEmbeddings
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val embeddings = BertEmbeddings
      .pretrained("bert_base_japanese", "ja")
      .setInputCols("token", "document")
      .setOutputCol("bert_embeddings")

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("bert_embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, tokenizer, embeddings, embeddingsFinisher))

    val data = Seq("This is a sentence.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  }

  it should "WordEmbeddingsModel" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.EmbeddingsFinisher
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, embeddingsFinisher))

    val data = Seq("This is a sentence.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(finished_embeddings) as result").show(5, 80)

    val wordsCoverage =
      WordEmbeddingsModel.withCoverageColumn(result, "embeddings", "cov_embeddings")
    wordsCoverage.select("text", "cov_embeddings").show(false)

    val wordsOverallCoverage =
      WordEmbeddingsModel.overallCoverage(wordsCoverage, "embeddings").percentage
    println(wordsOverallCoverage)
  }

  it should "WordEmbeddings" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.EmbeddingsFinisher
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.embeddings.WordEmbeddings
    import com.johnsnowlabs.nlp.util.io.ReadAs
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = new WordEmbeddings()
      .setStoragePath("src/test/resources/random_embeddings_dim4.txt", ReadAs.TEXT)
      .setStorageRef("glove_4d")
      .setDimension(4)
      .setInputCols("document", "token")
      .setOutputCol("embeddings")

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, embeddingsFinisher))

    val data = Seq("The patient was diagnosed with diabetes.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(finished_embeddings) as result").show(false)

  }
  it should "SentimentDetector" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.DocumentAssembler
    import com.johnsnowlabs.nlp.annotator.Tokenizer
    import com.johnsnowlabs.nlp.annotators.Lemmatizer
    import com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector
    import com.johnsnowlabs.nlp.util.io.ReadAs
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val lemmatizer = new Lemmatizer()
      .setInputCols("token")
      .setOutputCol("lemma")
      .setDictionary("src/test/resources/lemma-corpus-small/lemmas_small.txt", "->", "\t")

    val sentimentDetector = new SentimentDetector()
      .setInputCols("lemma", "document")
      .setOutputCol("sentimentScore")
      .setDictionary(
        "src/test/resources/sentiment-corpus/default-sentiment-dict.txt",
        ",",
        ReadAs.TEXT)
      .setEnableScore(true)

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, tokenizer, lemmatizer, sentimentDetector))

    val data = Seq(
      "The staff of the restaurant is nice",
      "I recommend others to avoid because it is too expensive").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("sentimentScore.result").show(false)
  }

  it should "vivekn" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.Finisher
    import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
    import com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentModel
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val token = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val normalizer = new Normalizer()
      .setInputCols("token")
      .setOutputCol("normal")

    val vivekn = ViveknSentimentModel
      .pretrained()
      .setInputCols("document", "normal")
      .setOutputCol("result_sentiment")

    val finisher = new Finisher()
      .setInputCols("result_sentiment")
      .setOutputCols("final_sentiment")

    val pipeline = new Pipeline().setStages(Array(document, token, normalizer, vivekn, finisher))

    val data = Seq("I recommend this movie", "Dont waste your time!!!").toDF("text")

    val pipelineModel = pipeline.fit(data)

    val result = pipelineModel.transform(data)
    result.select("final_sentiment").show(false)
  }

  it should "POSTagger pretrained" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val posTagger = PerceptronModel
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("pos")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, posTagger))

    val data = Seq("Peter Pipers employees are picking pecks of pickled peppers").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(pos) as pos").show(false)
  }

  it should "POS" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.training.POS

    val pos = POS()
    val path = "src/test/resources/anc-pos-corpus-small/test-training.txt"
    val posDf = pos.readDataset(spark, path)

    posDf.printSchema

    posDf.selectExpr("explode(tags) as tags").show(false)
  }
  it should "POSTagger" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator.SentenceDetector
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.training.POS
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    val datasetPath = "src/test/resources/anc-pos-corpus-small/test-training.txt"
    val trainingPerceptronDF = POS().readDataset(spark, datasetPath)

    val trainedPos = new PerceptronApproach()
      .setInputCols("document", "token")
      .setOutputCol("pos")
      .setPosColumn("tags")
      .fit(trainingPerceptronDF)

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, sentence, tokenizer, trainedPos))

    val data = Seq("To be or not to be, is this the question?").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("pos.result").show(false)
  }
  it should "SentenceDetectorDL" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator.SentenceDetector
    import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentences")

    val sentenceDL = SentenceDetectorDLModel
      .pretrained("sentence_detector_dl", "en")
      .setInputCols("document")
      .setOutputCol("sentencesDL")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, sentence, sentenceDL))

    val data = Seq("""John loves Mary.Mary loves Peter
      Peter loves Helen .Helen loves John;
      Total: four people involved.""").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(sentences.result) as sentences").show(false)
    result.selectExpr("explode(sentencesDL.result) as sentencesDL").show(false)
  }

  it should "SentenceDetector" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator.SentenceDetector
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, sentence))

    val data = Seq("This is my first sentence. This my second. How about a third?").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(sentence)").show(false)
  }

  it should "MultiDateMatcher" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.MultiDateMatcher
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val date = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setAnchorDateYear(2020)
      .setAnchorDateMonth(1)
      .setAnchorDateDay(11)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, date))

    val data = Seq("I saw him yesterday and he told me that he will visit us next week")
      .toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(date) as dates").show(false)
  }

  it should "DateMatcher" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotators.DateMatcher
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val date = new DateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setAnchorDateYear(2020)
      .setAnchorDateMonth(1)
      .setAnchorDateDay(11)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, date))

    val data = Seq("Fri, 21 Nov 1997", "next week at 7.30", "see you a day after").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("date").show(false)
  }

  it should "NGramGenerate" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.annotator.SentenceDetector
    import com.johnsnowlabs.nlp.annotators.{NGramGenerator, Tokenizer}
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val nGrams = new NGramGenerator()
      .setInputCols("token")
      .setOutputCol("ngrams")
      .setN(2)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, sentence, tokenizer, nGrams))

    val data = Seq("This is my sentence.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(ngrams) as result").show(false)
  }

  it should "Chunk" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.DocumentAssembler
    import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
    import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
    import com.johnsnowlabs.nlp.annotators.{Chunker, Tokenizer}
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val POSTag = PerceptronModel
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("pos")

    val chunker = new Chunker()
      .setInputCols("sentence", "pos")
      .setOutputCol("chunk")
      .setRegexParsers(Array("<NNP>+", "<NNS>+"))

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, tokenizer, POSTag, chunker))

    val data = Seq("Peter Pipers employees are picking pecks of pickled peppers.").toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(chunk)").show(false)
  }

  it should "BigTextMatcher" taggedAs SlowTest ignore {
    /*
    ...
    dolore magna aliqua
    lorem ipsum dolor. sit
    laborum
    ...
     */

    import com.johnsnowlabs.nlp.DocumentAssembler
    import com.johnsnowlabs.nlp.annotator.{BigTextMatcher, Tokenizer}
    import com.johnsnowlabs.nlp.util.io.ReadAs
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val data = Seq("Hello dolore magna aliqua. Lorem ipsum dolor. sit in laborum").toDF("text")
    val entityExtractor = new BigTextMatcher()
      .setInputCols("document", "token")
      .setStoragePath("src/test/resources/entity-extractor/test-phrases.txt", ReadAs.TEXT)
      .setOutputCol("entity")
      .setCaseSensitive(false)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityExtractor))
    val results = pipeline.fit(data).transform(data)
    results.selectExpr("explode(entity)").show(false)
  }

  it should "TextMatcher" taggedAs SlowTest ignore {
    /*
    ...
    dolore magna aliqua
    lorem ipsum dolor. sit
    laborum
    ...
     */

    import com.johnsnowlabs.nlp.DocumentAssembler
    import com.johnsnowlabs.nlp.annotator.{TextMatcher, Tokenizer}
    import com.johnsnowlabs.nlp.util.io.ReadAs
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val data = Seq("Hello dolore magna aliqua. Lorem ipsum dolor. sit in laborum").toDF("text")
    val entityExtractor = new TextMatcher()
      .setInputCols("document", "token")
      .setEntities("src/test/resources/entity-extractor/test-phrases.txt", ReadAs.TEXT)
      .setOutputCol("entity")
      .setCaseSensitive(false)
    //      .setTokenizer(tokenizer.fit(data))

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityExtractor))
    val results = pipeline.fit(data).transform(data)
    results.selectExpr("explode(entity)").show(false)
  }

  it should "RegexMatcher" taggedAs SlowTest ignore {
    import ResourceHelper.spark.implicits._
    import com.johnsnowlabs.nlp.annotator.SentenceDetector
    import com.johnsnowlabs.nlp.annotators.RegexMatcher
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import org.apache.spark.ml.Pipeline

    val sampleDataset = Seq(
      "My first sentence with the first rule. This is my second sentence with ceremonies rule.")
      .toDF("text")

    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")

    val sentence = new SentenceDetector().setInputCols("document").setOutputCol("sentence")

    val regexMatcher = new RegexMatcher()
      .setExternalRules("src/test/resources/regex-matcher/rules.txt", ",")
      .setInputCols(Array("sentence"))
      .setOutputCol("regex")
      .setStrategy("MATCH_ALL")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, sentence, regexMatcher))

    val results = pipeline.fit(sampleDataset).transform(sampleDataset)
    results.selectExpr("explode(regex) as result").show(false)
  }

  it should "StopWords" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.DocumentAssembler
    import com.johnsnowlabs.nlp.annotator.{SentenceDetector, Tokenizer}
    import com.johnsnowlabs.nlp.annotators.StopWordsCleaner
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val stopWords = new StopWordsCleaner()
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setCaseSensitive(false)

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, stopWords))

    val data = Seq(
      "This is my first sentence. This is my second.",
      "This is my third sentence. This is my forth.").toDF("text")
    val result = pipeline.fit(data).transform(data)
    result.selectExpr("cleanTokens.result").show(false)
  }

  it should "Lemma" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.DocumentAssembler
    import com.johnsnowlabs.nlp.annotator.{SentenceDetector, Tokenizer}
    import com.johnsnowlabs.nlp.annotators.Lemmatizer
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val lemmatizer = new Lemmatizer()
      .setInputCols(Array("token"))
      .setOutputCol("lemma")
      .setDictionary("src/test/resources/lemma-corpus-small/lemmas_small.txt", "->", "\t")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentenceDetector, tokenizer, lemmatizer))

    val data = Seq("Peter Pipers employees are picking pecks of pickled peppers.")
      .toDF("text")

    val result = pipeline.fit(data).transform(data)
    result.selectExpr("lemma.result").show(false)
  }

  "Example" should "Stem" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.DocumentAssembler
    import com.johnsnowlabs.nlp.annotator.{Stemmer, Tokenizer}
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val stemmer = new Stemmer()
      .setInputCols("token")
      .setOutputCol("stem")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, stemmer))

    val data = Seq("Peter Pipers employees are picking pecks of pickled peppers.")
      .toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("stem.result").show(truncate = false)
  }

  "Example" should "Normalize" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.DocumentAssembler
    import com.johnsnowlabs.nlp.annotator.{Normalizer, Tokenizer}
    import org.apache.spark.ml.Pipeline
    import spark.implicits._
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val normalizer = new Normalizer()
      .setInputCols("token")
      .setOutputCol("normalized")
      .setLowercase(true)
      .setCleanupPatterns(Array("""[^\w\d\s]""")) // remove punctuations (keep alphanumeric chars)
    // if we don't set CleanupPatterns, it will only keep alphabet letters ([^A-Za-z])

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, normalizer))

    val data =
      Seq("John and Peter are brothers. However they don't support each other that much.")
        .toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("normalized.result").show(truncate = false)
  }

  "Example" should "DocumentNormalise" taggedAs SlowTest ignore {
    import com.johnsnowlabs.nlp.DocumentAssembler
    import com.johnsnowlabs.nlp.annotator.DocumentNormalizer
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val cleanUpPatterns = Array("<[^>]*>")

    val documentNormalizer = new DocumentNormalizer()
      .setInputCols("document")
      .setOutputCol("normalizedDocument")
      .setAction("clean")
      .setPatterns(cleanUpPatterns)
      .setReplacement(" ")
      .setPolicy("pretty_all")
      .setLowercase(true)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, documentNormalizer))

    val text =
      """
    <div id="theworldsgreatest" class='my-right my-hide-small my-wide toptext' style="font-family:'Segoe UI',Arial,sans-serif">
      THE WORLD'S LARGEST WEB DEVELOPER SITE
      <h1 style="font-size:300%;">THE WORLD'S LARGEST WEB DEVELOPER SITE</h1>
      <p style="font-size:160%;">Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum..</p>
    </div>

    </div>"""
    val data = Seq(text).toDF("text")
    val pipelineModel = pipeline.fit(data)
    val result = pipelineModel.transform(data)
    result.selectExpr("normalizedDocument.result").show(truncate = false)
  }

  "Example" should "Embeddingsfinish" ignore {
    import com.johnsnowlabs.nlp.annotator.{
      Normalizer,
      StopWordsCleaner,
      Tokenizer,
      WordEmbeddingsModel
    }
    import com.johnsnowlabs.nlp.{DocumentAssembler, EmbeddingsFinisher}
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val normalizer = new Normalizer()
      .setInputCols("token")
      .setOutputCol("normalized")

    val stopwordsCleaner = new StopWordsCleaner()
      .setInputCols("normalized")
      .setOutputCol("cleanTokens")
      .setCaseSensitive(false)

    val gloveEmbeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("document", "cleanTokens")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val embeddingsFinisher = new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_sentence_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val data = Seq("Spark NLP is an open-source text processing library.")
      .toDF("text")
    val pipeline = new Pipeline()
      .setStages(
        Array(
          documentAssembler,
          tokenizer,
          normalizer,
          stopwordsCleaner,
          gloveEmbeddings,
          embeddingsFinisher))
      .fit(data)

    val result = pipeline.transform(data)
    val resultWithSize = result
      .selectExpr("explode(finished_sentence_embeddings)")
      .map { row =>
        val vector = row.getAs[org.apache.spark.ml.linalg.DenseVector](0)
        (vector.size, vector)
      }
      .toDF("size", "vector")

    resultWithSize.show(5, 80)
  }

  "Example" should "Finish" ignore {
    import com.johnsnowlabs.nlp.Finisher
    import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
    import spark.implicits._

    val data =
      Seq((1, "New York and New Jersey aren't that far apart actually.")).toDF("id", "text")

    // Extracts Named Entities amongst other things
    val pipeline = PretrainedPipeline("explain_document_dl")

    val finisher = new Finisher().setInputCols("entities").setOutputCols("output")
    val explainResult = pipeline.transform(data)

    val result = finisher.transform(explainResult)
    result.select("output").show(false)
  }

//  "Example" should "Chunk2Doc" ignore {
//    import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
//    import spark.implicits._
//
//    val data =
//      Seq((1, "New York and New Jersey aren't that far apart actually.")).toDF("id", "text")
//
//    // Extracts Named Entities amongst other things
//    val pipeline = PretrainedPipeline("explain_document_dl")
//
//    val chunkToDoc = new Chunk2Doc().setInputCols("entities").setOutputCol("chunkConverted")
//    val explainResult = pipeline.transform(data)
//
//    val result = chunkToDoc.transform(explainResult)
//    result.selectExpr("explode(chunkConverted)").show(false)
//  }

  "Example" should "Doc2Chunk" ignore {
    import com.johnsnowlabs.nlp.{Doc2Chunk, DocumentAssembler}
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val chunkAssembler = new Doc2Chunk()
      .setInputCols("document")
      .setChunkCol("target")
      .setOutputCol("chunk")
      .setIsArray(true)

    val data = Seq(
      (
        "Spark NLP is an open-source text processing library for advanced natural language processing.",
        Seq("Spark NLP", "text processing library", "natural language processing")))
      .toDF("text", "target")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, chunkAssembler)).fit(data)
    val result = pipeline.transform(data)

    result.selectExpr("chunk.result", "chunk.annotatorType").show(false)
  }

  "Example" should "DocumentAssemble" ignore {
    import com.johnsnowlabs.nlp.DocumentAssembler
    import spark.implicits._
    val data = Seq("Spark NLP is an open-source text processing library.")
      .toDF("text")
    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")

    val result = documentAssembler.transform(data)
    result.select("document").show(false)
    result.select("document").printSchema
  }

  "Example" should "Tokenizer" ignore {
    import com.johnsnowlabs.nlp.DocumentAssembler
    import com.johnsnowlabs.nlp.annotators.Tokenizer
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val data = Seq(
      "Spark NLP is an open-source text processing library for advanced natural language processing.")
      .toDF("text")
    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").fit(data)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer)).fit(data)
    val result = pipeline.transform(data)
    result.selectExpr("token.resultDocumentAssembler").show(false)
  }

  "Example" should "TokenizerAssemble" ignore {
    import com.johnsnowlabs.nlp.{DocumentAssembler, TokenAssembler}
    import com.johnsnowlabs.nlp.annotator.{
      Normalizer,
      SentenceDetector,
      StopWordsCleaner,
      Tokenizer
    }
    import org.apache.spark.ml.Pipeline
    import spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentences")

    val tokenizer = new Tokenizer()
      .setInputCols("sentences")
      .setOutputCol("token")

    val normalizer = new Normalizer()
      .setInputCols("token")
      .setOutputCol("normalized")
      .setLowercase(false)

    val stopwordsCleaner = new StopWordsCleaner()
      .setInputCols("normalized")
      .setOutputCol("cleanTokens")
      .setCaseSensitive(false)

    val tokenAssembler = new TokenAssembler()
      .setInputCols("sentences", "cleanTokens")
      .setOutputCol("cleanText")

    val data = Seq(
      "Spark NLP is an open-source text processing library for advanced natural language processing.")
      .toDF("text")

    val pipeline = new Pipeline()
      .setStages(
        Array(
          documentAssembler,
          sentenceDetector,
          tokenizer,
          normalizer,
          stopwordsCleaner,
          tokenAssembler))
      .fit(data)

    val result = pipeline.transform(data)
    result.select("cleanText").show(false)
  }
}
