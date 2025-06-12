package com.johnsnowlabs.nlp.serialization

import com.johnsnowlabs.collections.SearchTrie
import com.johnsnowlabs.ml.crf._
import com.johnsnowlabs.ml.tensorflow.{ClassifierDatasetEncoderParams, DatasetEncoderParams}
import com.johnsnowlabs.nlp.annotators.TokenizerModel
import com.johnsnowlabs.nlp.annotators.er.{
  AhoCorasickAutomaton,
  EntityPattern,
  EntityRulerFeatures,
  FlattenEntityPattern
}
import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition.DependencyMaker
import com.johnsnowlabs.nlp.annotators.parser.dep.Tagger
import com.johnsnowlabs.nlp.annotators.parser.typdep.{DependencyPipe, Options, Parameters}
import com.johnsnowlabs.nlp.annotators.pos.perceptron.AveragedPerceptron
import com.johnsnowlabs.nlp.util.io.MatchStrategy
import com.johnsnowlabs.nlp.util.regex.{RuleFactory, TransformStrategy}
import com.johnsnowlabs.nlp.{
  AnnotatorModel,
  AnnotatorType,
  HasFeatures,
  ParamsAndFeaturesReadable
}
import com.johnsnowlabs.tags.SlowTest
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers._

import java.lang.reflect.{Field, Modifier}

class MockFeaturesModel(override val uid: String)
    extends AnnotatorModel[MockFeaturesModel]
    with HasFeatures {
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(AnnotatorType.DUMMY)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.DUMMY

  def this() = this("MockFeaturesModel")

  // All Array Features
  val arrayTupleStringString =
    new ArrayFeature[(String, String)](this, "arrayTupleStringString")
  val arrayString =
    new ArrayFeature[String](this, "arrayString")

  // All MapFeatures currently in use
  val mapIntTupleInt = new MapFeature[Int, (Int, Int)](this, "mapIntTupleInt")
  val mapIntString = new MapFeature[Int, String](this, "mapIntString")
  val mapStringArrayFloat =
    new MapFeature[String, Array[Float]](this, "mapStringArrayFloat")
  val mapStringArrayString =
    new MapFeature[String, Array[String]](this, "mapStringArrayString")
  val mapStringBigInt = new MapFeature[String, BigInt](this, "mapStringBigInt")
  val mapStringDouble = new MapFeature[String, Double](this, "mapStringDouble")
  val mapStringInt = new MapFeature[String, Int](this, "mapStringInt")
  val mapStringLong = new MapFeature[String, Long](this, "mapStringLong")
  val mapStringMapStringFloat =
    new MapFeature[String, Map[String, Float]](this, "mapStringMapStringFloat")
  val mapStringString = new MapFeature[String, String](this, "mapStringString")

  val structAveragedPerceptron =
    new StructFeature[AveragedPerceptron](this, "structAveragedPerceptron")
  val structClassifierDatasetEncoderParams = new StructFeature[ClassifierDatasetEncoderParams](
    this,
    "structClassifierDatasetEncoderParams")
  val structDatasetEncoderParams =
    new StructFeature[DatasetEncoderParams](this, "structDatasetEncoderParams")
  val structDependencyMaker = new StructFeature[DependencyMaker](this, "structDependencyMaker")
  val structDependencyPipe = new StructFeature[DependencyPipe](this, "structDependencyPipe")
  val structEntityRulerFeatures =
    new StructFeature[EntityRulerFeatures](this, "structEntityRulerFeatures")
  val structLinearChainCrfModel =
    new StructFeature[LinearChainCrfModel](this, "structLinearChainCrfModel")
  val structMapIntFloat = new StructFeature[Map[Int, Float]](this, "structMapIntFloat")
  val structMapStringFloat = new StructFeature[Map[String, Float]](this, "structMapStringFloat")
  val structOptionAhoCorasickAutomaton =
    new StructFeature[Option[AhoCorasickAutomaton]](this, "structOptionAhoCorasickAutomaton")
  val structOptionMapStringInt =
    new StructFeature[Option[Map[String, Int]]](this, "structOptionMapStringInt")
  val structOptions = new StructFeature[Options](this, "structOptions")
  val structParameters = new StructFeature[Parameters](this, "structParameters")
  val structRuleFactory = new StructFeature[RuleFactory](this, "structRuleFactory")
  val structSearchTrie = new StructFeature[SearchTrie](this, "structSearchTrie")
  val structString = new StructFeature[String](this, "structString")
  val structTokenizerModel = new StructFeature[TokenizerModel](this, "structTokenizerModel")

  // Getters
  def getArrayTupleStringString: Array[(String, String)] = arrayTupleStringString.getOrDefault
  def getArrayString: Array[String] = arrayString.getOrDefault
  def getMapIntTupleInt: Map[Int, (Int, Int)] = mapIntTupleInt.getOrDefault
  def getMapIntString: Map[Int, String] = mapIntString.getOrDefault
  def getMapStringArrayFloat: Map[String, Array[Float]] = mapStringArrayFloat.getOrDefault
  def getMapStringArrayString: Map[String, Array[String]] = mapStringArrayString.getOrDefault
  def getMapStringBigInt: Map[String, BigInt] = mapStringBigInt.getOrDefault
  def getMapStringDouble: Map[String, Double] = mapStringDouble.getOrDefault
  def getMapStringInt: Map[String, Int] = mapStringInt.getOrDefault
  def getMapStringLong: Map[String, Long] = mapStringLong.getOrDefault
  def getMapStringMapStringFloat: Map[String, Map[String, Float]] =
    mapStringMapStringFloat.getOrDefault
  def getMapStringString: Map[String, String] = mapStringString.getOrDefault

  // StructFeature Getters
  def getStructAveragedPerceptron: AveragedPerceptron = structAveragedPerceptron.getOrDefault
  def getStructClassifierDatasetEncoderParams: ClassifierDatasetEncoderParams =
    structClassifierDatasetEncoderParams.getOrDefault
  def getStructDatasetEncoderParams: DatasetEncoderParams =
    structDatasetEncoderParams.getOrDefault
  def getStructDependencyMaker: DependencyMaker = structDependencyMaker.getOrDefault
  def getStructDependencyPipe: DependencyPipe = structDependencyPipe.getOrDefault
  def getStructEntityRulerFeatures: EntityRulerFeatures = structEntityRulerFeatures.getOrDefault
  def getStructLinearChainCrfModel: LinearChainCrfModel = structLinearChainCrfModel.getOrDefault
  def getStructMapIntFloat: Map[Int, Float] = structMapIntFloat.getOrDefault
  def getStructMapStringFloat: Map[String, Float] = structMapStringFloat.getOrDefault
  def getStructOptionAhoCorasickAutomaton: Option[AhoCorasickAutomaton] =
    structOptionAhoCorasickAutomaton.getOrDefault
  def getStructOptionMapStringInt: Option[Map[String, Int]] =
    structOptionMapStringInt.getOrDefault
  def getStructOptions: Options = structOptions.getOrDefault
  def getStructParameters: Parameters = structParameters.getOrDefault
  def getStructRuleFactory: RuleFactory = structRuleFactory.getOrDefault
  def getStructSearchTrie: SearchTrie = structSearchTrie.getOrDefault
  def getStructString: String = structString.getOrDefault
  def getStructTokenizerModel: TokenizerModel = structTokenizerModel.getOrDefault

  // Setters
  def setArrayTupleStringString(value: Array[(String, String)]): this.type =
    set(arrayTupleStringString, value)
  def setArrayString(value: Array[String]): this.type = set(arrayString, value)
  def setMapIntTupleInt(value: Map[Int, (Int, Int)]): this.type = set(mapIntTupleInt, value)
  def setMapIntString(value: Map[Int, String]): this.type = set(mapIntString, value)
  def setMapStringArrayFloat(value: Map[String, Array[Float]]): this.type =
    set(mapStringArrayFloat, value)
  def setMapStringArrayString(value: Map[String, Array[String]]): this.type =
    set(mapStringArrayString, value)
  def setMapStringBigInt(value: Map[String, BigInt]): this.type = set(mapStringBigInt, value)
  def setMapStringDouble(value: Map[String, Double]): this.type = set(mapStringDouble, value)
  def setMapStringInt(value: Map[String, Int]): this.type = set(mapStringInt, value)
  def setMapStringLong(value: Map[String, Long]): this.type = set(mapStringLong, value)
  def setMapStringMapStringFloat(value: Map[String, Map[String, Float]]): this.type =
    set(mapStringMapStringFloat, value)
  def setMapStringString(value: Map[String, String]): this.type = set(mapStringString, value)

  // StructFeature Setters
  def setStructAveragedPerceptron(value: AveragedPerceptron): this.type =
    set(structAveragedPerceptron, value)
  def setStructClassifierDatasetEncoderParams(value: ClassifierDatasetEncoderParams): this.type =
    set(structClassifierDatasetEncoderParams, value)
  def setStructDatasetEncoderParams(value: DatasetEncoderParams): this.type =
    set(structDatasetEncoderParams, value)
  def setStructDependencyMaker(value: DependencyMaker): this.type =
    set(structDependencyMaker, value)
  def setStructDependencyPipe(value: DependencyPipe): this.type = set(structDependencyPipe, value)
  def setStructEntityRulerFeatures(value: EntityRulerFeatures): this.type =
    set(structEntityRulerFeatures, value)
  def setStructLinearChainCrfModel(value: LinearChainCrfModel): this.type =
    set(structLinearChainCrfModel, value)
  def setStructMapIntFloat(value: Map[Int, Float]): this.type = set(structMapIntFloat, value)
  def setStructMapStringFloat(value: Map[String, Float]): this.type =
    set(structMapStringFloat, value)
  def setStructOptionAhoCorasickAutomaton(value: Option[AhoCorasickAutomaton]): this.type =
    set(structOptionAhoCorasickAutomaton, value)
  def setStructOptionMapStringInt(value: Option[Map[String, Int]]): this.type =
    set(structOptionMapStringInt, value)
  def setStructOptions(value: Options): this.type = set(structOptions, value)
  def setStructParameters(value: Parameters): this.type = set(structParameters, value)
  def setStructRuleFactory(value: RuleFactory): this.type = set(structRuleFactory, value)
  def setStructSearchTrie(value: SearchTrie): this.type = set(structSearchTrie, value)
  def setStructString(value: String): this.type = set(structString, value)
  def setStructTokenizerModel(value: TokenizerModel): this.type = set(structTokenizerModel, value)
}

object MockFeaturesModel extends ParamsAndFeaturesReadable[MockFeaturesModel] {}

class FeaturesProtoTestSpec extends AnyFlatSpec {
  val dummyArrayTupleStringString: Array[(String, String)] = Array(("a", "b"), ("c", "d"))
  val dummyArrayString: Array[String] = Array("foo", "bar")
  val dummyMapIntTupleInt: Map[Int, (Int, Int)] = Map(1 -> (2, 3), 4 -> (5, 6))
  val dummyMapIntString: Map[Int, String] = Map(1 -> "one", 2 -> "two")
  val dummyMapStringArrayFloat: Map[String, Array[Float]] = Map("a" -> Array(1.0f, 2.0f))
  val dummyMapStringArrayString: Map[String, Array[String]] = Map("a" -> Array("b", "c"))
  val dummyMapStringBigInt: Map[String, BigInt] = Map("big" -> BigInt(1234567890))
  val dummyMapStringDouble: Map[String, Double] = Map("pi" -> 3.14)
  val dummyMapStringInt: Map[String, Int] = Map("one" -> 1, "two" -> 2)
  val dummyMapStringLong: Map[String, Long] = Map("long" -> 123456789L)
  val dummyMapStringMapStringFloat: Map[String, Map[String, Float]] = Map(
    "outer" -> Map("inner" -> 1.23f))
  val dummyMapStringString: Map[String, String] = Map("hello" -> "world")

  val dummyMapStringMapStringDouble: Map[String, Map[String, Double]] = Map(
    "outer" -> Map(("inner" -> 1.23d)))
  val dummyAveragedPerceptron: AveragedPerceptron =
    AveragedPerceptron(dummyArrayString, dummyMapStringString, dummyMapStringMapStringDouble)
  val dummyClassifierDatasetEncoderParams: ClassifierDatasetEncoderParams =
    ClassifierDatasetEncoderParams(dummyArrayString)
  val dummyDatasetEncoderParams: DatasetEncoderParams =
    DatasetEncoderParams(dummyArrayString.toList, "test".toList, List.empty, 123, "test")
  val dummyDependencyMaker = new DependencyMaker(
    new Tagger(dummyArrayString.toVector, dummyMapStringInt))
  val dummyDependencyPipe: DependencyPipe = createDependencyPipeForTesting(new Options())
  val dummyArrayFloat: Array[Float] = Array(1.0f, 2.0f)
  val dummyLinearChainCrfModel = new LinearChainCrfModel(
    dummyArrayFloat,
    new DatasetMetadata(
      Array("A", "B"),
      Array(Attr(0, "attr1"), Attr(1, "attr2")),
      Array(AttrFeature(0, 0, 0), AttrFeature(1, 1, 1)),
      Array(Transition(0, 1), Transition(1, 0)),
      Array(AttrStat(1, 1.0f), AttrStat(2, 2.0f), AttrStat(3, 3.0f), AttrStat(4, 4.0f))))
  val dummyAhoCorasickAutomaton = new AhoCorasickAutomaton(
    "test",
    Array(EntityPattern("test", Seq("test"), Some("test"), Some(true))),
    false)
  val dummyVectorTupleIntBoolIntInt: Vector[(Int, Boolean, Int, Int)] = Vector((1, true, 3, 4))
  val dummyMapTupleIntInt: Map[(Int, Int), Int] = Map((1, 2) -> 3)

  def setupMockFeaturesModel(): MockFeaturesModel = {
    val mockModel = new MockFeaturesModel
    // Fill with dummy data
    mockModel.setArrayTupleStringString(dummyArrayTupleStringString)
    mockModel.setArrayString(dummyArrayString)
    mockModel.setMapIntTupleInt(dummyMapIntTupleInt)
    mockModel.setMapIntString(dummyMapIntString)
    mockModel.setMapStringArrayFloat(dummyMapStringArrayFloat)
    mockModel.setMapStringArrayString(dummyMapStringArrayString)
    mockModel.setMapStringBigInt(dummyMapStringBigInt)
    mockModel.setMapStringDouble(dummyMapStringDouble)
    mockModel.setMapStringInt(dummyMapStringInt)
    mockModel.setMapStringLong(dummyMapStringLong)
    mockModel.setMapStringMapStringFloat(dummyMapStringMapStringFloat)
    mockModel.setMapStringString(dummyMapStringString)

    // Fill struct features with dummy data
    mockModel.setStructAveragedPerceptron(dummyAveragedPerceptron)
    mockModel.setStructClassifierDatasetEncoderParams(dummyClassifierDatasetEncoderParams)
    mockModel.setStructDatasetEncoderParams(dummyDatasetEncoderParams)
    mockModel.setStructDependencyMaker(dummyDependencyMaker)
    mockModel.setStructDependencyPipe(dummyDependencyPipe)
    mockModel.setStructEntityRulerFeatures(
      EntityRulerFeatures(
        dummyMapStringString,
        dummyMapStringArrayString.map { case ((str, strings)) => (str, strings.toSeq) }))
    mockModel.setStructLinearChainCrfModel(dummyLinearChainCrfModel)
    mockModel.setStructMapIntFloat(Map(1 -> 1.0f, 2 -> 2.0f))
    mockModel.setStructMapStringFloat(Map("a" -> 1.0f, "b" -> 2.0f))
    mockModel.setStructOptionAhoCorasickAutomaton(Some(dummyAhoCorasickAutomaton))
    mockModel.setStructOptionMapStringInt(Some(Map("a" -> 1, "b" -> 2)))
    mockModel.setStructOptions(new Options())
    mockModel.setStructParameters(new Parameters(dummyDependencyPipe, new Options()))
    mockModel.setStructRuleFactory(
      new RuleFactory(MatchStrategy.MATCH_ALL, TransformStrategy.NO_TRANSFORM))
    mockModel.setStructSearchTrie(
      new SearchTrie(dummyMapStringInt, dummyMapTupleIntInt, dummyVectorTupleIntBoolIntInt, true))
    mockModel.setStructString("dummy string")
    mockModel.setStructTokenizerModel(new TokenizerModel())

    mockModel
  }

  // Helper method to create DependencyPipe for testing
  private def createDependencyPipeForTesting(options: Options): DependencyPipe = {
    // Use reflection to access the package-private constructor
    val dependencyPipeClass = classOf[DependencyPipe]
    val constructor = dependencyPipeClass.getDeclaredConstructor(classOf[Options])
    constructor.setAccessible(true)
    val instance: DependencyPipe = constructor.newInstance(options)

    val typesField = dependencyPipeClass.getDeclaredField("types")
    typesField.setAccessible(true)
    typesField.set(instance, dummyArrayString)

    instance
  }

  private def assertSerializableFields(obj1: Any, obj2: Any): Unit = {
    def getAllNonTransientFields(obj: Any): Seq[Field] = {
      def getAllFieldsRecursively(c: Class[_]): Seq[Field] =
        if (c == null || c == classOf[Object]) Seq.empty
        else c.getDeclaredFields ++ getAllFieldsRecursively(c.getSuperclass)

      val nonTransientFields =
        getAllFieldsRecursively(obj.getClass).filter(f => !Modifier.isTransient(f.getModifiers))
      nonTransientFields.foreach(_.setAccessible(true))
      nonTransientFields
    }

    def getValues(obj: Any): Seq[Any] = getAllNonTransientFields(obj).map(_.get(obj))

    val values1 = getValues(obj1)
    val values2 = getValues(obj2)

    values1 shouldBe values2
  }

  // Maps
  private def assertSameMap[K, V](map1: Map[K, V], map2: Map[K, V]): Unit = map1 shouldBe map2

  // Maps with Arrays as values
  private def assertMapArray[K](map1: Map[K, Array[_]], map2: Map[K, Array[_]]): Unit =
    map1.zip(map2).map { case ((k1, v1), (k2, v2)) =>
      assert(k1 == k2)
      assert(v1 sameElements v2)
    }

  private def assertAveragedPerceptron(
      model: MockFeaturesModel,
      loadedModel: MockFeaturesModel): Unit = {
//    model.getStructAveragedPerceptron shouldEqual loadedModel.getStructAveragedPerceptron
    val AveragedPerceptron(tags, taggedWorkBook, featuresWeight) =
      model.getStructAveragedPerceptron
    val AveragedPerceptron(loadedTags, loadedTaggedWorkBook, loadedFeaturesWeight) =
      loadedModel.getStructAveragedPerceptron

    loadedTags shouldBe tags
    loadedTaggedWorkBook shouldBe taggedWorkBook
    assertMapArray(
      loadedFeaturesWeight.map { case (str, stringToDouble) => (str, stringToDouble.toArray) },
      featuresWeight.map { case (str, stringToDouble) => (str, stringToDouble.toArray) })

  }

  private def assertDependencyMaker(
      model: MockFeaturesModel,
      loadedModel: MockFeaturesModel): Unit = {
    val data = model.getStructDependencyMaker
    val dataLoaded = loadedModel.getStructDependencyMaker

    // TODO: This should compare all fields of the class...
    val taggerField = classOf[DependencyMaker].getDeclaredField("tagger")
    taggerField.setAccessible(true)
    val taggerValue = taggerField.get(data).asInstanceOf[Tagger]
    val taggerValueLoaded = taggerField.get(dataLoaded).asInstanceOf[Tagger]

    taggerValue.toString shouldEqual taggerValueLoaded.toString
  }

  private def assertStructDependencyPipe(
      model: MockFeaturesModel,
      loadedModel: MockFeaturesModel) = {
    val data = model.getStructDependencyPipe
    val dataLoaded = loadedModel.getStructDependencyPipe

    // TODO: This should compare all fields of the class...
    data.getTypes shouldBe dataLoaded.getTypes
  }

  // TODO: Something is really slow. need to profile.
  "MockFeaturesModel" should "serialize and deserialize correctly using proto" taggedAs SlowTest in {
    val model = setupMockFeaturesModel()
    // Save model features
    val tmpPath = "scala212_MockFeaturesModel"
    model.write.overwrite().save(tmpPath)
    // Create a new model and load features
    val loadedModel = MockFeaturesModel.load(tmpPath)

    // Compare all features
    // Arrays
    model.arrayTupleStringString.getOrDefault shouldBe loadedModel.arrayTupleStringString.getOrDefault
    model.arrayString.getOrDefault shouldBe loadedModel.arrayString.getOrDefault

    assertSameMap(model.mapIntTupleInt.getOrDefault, loadedModel.mapIntTupleInt.getOrDefault)
    assertSameMap(model.mapIntString.getOrDefault, loadedModel.mapIntString.getOrDefault)
    assertSameMap(model.mapStringBigInt.getOrDefault, loadedModel.mapStringBigInt.getOrDefault)
    assertSameMap(model.mapStringDouble.getOrDefault, loadedModel.mapStringDouble.getOrDefault)
    assertSameMap(model.mapStringInt.getOrDefault, loadedModel.mapStringInt.getOrDefault)
    assertSameMap(model.mapStringLong.getOrDefault, loadedModel.mapStringLong.getOrDefault)
    assertSameMap(
      model.mapStringMapStringFloat.getOrDefault,
      loadedModel.mapStringMapStringFloat.getOrDefault)
    assertSameMap(model.mapStringString.getOrDefault, loadedModel.mapStringString.getOrDefault)

    assertMapArray(
      model.mapStringArrayFloat.getOrDefault,
      loadedModel.mapStringArrayFloat.getOrDefault)
    assertMapArray(
      model.mapStringArrayFloat.getOrDefault,
      loadedModel.mapStringArrayFloat.getOrDefault)
    assertMapArray(
      model.mapStringArrayString.getOrDefault,
      loadedModel.mapStringArrayString.getOrDefault)

    // StructFeatures
    assertAveragedPerceptron(model, loadedModel)
    model.getStructClassifierDatasetEncoderParams.tags shouldBe loadedModel.getStructClassifierDatasetEncoderParams.tags
    model.getStructDatasetEncoderParams shouldBe loadedModel.getStructDatasetEncoderParams
    assertDependencyMaker(model, loadedModel)
    assertStructDependencyPipe(model, loadedModel)
    assertEntityRuler(model, loadedModel)
    assertLinearChainCrfModel(model, loadedModel)
    model.getStructMapIntFloat shouldBe loadedModel.getStructMapIntFloat
    model.getStructMapStringFloat shouldBe loadedModel.getStructMapStringFloat
    assertAhoCorasickAutomaton(model, loadedModel)
    model.getStructOptionMapStringInt.get shouldBe loadedModel.getStructOptionMapStringInt.get
    assertOptions(model, loadedModel)
//    assert(model.getStructParameters == loadedModel.getStructParameters)
//    assert(model.getStructRuleFactory == loadedModel.getStructRuleFactory)
//    assert(model.getStructSearchTrie == loadedModel.getStructSearchTrie)
//    assert(model.getStructString == loadedModel.getStructString)
//    assert(model.getStructTokenizerModel == loadedModel.getStructTokenizerModel)
  }

  private def assertOptions(model: MockFeaturesModel, loadedModel: MockFeaturesModel): Unit = {
    val data = model.getStructOptions
    val dataLoaded = loadedModel.getStructOptions

    assertSerializableFields(data, dataLoaded)
  }

  private def assertAhoCorasickAutomaton(
      model: MockFeaturesModel,
      loadedModel: MockFeaturesModel): Unit = {
    val data = model.getStructOptionAhoCorasickAutomaton.get
    val dataLoaded = loadedModel.getStructOptionAhoCorasickAutomaton.get

    data.alphabet shouldBe dataLoaded.alphabet

    val patternsField = classOf[AhoCorasickAutomaton].getDeclaredField("flattenEntityPatterns")
    patternsField.setAccessible(true)
    val patterns = patternsField.get(data).asInstanceOf[Array[FlattenEntityPattern]]
    val patternsLoaded = patternsField.get(dataLoaded).asInstanceOf[Array[FlattenEntityPattern]]

    patterns shouldBe patternsLoaded
  }

  private def assertLinearChainCrfModel(
      model: MockFeaturesModel,
      loadedModel: MockFeaturesModel) = {
    val data = model.getStructLinearChainCrfModel
    val loadedData = loadedModel.getStructLinearChainCrfModel
    data.weights shouldBe loadedData.weights
    data.weights shouldBe loadedData.weights

    val metadata: DatasetMetadata = data.metadata
    val loadedMetadata = loadedData.metadata

    metadata.labels shouldBe loadedMetadata.labels
    metadata.attrs shouldBe loadedMetadata.attrs
    metadata.attrFeatures shouldBe loadedMetadata.attrFeatures
    metadata.transitions shouldBe loadedMetadata.transitions
    metadata.featuresStat shouldBe loadedMetadata.featuresStat
  }

  private def assertEntityRuler(model: MockFeaturesModel, loadedModel: MockFeaturesModel) = {
    val data = model.getStructEntityRulerFeatures
    val loadedData = loadedModel.getStructEntityRulerFeatures

    data.patterns shouldBe loadedData.patterns
    data.regexPatterns shouldBe loadedData.regexPatterns
  }

}
