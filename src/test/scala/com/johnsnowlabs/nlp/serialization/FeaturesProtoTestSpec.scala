package com.johnsnowlabs.nlp.serialization

import com.johnsnowlabs.collections.SearchTrie
import com.johnsnowlabs.ml.crf._
import com.johnsnowlabs.ml.tensorflow.{ClassifierDatasetEncoderParams, DatasetEncoderParams}
import com.johnsnowlabs.nlp.annotators.TokenizerModel
import com.johnsnowlabs.nlp.annotators.er.{
  AhoCorasickAutomaton,
  EntityPattern,
  EntityRulerFeatures
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
import org.scalatest.flatspec.AnyFlatSpec

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

  def setupMockFeaturesModel(): MockFeaturesModel = {
    val mockModel = new MockFeaturesModel
    // Fill with dummy data
    val dummyArrayTupleStringString = Array(("a", "b"), ("c", "d"))
    val dummyArrayString = Array("foo", "bar")
    val dummyMapIntTupleInt = Map(1 -> (2, 3), 4 -> (5, 6))
    val dummyMapIntString = Map(1 -> "one", 2 -> "two")
    val dummyMapStringArrayFloat = Map("a" -> Array(1.0f, 2.0f))
    val dummyMapStringArrayString = Map("a" -> Array("b", "c"))
    val dummyMapStringBigInt = Map("big" -> BigInt(1234567890))
    val dummyMapStringDouble = Map("pi" -> 3.14)
    val dummyMapStringInt = Map("one" -> 1, "two" -> 2)
    val dummyMapStringLong = Map("long" -> 123456789L)
    val dummyMapStringMapStringFloat = Map("outer" -> Map("inner" -> 1.23f))
    val dummyMapStringString = Map("hello" -> "world")

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
    val dummyMapStringMapStringDouble = Map("outer" -> Map(("inner" -> 1.23d)))
    val dummyAveragedPerceptron =
      AveragedPerceptron(dummyArrayString, dummyMapStringString, dummyMapStringMapStringDouble)
    val dummyClassifierDatasetEncoderParams = ClassifierDatasetEncoderParams(dummyArrayString)
    val dummyDatasetEncoderParams =
      DatasetEncoderParams(dummyArrayString.toList, "test".toList, List.empty, 123, "test")
    val dummyDependencyMaker = new DependencyMaker(
      new Tagger(dummyArrayString.toVector, dummyMapStringInt))
    val dummyDependencyPipe = new DependencyPipe(new Options())
    val dummyArrayFloat = Array(1.0f, 2.0f)
    val dummyLinearChainCrfModel = new LinearChainCrfModel(
      dummyArrayFloat,
      new DatasetMetadata(
        dummyArrayString,
        Array(Attr(1, "one", isNumerical = true)),
        Array(AttrFeature(1, 2, 3)),
        Array(Transition(1, 2)),
        Array(AttrStat(1, 123f))))
    val dummyAhoCorasickAutomaton = new AhoCorasickAutomaton(
      dummyArrayString.head,
      Array(EntityPattern("test", dummyArrayString.toSeq, Some("test"), Some(true))),
      false)
    val dummyVectorTupleIntBoolIntInt = Vector((1, true, 3, 4))
    val dummyMapTupleIntInt = Map((1, 2) -> 3)

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
    // TODO: We should check for this data and if it is readable.
    mockModel.setStructParameters(new Parameters(dummyDependencyPipe, new Options()))
    mockModel.setStructRuleFactory(
      new RuleFactory(MatchStrategy.MATCH_ALL, TransformStrategy.NO_TRANSFORM))
    mockModel.setStructSearchTrie(
      new SearchTrie(dummyMapStringInt, dummyMapTupleIntInt, dummyVectorTupleIntBoolIntInt, true))
    mockModel.setStructString("dummy string")
    mockModel.setStructTokenizerModel(new TokenizerModel())

    mockModel
  }

  "MockFeaturesModel" should "serialize and deserialize correctly using proto" in {
    val model = setupMockFeaturesModel()
    // Save model features
    val tmpPath = "scala212_MockFeaturesModel"
    model.write.overwrite().save(tmpPath)
    // Create a new model and load features
    val loadedModel = MockFeaturesModel.load(tmpPath)

    // Compare all features
    // Arrays
    assert(
      model.arrayTupleStringString.getOrDefault.sameElements(
        loadedModel.arrayTupleStringString.getOrDefault))
    assert(model.arrayString.getOrDefault.sameElements(loadedModel.arrayString.getOrDefault))

    // Maps
    def assertSameMap[K, V](map1: Map[K, V], map2: Map[K, V]) =
      assert(map1.toSeq == map2.toSeq)

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

    // Maps with Arrays as values
    def assertMapArray[K](map1: Map[K, Array[_]], map2: Map[K, Array[_]]): Unit =
      map1.zip(map2).map { case ((k1, v1), (k2, v2)) =>
        assert(k1 == k2)
        assert(v1 sameElements v2)
      }
    assertMapArray(
      model.mapStringArrayFloat.getOrDefault,
      loadedModel.mapStringArrayFloat.getOrDefault)
    assertMapArray(
      model.mapStringArrayFloat.getOrDefault,
      loadedModel.mapStringArrayFloat.getOrDefault)
    assertMapArray(
      model.mapStringArrayString.getOrDefault,
      loadedModel.mapStringArrayString.getOrDefault)
  }

}
