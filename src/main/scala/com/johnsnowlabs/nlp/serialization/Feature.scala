/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.serialization

import com.github.liblevenshtein.serialization.PlainTextSerializer
import com.johnsnowlabs.nlp.HasFeatures
import com.johnsnowlabs.nlp.annotators.spell.context.parser.VocabParser
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Encoder, Encoders, SparkSession}

import java.io.{ByteArrayInputStream, InputStream, ObjectInputStream, ObjectStreamClass}
import scala.reflect.ClassTag

abstract class Feature[Serializable1, Serializable2, TComplete: ClassTag](
    model: HasFeatures,
    val name: String)
    extends Serializable {
  model.features.append(this)

  private val spark: SparkSession = ResourceHelper.spark

  val serializationMode: String =
    ConfigLoader.getConfigStringValue(ConfigHelper.serializationMode)
  val useBroadcast: Boolean = ConfigLoader.getConfigBooleanValue(ConfigHelper.useBroadcast)
  final protected var broadcastValue: Option[Broadcast[TComplete]] = None

  final protected var rawValue: Option[TComplete] = None
  final protected var fallbackRawValue: Option[TComplete] = None

  final protected var fallbackLazyValue: Option[() => TComplete] = None
  final protected var isProtected: Boolean = false

  // TODO: This should be kryo?
  final def serialize(
      spark: SparkSession,
      path: String,
      field: String,
      value: TComplete): Unit = {
    serializationMode match {
      case "dataset" => serializeDataset(spark, path, field, value)
      case "object" => serializeObject(spark, path, field, value)
      case _ =>
        throw new IllegalArgumentException(
          "Illegal performance.serialization setting. Can be 'dataset' or 'object'")
    }
  }

  final def serializeInfer(spark: SparkSession, path: String, field: String, value: Any): Unit =
    serialize(spark, path, field, value.asInstanceOf[TComplete])

  final def deserialize(spark: SparkSession, path: String, field: String): Option[_] = {
    if (broadcastValue.isDefined || rawValue.isDefined)
      throw new Exception(
        s"Trying de deserialize an already set value for ${this.name}. This should not happen.")
    serializationMode match {
      case "dataset" => deserializeDataset(spark, path, field)
      case "object" => deserializeObject(spark, path, field)
      case _ =>
        throw new IllegalArgumentException(
          "Illegal performance.serialization setting. Can be 'dataset' or 'object'")
    }
  }

  protected def serializeDataset(
      spark: SparkSession,
      path: String,
      field: String,
      value: TComplete): Unit

  protected def deserializeDataset(spark: SparkSession, path: String, field: String): Option[_]

  protected def serializeObject(
      spark: SparkSession,
      path: String,
      field: String,
      value: TComplete): Unit

  protected def deserializeObject(spark: SparkSession, path: String, field: String): Option[_]

  final protected def getFieldPath(path: String, field: String): Path =
    Path.mergePaths(new Path(path), new Path("/fields/" + field))

  private def callAndSetFallback: Option[TComplete] = {
    fallbackRawValue = fallbackLazyValue.map(_())
    fallbackRawValue
  }

  final def get: Option[TComplete] = {
    broadcastValue.map(_.value).orElse(rawValue)
  }

  final def orDefault: Option[TComplete] = {
    broadcastValue
      .map(_.value)
      .orElse(rawValue)
      .orElse(fallbackRawValue)
      .orElse(callAndSetFallback)
  }

  final def getOrDefault: TComplete = {
    orDefault
      .getOrElse(throw new Exception(s"feature $name is not set"))
  }

  final def setValue(value: Option[Any]): HasFeatures = {
    if (isProtected && isSet) {
      val warnString =
        s"Warning: The parameter ${this.name} is protected and can only be set once." +
          " For a pretrained model, this was done during the initialization process." +
          " If you are trying to train your own model, please check the documentation."
      println(warnString)
    } else {
      if (useBroadcast) {
        if (isSet) broadcastValue.get.destroy()
        broadcastValue =
          value.map(v => spark.sparkContext.broadcast[TComplete](v.asInstanceOf[TComplete]))
      } else {
        rawValue = value.map(_.asInstanceOf[TComplete])
      }
    }
    model
  }

  def setFallback(v: Option[() => TComplete]): HasFeatures = {
    fallbackLazyValue = v
    model
  }

  final def isSet: Boolean = {
    broadcastValue.isDefined || rawValue.isDefined
  }

  /** Sets this feature to be protected and only settable once.
    *
    * @return
    *   This Feature
    */
  final def setProtected(): this.type = {
    isProtected = true
    this
  }

}

class StructFeature[TValue: ClassTag](model: HasFeatures, override val name: String)
    extends Feature[TValue, TValue, TValue](model, name) {

  implicit val encoder: Encoder[TValue] = Encoders.kryo[TValue]

  override def serializeObject(
      spark: SparkSession,
      path: String,
      field: String,
      value: TValue): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.sparkContext.parallelize(Seq(value)).saveAsObjectFile(dataPath.toString)
  }

  override def deserializeObject(
      spark: SparkSession,
      path: String,
      field: String): Option[TValue] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.sparkContext.objectFile[TValue](dataPath.toString).first)
    } else {
      None
    }
  }

  override def serializeDataset(
      spark: SparkSession,
      path: String,
      field: String,
      value: TValue): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.createDataset(Seq(value)).write.mode("overwrite").parquet(dataPath.toString)
  }

  override def deserializeDataset(
      spark: SparkSession,
      path: String,
      field: String): Option[TValue] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.read.parquet(dataPath.toString).as[TValue].first)
    } else {
      None
    }
  }

}

class MapFeature[TKey: ClassTag, TValue: ClassTag](model: HasFeatures, override val name: String)
    extends Feature[TKey, TValue, Map[TKey, TValue]](model, name) {

  implicit val encoder: Encoder[(TKey, TValue)] = Encoders.kryo[(TKey, TValue)]

  override def serializeObject(
      spark: SparkSession,
      path: String,
      field: String,
      value: Map[TKey, TValue]): Unit = {
    val dataPath = getFieldPath(path, field)
    // TODO: Change this to kryo or some better serializer
    spark.sparkContext.parallelize(value.toSeq).saveAsObjectFile(dataPath.toString)
  }

  /** Loads a scala tuple of TKey and TValue from a SequenceFile containing serialized objects. It
    * tries to load the tuple across Scala version.
    *
    * Copied:
    *
    * Load an RDD saved as a SequenceFile containing serialized objects, with NullWritable keys
    * and BytesWritable values that contain a serialized partition. This is still an experimental
    * storage format and may not be supported exactly as is in future Spark releases. It will also
    * be pretty slow if you use the default serializer (Java serialization), though the nice thing
    * about it is that there's very little effort required to save arbitrary objects.
    *
    * @param path
    *   directory to the input data files, the path can be comma separated paths as a list of
    *   inputs
    * @return
    *   RDD representing deserialized data from the file(s)
    */
  private def deserializeOldTupleObjects[K, V](spark: SparkSession, path: String): RDD[(K, V)] = {

    /** Classes that require special handling during the serialization and deserialization process
      * must implement special methods with these exact signatures:
      *   - private void writeObject(java.io.ObjectOutputStream out) throws IOException
      *   - private void readObject(java.io.ObjectInputStream in) throws IOException,
      *     ClassNotFoundException;
      *   - private void readObjectNoData() throws ObjectStreamException;
      */
//    class OldScalaTuple(var _1: K, var _2: V) extends Serializable {
//    class OldScalaTuple() extends Serializable {
//      println("Construtor")
//    }

    /** Deserialize this class using a custom object input stream */
    def deserialize[T](bytes: Array[Byte]): T = {
      val bis = new ByteArrayInputStream(bytes)
      val ois = new LegacyObjectInputStream(bis, classOf[(K, V)])
//      val ois = new LegacyObjectInputStream(bis, classOf[Array[(String, Int)]])

      ois.readObject.asInstanceOf[T]
    }

    spark.sparkContext
      .sequenceFile(
        path,
        classOf[NullWritable],
        classOf[BytesWritable],
        spark.sparkContext.defaultMinPartitions)
      .flatMap((x: (NullWritable, BytesWritable)) => deserialize[Array[(K, V)]](x._2.getBytes))
  }

  override def deserializeObject(
      spark: SparkSession,
      path: String,
      field: String): Option[Map[TKey, TValue]] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    // TODO: We can provide a custom deserializer here, replace (TKey, TValue) with it
    if (fs.exists(dataPath)) {
//      Some(spark.sparkContext.objectFile[(TKey, TValue)](dataPath.toString).collect.toMap)
      Some(deserializeOldTupleObjects[TKey, TValue](spark, dataPath.toString).collect.toMap)
    } else {
      None
    }
  }

  override def serializeDataset(
      spark: SparkSession,
      path: String,
      field: String,
      value: Map[TKey, TValue]): Unit = {
    import spark.implicits._
    val dataPath = getFieldPath(path, field)
    value.toSeq.toDS.write.mode("overwrite").parquet(dataPath.toString)
  }

  override def deserializeDataset(
      spark: SparkSession,
      path: String,
      field: String): Option[Map[TKey, TValue]] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.read.parquet(dataPath.toString).as[(TKey, TValue)].collect.toMap)
    } else {
      None
    }
  }

}

class ArrayFeature[TValue: ClassTag](model: HasFeatures, override val name: String)
    extends Feature[TValue, TValue, Array[TValue]](model, name) {

  implicit val encoder: Encoder[TValue] = Encoders.kryo[TValue]

  override def serializeObject(
      spark: SparkSession,
      path: String,
      field: String,
      value: Array[TValue]): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.sparkContext.parallelize(value).saveAsObjectFile(dataPath.toString)
  }

  override def deserializeObject(
      spark: SparkSession,
      path: String,
      field: String): Option[Array[TValue]] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.sparkContext.objectFile[TValue](dataPath.toString).collect())
    } else {
      None
    }
  }

  override def serializeDataset(
      spark: SparkSession,
      path: String,
      field: String,
      value: Array[TValue]): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.createDataset(value).write.mode("overwrite").parquet(dataPath.toString)
  }

  override def deserializeDataset(
      spark: SparkSession,
      path: String,
      field: String): Option[Array[TValue]] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.read.parquet(dataPath.toString).as[TValue].collect)
    } else {
      None
    }
  }

}

class SetFeature[TValue: ClassTag](model: HasFeatures, override val name: String)
    extends Feature[TValue, TValue, Set[TValue]](model, name) {

  implicit val encoder: Encoder[TValue] = Encoders.kryo[TValue]

  override def serializeObject(
      spark: SparkSession,
      path: String,
      field: String,
      value: Set[TValue]): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.sparkContext.parallelize(value.toSeq).saveAsObjectFile(dataPath.toString)
  }

  override def deserializeObject(
      spark: SparkSession,
      path: String,
      field: String): Option[Set[TValue]] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.sparkContext.objectFile[TValue](dataPath.toString).collect().toSet)
    } else {
      None
    }
  }

  override def serializeDataset(
      spark: SparkSession,
      path: String,
      field: String,
      value: Set[TValue]): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.createDataset(value.toSeq).write.mode("overwrite").parquet(dataPath.toString)
  }

  override def deserializeDataset(
      spark: SparkSession,
      path: String,
      field: String): Option[Set[TValue]] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      Some(spark.read.parquet(dataPath.toString).as[TValue].collect.toSet)
    } else {
      None
    }
  }

}

class TransducerFeature(model: HasFeatures, override val name: String)
    extends Feature[VocabParser, VocabParser, VocabParser](model, name) {

  override def serializeObject(
      spark: SparkSession,
      path: String,
      field: String,
      trans: VocabParser): Unit = {
    val dataPath = getFieldPath(path, field)
    spark.sparkContext.parallelize(Seq(trans), 1).saveAsObjectFile(dataPath.toString)

  }

  override def deserializeObject(
      spark: SparkSession,
      path: String,
      field: String): Option[VocabParser] = {
    val serializer = new PlainTextSerializer
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      val sc = spark.sparkContext.objectFile[VocabParser](dataPath.toString).collect().head
      Some(sc)
    } else {
      None
    }
  }

  override def serializeDataset(
      spark: SparkSession,
      path: String,
      field: String,
      trans: VocabParser): Unit = {
    implicit val encoder: Encoder[VocabParser] = Encoders.kryo[VocabParser]
    val serializer = new PlainTextSerializer
    val dataPath = getFieldPath(path, field)
    spark.createDataset(Seq(trans)).write.mode("overwrite").parquet(dataPath.toString)
  }

  override def deserializeDataset(
      spark: SparkSession,
      path: String,
      field: String): Option[VocabParser] = {
    implicit val encoder: Encoder[VocabParser] = Encoders.kryo[VocabParser]
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dataPath = getFieldPath(path, field)
    if (fs.exists(dataPath)) {
      val sc = spark.read.parquet(dataPath.toString).as[VocabParser].collect.head
      Some(sc)
    } else {
      None
    }
  }

}
