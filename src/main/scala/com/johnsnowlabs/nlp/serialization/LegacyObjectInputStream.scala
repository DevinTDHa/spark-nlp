package com.johnsnowlabs.nlp.serialization

import java.io.{
  ByteArrayInputStream,
  IOException,
  InputStream,
  InvalidClassException,
  ObjectInputStream,
  ObjectStreamClass
}
import scala.reflect.ClassTag

/** Custom ObjectInputStream that ignores the serialVersionUID check for a provided class during
  * deserialization.
  *
  * @param in
  *   ByteArrayInputStream of the deserialization
  * @param replacementClass
  *   The class that should be the replacement
  * @param serializedClassName
  *   The name of the serialized class in the input stream
  */
class LegacyObjectInputStream(
    in: ByteArrayInputStream,
    val replacementClass: Class[_],
    val serializedClassName: String)
    extends ObjectInputStream(in) {

  /** Taken from
    * https://stackoverflow.com/questions/795470/how-to-deserialize-an-object-persisted-in-a-db-now-when-the-object-has-different
    * @throws IOException
    *   if an I/O error occurs
    * @throws ClassNotFoundException
    *   if the class of a serialized object could not be found
    *
    * @return
    */
  @throws[IOException]("I/O error occurred")
  @throws[ClassNotFoundException]("class of a serialized object could not be found")
  override protected def readClassDescriptor: ObjectStreamClass = {
    var resultClassDescriptor = super.readClassDescriptor // initially streams descriptor

    // only if class implements serializable and original class is scala.Tuple2 and NOT the array
    if (resultClassDescriptor.getName == serializedClassName) {
      val localClassDescriptor: ObjectStreamClass = ObjectStreamClass.lookup(replacementClass)
      if (localClassDescriptor != null) {
        val localSUID = localClassDescriptor.getSerialVersionUID
        val streamSUID = resultClassDescriptor.getSerialVersionUID
        if (streamSUID != localSUID) { // check for serialVersionUID mismatch.
          // Use local class descriptor for deserialization
          resultClassDescriptor = localClassDescriptor
        }
      } else throw new InvalidClassException("provided class is not serializable.")
    }

    resultClassDescriptor
  }

}
object LegacyObjectInputStream {

  /** Deserialize this class using a custom object input stream, that ignores the serialVersionUID
    * and loads a replacement class instead. This assumes that the objects were serialized as an
    * array.
    *
    * @param bytes
    *   The bytes to deserialized (read by BytesWritable)
    * @param serializedClassName
    *   The name of the serialized class to replace
    * @tparam T
    *   The type of the array contents, which will be the replacement for serializedClassName
    * @return
    */
  def deserializeArray[T: ClassTag](bytes: Array[Byte], serializedClassName: String): Array[T] = {
    val bis = new ByteArrayInputStream(bytes)
    // Use ClassTag to store runtime information of class and avoid type erasure.
    // Retrieves the implicitly context-bound parameter of the ClassTag
    val ois =
      new LegacyObjectInputStream(bis, implicitly[ClassTag[T]].runtimeClass, serializedClassName)

    ois.readObject.asInstanceOf[Array[T]]
  }
}
