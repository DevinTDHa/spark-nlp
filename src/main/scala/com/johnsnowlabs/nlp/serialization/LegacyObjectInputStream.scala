package com.johnsnowlabs.nlp.serialization

import java.io.{
  IOException,
  InputStream,
  InvalidClassException,
  ObjectInputStream,
  ObjectStreamClass
}

class LegacyObjectInputStream(in: InputStream, val localClass: Class[_])
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
  @throws[IOException]("if an I/O error occurs")
  @throws[ClassNotFoundException]("if the class of a serialized object could not be found")
  override protected def readClassDescriptor: ObjectStreamClass = {
    var resultClassDescriptor = super.readClassDescriptor // initially streams descriptor

    // only if class implements serializable and original class is scala.Tuple2 and NOT the array
    if (resultClassDescriptor.getName == "scala.Tuple2") {
      val localClassDescriptor: ObjectStreamClass = ObjectStreamClass.lookup(
        localClass
      ) // TODO: the OSC here contains the exception "no valid constructor" because the array doesn't have one
      // TODO: Perhaps we need to define specific cases. This function might be called to get other types as well
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
