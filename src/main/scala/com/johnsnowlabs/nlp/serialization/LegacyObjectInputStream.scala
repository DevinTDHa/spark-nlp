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

    val localClassDescriptor: ObjectStreamClass = ObjectStreamClass.lookup(localClass)
    if (localClassDescriptor != null) { // only if class implements serializable
      val localSUID = localClassDescriptor.getSerialVersionUID
      val streamSUID = resultClassDescriptor.getSerialVersionUID
      if (streamSUID != localSUID) { // check for serialVersionUID mismatch.
        // Use local class descriptor for deserialization
        resultClassDescriptor = localClassDescriptor
      }
    } else throw new InvalidClassException("provided class is not serializable.")
    resultClassDescriptor
  }
}
