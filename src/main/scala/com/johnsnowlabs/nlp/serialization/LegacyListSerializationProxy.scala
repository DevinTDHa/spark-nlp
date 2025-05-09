package com.johnsnowlabs.nlp.serialization

import java.io.{ObjectInputStream, ObjectOutputStream}

@SerialVersionUID(212L)
class LegacyListSerializationProxy(@transient private var orig: List[Any]) extends Serializable {

  private def writeObject(out: ObjectOutputStream) {
    out.defaultWriteObject()
    var xs: List[Any] = orig
    while (!xs.isEmpty) {
      out.writeObject(xs.head)
      xs = xs.tail
    }
    out.writeObject(ListSerializeEnd)
  }

  // Java serialization calls this before readResolve during deserialization.
  // Read the whole list and store it in `orig`.
  private def readObject(in: ObjectInputStream) {
    in.defaultReadObject()
    val builder = List.newBuilder[Any]
    while (true) in.readObject match {
      case ListSerializeEnd =>
        orig = builder.result()
        return
      case a =>
        builder += a // original code casts to type, we use Any
    }
  }

  // Provide the result stored in `orig` for Java serialization
  private def readResolve(): AnyRef = orig
}

/** Only used for list serialization */
@SerialVersionUID(0L - 8476791151975527571L)
case object ListSerializeEnd
