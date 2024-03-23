/*
 * Copyright 2017-2024 John Snow Labs
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

package com.johnsnowlabs.ml.gguf
import com.johnsnowlabs.util.{FileHelper, ZipArchiveUtil}
import de.kherud.llama.LlamaModel
import org.apache.commons.io.FileUtils
import org.slf4j.{Logger, LoggerFactory}

import java.io._
import java.nio.file.{Files, Paths}
import java.util.UUID

private[johnsnowlabs] class GGUFWrapper(var ggufModel: Array[Byte])
    extends Serializable
    with AutoCloseable {

  /** For Deserialization */
  def this() = {
    this(null)
  }

  // Important for serialization on none-kyro serializers
  @transient private var llamaModel: LlamaModel = _

  def getSession: LlamaModel =
    this.synchronized {
      if (llamaModel == null) {
        llamaModel = GGUFWrapper.withSafeGGUFModelLoader(ggufModel)
      }
      llamaModel
    }

  def saveToFile(file: String, zip: Boolean = true): Unit = {
    // 1. Create tmp director
    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_gguf")
      .toAbsolutePath
      .toString

    // 2. Save gguf model
    val fileName = Paths.get(file).getFileName.toString
    val ggufFile = Paths
      .get(tmpFolder, fileName)
      .toString

    FileUtils.writeByteArrayToFile(new File(ggufFile), ggufModel)
    // 4. Zip folder
    if (zip) ZipArchiveUtil.zip(tmpFolder, file)

    // 5. Remove tmp directory
    FileHelper.delete(tmpFolder)
  }

  // TODO: When to do this actually
  override def close(): Unit = if (llamaModel != null) llamaModel.close()
}

/** Companion object */
object GGUFWrapper {
  private[GGUFWrapper] val logger: Logger = LoggerFactory.getLogger("GGUFWrapper")

  // TODO: make sure this.synchronized is needed or it's not a bottleneck
  private def withSafeGGUFModelLoader(ggufModel: Array[Byte]): LlamaModel =
    this.synchronized {
      // TODO: tmp solution, maybe we can load the model directly from addFile or from bytes
      val tmpFile = Files
        .createTempFile(UUID.randomUUID().toString.takeRight(12), ".gguf")

      // write the array of bytes to file
      Files.write(tmpFile, ggufModel)

      // TODO: Add model parameters
      new LlamaModel(tmpFile.toAbsolutePath.toString)
    }

  def read(modelPath: String): GGUFWrapper = {
    // TODO Does not work for large files...
    val ggufModel = Files.readAllBytes(Paths.get(modelPath))
    new GGUFWrapper(ggufModel)
  }
}
