package com.johnsnowlabs.nlp.annotators.tokenizer.moses

import com.johnsnowlabs.util.Benchmark

import scala.collection.mutable.ListBuffer

/**
  * Scala Port of the Moses Tokenizer from [[https://github.com/alvations/sacremoses scaremoses]].
  */
private[johnsnowlabs] class MosesTokenizer(lang: String) {
  require(lang == "en", "Only english is supported at the moment.")
  private final val DEDUPLICATE_SPACE = (raw"""\s+""", " ")
  private final val ASCII_JUNK = (raw"""[\000-\037]""", "")

  private final val IsAlpha = raw"""\p{L}"""
  private final val IsN = raw"""\p{N}"""

  private final val IsAlnum = IsAlpha + IsN // TODO: Lesser used languages like Tibetan, Khmer, Cham etc.
  private final val PAD_NOT_ISALNUM = (raw"""([^$IsAlnum\s\.'\`\,\-])""", " $1 ")

  private final val COMMA_SEPARATE_1 = (raw"""([^$IsN])[,]""", "$1 , ")
  private final val COMMA_SEPARATE_2 = (raw"""[,]([^$IsN])""", " , $1")
  private final val COMMA_SEPARATE_3 = (raw"""([$IsN])[,]$$""", "$1 , ")

  private final val EN_SPECIFIC_1 = (raw"""([^$IsAlpha])[']([^$IsAlpha])""", "$1 ' $2")
  private final val EN_SPECIFIC_2 = (raw"""([^$IsAlpha$IsN])[']([$IsAlpha])""", "$1 ' $2")
  private final val EN_SPECIFIC_3 = (raw"""([$IsAlpha])[']([^$IsAlpha])""", "$1 ' $2")
  private final val EN_SPECIFIC_4 = (raw"""([$IsAlpha])[']([$IsAlpha])""", "$1 '$2")
  private final val EN_SPECIFIC_5 = (raw"""([$IsN])[']([s])""", "$1 '$2")
  private final val ENGLISH_SPECIFIC_APOSTROPHE = Array(
    EN_SPECIFIC_1,
    EN_SPECIFIC_2,
    EN_SPECIFIC_3,
    EN_SPECIFIC_4,
    EN_SPECIFIC_5
  )
  private final val NON_SPECIFIC_APOSTROPHE = (raw"""\'""", " ' ")
  private final val TRAILING_DOT_APOSTROPHE = (raw"""\.' ?$$""", " . ' ")
  // TODO: Dynamic from file
  private final val NONBREAKING_PREFIXES = Array("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Adj", "Adm", "Adv", "Asst", "Bart", "Bldg", "Brig", "Bros", "Capt", "Cmdr", "Col", "Comdr", "Con", "Corp", "Cpl", "DR", "Dr", "Drs", "Ens", "Gen", "Gov", "Hon", "Hr", "Hosp", "Insp", "Lt", "MM", "MR", "MRS", "MS", "Maj", "Messrs", "Mlle", "Mme", "Mr", "Mrs", "Ms", "Msgr", "Op", "Ord", "Pfc", "Ph", "Prof", "Pvt", "Rep", "Reps", "Res", "Rev", "Rt", "Sen", "Sens", "Sfc", "Sgt", "Sr", "St", "Supt", "Surg", "v", "vs", "i.e", "rev", "e.g", "No #NUMERIC_ONLY#", "Nos", "Art #NUMERIC_ONLY#", "Nr", "pp #NUMERIC_ONLY#", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
  private final val NUMERIC_ONLY_PREFIXES = Array("No", "Art", "pp")

  private def replaceMultidots(text: String): String = {
    var processed: String = text
    processed = processed.replaceAll(raw"""\.([\.]+)""", " DOTMULTI$1")
    while (processed.indexOf("DOTMULTI.") >= 0) { // re.search(raw""""DOTMULTI\.", text)
      processed = processed.replaceAll(raw"""DOTMULTI\.([^\.])""", "DOTDOTMULTI $1")
      processed = processed.replaceAll(raw"""DOTMULTI\.""", "DOTDOTMULTI")
    }
    processed
  }

  private def isAnyAlpha(s: String): Boolean = s"[$IsAlnum]".r.findFirstIn(s) match {
    case Some(_) => true
    case None => false
  }


  private def isLower(s: String): Boolean = s.matches(raw"""\p{Ll}*""") // TODO Some languages missing

  def handlesNonBreakingPrefixes(text: String): String = {
    // Splits the text into tokens to check for nonbreaking prefixes.
    val tokens = text.split(" ")
    val numTokens = tokens.length
    for ((token, i) <- tokens.zipWithIndex) {
      // Checks if token ends with a full stop
      val tokenEndsWithPeriod = raw"""^(\S+)\.$$""".r.findFirstMatchIn(token)
      tokenEndsWithPeriod match {
        case None => tokenEndsWithPeriod
        case Some(prefixMatch) =>
          val prefix = prefixMatch.group(1)

          // Checks for 3 conditions if
          // i.   the prefix contains a fullstop and
          //      any char in the prefix is within the IsAlpha charset
          // ii.  the prefix is in the list of nonbreaking prefixes and
          //      does not contain #NUMERIC_ONLY#
          // iii. the token is not the last token and that the
          //      next token contains all lowercase.

          // No change to the token.
          // Checks if the prefix is in NUMERIC_ONLY_PREFIXES
          // and ensures that the next word is a digit.
          def containsFullStopAndIsAlpha = ((prefix contains ".") && isAnyAlpha(prefix)) ||
            (NONBREAKING_PREFIXES.contains(prefix) && !NUMERIC_ONLY_PREFIXES.contains(prefix)) ||
            (
              (i != numTokens - 1)
                && tokens(i + 1).nonEmpty
                && isLower(tokens(i + 1)(0).toString)
              )

          // No change to the token.
          def isNonBreakingAndNumericOnly = (
            NONBREAKING_PREFIXES.contains(prefix)
              && ((i + 1) < numTokens)
              && raw"""^[0-9]+""".r.findFirstIn(tokens(i + 1)).isDefined
            )
          // Otherwise, adds a space after the tokens before a dot.
          if (!containsFullStopAndIsAlpha && !isNonBreakingAndNumericOnly) tokens(i) = prefix + " ."
      }
    }
    tokens.mkString(" ") // Stitch the tokens back.
  }

  private def restoreMultidots(text: String) = {
    var processed = text
    while (processed.indexOf("DOTDOTMULTI") > 0) { // re.search(r"DOTDOTMULTI", text):
      processed = processed.replace("DOTDOTMULTI", "DOTMULTI.")
    }
    processed.replace("DOTMULTI", ".")
  }

  def tokenize(text: String): Array[String] = {
    var processed = text

    def applySubstitution(text: String, patternReplacements: (String, String)*): String = {
      var processed = text
      for ((pattern, sub) <- patternReplacements) {
        processed = processed.replaceAll(pattern, sub)
      }
      processed
    }

//    val bdeduplicateSpace = Benchmark.time2("DEDUPLICATE_SPACE, ASCII_JUNK") {
      processed = applySubstitution(processed, DEDUPLICATE_SPACE, ASCII_JUNK)
      processed = processed.trim()
//    }

    //    if (protectedPatterns) ???

//    val bpadNotIsalnum = Benchmark.time2("PAD_NOT_ISALNUM") {
      processed = applySubstitution(processed, PAD_NOT_ISALNUM)
//    }
//    MosesTokenizerBenchmark.padNotIsalnum.append(bpadNotIsalnum)

    //    if (aggressiveDashSplits) ???

//    val breplaceMultidots = Benchmark.time2("replaceMultidots") {
      processed = replaceMultidots(processed)
//    }
//    MosesTokenizerBenchmark.replaceMultidots.append(breplaceMultidots)

//    val bcommaSeparate = Benchmark.time2("COMMA_SEPARATE") {
      processed = applySubstitution(processed, COMMA_SEPARATE_1, COMMA_SEPARATE_2, COMMA_SEPARATE_3)
//    }
//    MosesTokenizerBenchmark.commaSeparate.append(bcommaSeparate)

//    val benglishSpecificApostrophe = Benchmark.time2("ENGLISH_SPECIFIC_APOSTROPHE") {
      if (lang == "en") processed = applySubstitution(processed, ENGLISH_SPECIFIC_APOSTROPHE: _*)
      else if (lang == "it" || lang == "fr") ??? // TODO
      else processed = applySubstitution(processed, NON_SPECIFIC_APOSTROPHE)
//    }

//    val bhandlesNonBreakingPrefixes = Benchmark.time2("handlesNonBreakingPrefixes") {
      processed = handlesNonBreakingPrefixes(processed)
//    }
//    MosesTokenizerBenchmark.handlesNonBreakingPrefixes.append(bhandlesNonBreakingPrefixes)

//    val bdeduplicateSpace2 = Benchmark.time2("DEDUPLICATE_SPACE") {
      processed = applySubstitution(processed, DEDUPLICATE_SPACE).trim()
//    }
//    MosesTokenizerBenchmark.deduplicateSpace2.append(bdeduplicateSpace2)

//    val btrailingDotApostrophe = Benchmark.time2("TRAILING_DOT_APOSTROPHE") {
      processed = applySubstitution(processed, TRAILING_DOT_APOSTROPHE)
//    }
//    MosesTokenizerBenchmark.trailingDotApostrophe.append(btrailingDotApostrophe)
    // Restore the protected tokens.
    // if (protectedPatterns) ???
//    val brestoreMultidots = Benchmark.time2("restoreMultidots") {
      processed = restoreMultidots(processed)
//    }
//    MosesTokenizerBenchmark.restoreMultidots.append(brestoreMultidots)
    processed.split(" ")
  }
}

object MosesTokenizer {
  val deduplicateSpace: ListBuffer[Double] = ListBuffer()
  val padNotIsalnum: ListBuffer[Double] = ListBuffer()
  val replaceMultidots: ListBuffer[Double] = ListBuffer()
  val commaSeparate: ListBuffer[Double] = ListBuffer()
  val englishSpecificApostrophe: ListBuffer[Double] = ListBuffer()
  val handlesNonBreakingPrefixes: ListBuffer[Double] = ListBuffer()
  val deduplicateSpace2: ListBuffer[Double] = ListBuffer()
  val trailingDotApostrophe: ListBuffer[Double] = ListBuffer()
  val restoreMultidots: ListBuffer[Double] = ListBuffer()
}