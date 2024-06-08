package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntArrayParam, IntParam, Param, StringArrayParam}

/** Parameters to configure beam search text generation. */
trait HasLlamaCppProperties {
  this: ParamsAndFeaturesWritable =>


  // ---------------- MODEL PARAMETERS ----------------
//  val prompt = new Param[String]("prompt", "")
  val inputPrefix = new Param[String](this, "inputPrefix", "Set the prompt to start generation with (default: empty)")
  val inputSuffix = new Param[String](this, "inputSuffix", "Set a suffix for infilling (default: empty)")
  val cachePrompt = new BooleanParam(this, "cachePrompt", "Whether to remember the prompt to avoid reprocessing it")
  val nPredict = new IntParam(this, "nPredict", "Set the number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)")
  val topK = new IntParam(this, "topK", "Set top-k sampling (default: 40, 0 = disabled)")
  val topP = new FloatParam(this, "topP", "Set top-p sampling (default: 0.9, 1.0 = disabled)")
  val minP = new FloatParam(this, "minP", "Set min-p sampling (default: 0.1, 0.0 = disabled)")
  val tfsZ = new FloatParam(this, "tfsZ", "Set tail free sampling, parameter z (default: 1.0, 1.0 = disabled)")
  val typicalP = new FloatParam(this, "typicalP", "Set locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)")
  val temperature = new FloatParam(this, "temperature", "Set the temperature (default: 0.8)")
  val dynamicTemperatureRange = new FloatParam(this, "dynatempRange", "Set the dynamic temperature range (default: 0.0, 0.0 = disabled)")
  val dynamicTemperatureExponent = new FloatParam(this, "dynatempExponent", "Set the dynamic temperature exponent (default: 1.0)")
  val repeatLastN = new IntParam(this, "repeatLastN", "Set the last n tokens to consider for penalties (default: 64, 0 = disabled, -1 = ctx_size)")
  val repeatPenalty = new FloatParam(this, "repeatPenalty", "Set the penalty of repeated sequences of tokens (default: 1.0, 1.0 = disabled)")
  val frequencyPenalty = new FloatParam(this, "frequencyPenalty", "Set the repetition alpha frequency penalty (default: 0.0, 0.0 = disabled)")
  val presencePenalty = new FloatParam(this, "presencePenalty", "Set the repetition alpha presence penalty (default: 0.0, 0.0 = disabled)")
  // Set MiroStat sampling strategies.
  //  val miroStat = MiroStat mirostat // enum DISABLED, V1, V2
  val miroStatTau = new FloatParam(this, "mirostatTau", "Set the MiroStat target entropy, parameter tau (default: 5.0)")
  val miroStatEta = new FloatParam(this, "mirostatEta", "Set the MiroStat learning rate, parameter eta (default: 0.1)")
  val penalizeNl = new BooleanParam(this, "penalizeNl", "Whether to penalize newline tokens")
  val nKeep = new IntParam(this, "nKeep", "Set the number of tokens to keep from the initial prompt (default: 0, -1 = all)")
  val seed = new IntParam(this, "seed", "Set the RNG seed (default: -1, use random seed for < 0)")
  val nProbs = new IntParam(this, "nProbs", "Set the amount top tokens probabilities to output if greater than 0.")
  val minKeep = new IntParam(this, "minKeep", "Set the amount of tokens the samplers should return at least (0 = disabled)")
  val grammar = new Param[String](this, "grammar", "Set BNF-like grammar to constrain generations (see samples in grammars/ dir)")
  val penaltyPrompt = new Param[String](this, "penaltyPrompt", "Override which part of the prompt is penalized for repetition.") // TODO: or provide token ids?
  val ignoreEos = new BooleanParam(this, "ignoreEos", "Set whether to ignore end of stream token and continue generating (implies --logit-bias 2-inf)")
  // Modify the likelihood of tokens appearing in the completion by their id.
  val tokenIdBias: Map[Integer, Float]

  // Modify the likelihood of tokens appearing in the completion by their string.
  val tokenBias: Map[String, Float]
  // 	 * Set tokens to disable, this corresponds to {@link #setTokenIdBias(Map)} with a value of
  //	 * {@link Float#NEGATIVE_INFINITY}.
  val disableTokenIds = new IntArrayParam(this, "disableTokenIds", "Set the token ids to disable in the completion")

  val stopStrings= new StringArrayParam(this, "stopStrings", "Set strings upon seeing which token generation is stopped")

  // Set which samplers to use for token generation in the given order
  // val samplers = Sampler... samplers // either TOP_K, TFS_Z, TYPICAL_P, TOP_P, MIN_P, TEMPERATURE
//  val stream = new BooleanParam(this, "stream", "Whether to stream the output or not")
  val useChatTemplate = new BooleanParam(this, "useChatTemplate", "Set whether or not generate should apply a chat template (default: false)")

  // ---------------- INFERENCE PARAMETERS ----------------
}
