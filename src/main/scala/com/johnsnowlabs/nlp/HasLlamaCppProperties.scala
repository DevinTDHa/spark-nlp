package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.{
  BooleanParam,
  FloatParam,
  IntArrayParam,
  IntParam,
  Param,
  StringArrayParam
}

/** Parameters to configure beam search text generation. */
trait HasLlamaCppProperties {
  this: ParamsAndFeaturesWritable =>

  // ---------------- INFERENCE PARAMETERS ----------------
//  val prompt = new Param[String]("prompt", "")
  /** @group param */
  val inputPrefix = new Param[String](
    this,
    "inputPrefix",
    "Set the prompt to start generation with (default: empty)")

  /** @group param */
  val inputSuffix =
    new Param[String](this, "inputSuffix", "Set a suffix for infilling (default: empty)")

  /** @group param */
  val cachePrompt = new BooleanParam(
    this,
    "cachePrompt",
    "Whether to remember the prompt to avoid reprocessing it")

  /** @group param */
  val nPredict = new IntParam(
    this,
    "nPredict",
    "Set the number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)")

  /** @group param */
  val topK = new IntParam(this, "topK", "Set top-k sampling (default: 40, 0 = disabled)")

  /** @group param */
  val topP = new FloatParam(this, "topP", "Set top-p sampling (default: 0.9, 1.0 = disabled)")

  /** @group param */
  val minP = new FloatParam(this, "minP", "Set min-p sampling (default: 0.1, 0.0 = disabled)")

  /** @group param */
  val tfsZ = new FloatParam(
    this,
    "tfsZ",
    "Set tail free sampling, parameter z (default: 1.0, 1.0 = disabled)")

  /** @group param */
  val typicalP = new FloatParam(
    this,
    "typicalP",
    "Set locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)")

  /** @group param */
  val temperature = new FloatParam(this, "temperature", "Set the temperature (default: 0.8)")

  /** @group param */
  val dynamicTemperatureRange = new FloatParam(
    this,
    "dynatempRange",
    "Set the dynamic temperature range (default: 0.0, 0.0 = disabled)")

  /** @group param */
  val dynamicTemperatureExponent = new FloatParam(
    this,
    "dynatempExponent",
    "Set the dynamic temperature exponent (default: 1.0)")

  /** @group param */
  val repeatLastN = new IntParam(
    this,
    "repeatLastN",
    "Set the last n tokens to consider for penalties (default: 64, 0 = disabled, -1 = ctx_size)")

  /** @group param */
  val repeatPenalty = new FloatParam(
    this,
    "repeatPenalty",
    "Set the penalty of repeated sequences of tokens (default: 1.0, 1.0 = disabled)")

  /** @group param */
  val frequencyPenalty = new FloatParam(
    this,
    "frequencyPenalty",
    "Set the repetition alpha frequency penalty (default: 0.0, 0.0 = disabled)")

  /** @group param */
  val presencePenalty = new FloatParam(
    this,
    "presencePenalty",
    "Set the repetition alpha presence penalty (default: 0.0, 0.0 = disabled)")
  // Set MiroStat sampling strategies.
  //  val miroStat = MiroStat mirostat // enum DISABLED, V1, V2
  /** @group param */
  val miroStatTau = new FloatParam(
    this,
    "mirostatTau",
    "Set the MiroStat target entropy, parameter tau (default: 5.0)")

  /** @group param */
  val miroStatEta = new FloatParam(
    this,
    "mirostatEta",
    "Set the MiroStat learning rate, parameter eta (default: 0.1)")

  /** @group param */
  val penalizeNl = new BooleanParam(this, "penalizeNl", "Whether to penalize newline tokens")

  /** @group param */
  val nKeep = new IntParam(
    this,
    "nKeep",
    "Set the number of tokens to keep from the initial prompt (default: 0, -1 = all)")

  /** @group param */
  val seed = new IntParam(this, "seed", "Set the RNG seed (default: -1, use random seed for < 0)")

  /** @group param */
  val nProbs = new IntParam(
    this,
    "nProbs",
    "Set the amount top tokens probabilities to output if greater than 0.")

  /** @group param */
  val minKeep = new IntParam(
    this,
    "minKeep",
    "Set the amount of tokens the samplers should return at least (0 = disabled)")

  /** @group param */
  val grammar =
    new Param[String](this, "grammar", "Set BNF-like grammar to constrain generations")

  /** @group param */
  val penaltyPrompt = new Param[String](
    this,
    "penaltyPrompt",
    "Override which part of the prompt is penalized for repetition.")

  /** @group param */
//  val penaltyPrompt = new Param[String](this, "penaltyPrompt", "Override which part of the prompt is penalized for repetition.") // TODO: or provide token ids?
  /** @group param */
  val ignoreEos = new BooleanParam(
    this,
    "ignoreEos",
    "Set whether to ignore end of stream token and continue generating (implies --logit-bias 2-inf)")
  // Modify the likelihood of tokens appearing in the completion by their id.
// TODO:  val tokenIdBias: Map[Integer, Float]

  // Modify the likelihood of tokens appearing in the completion by their string.
// TODO:  val tokenBias: Map[String, Float]
  // 	 * Set tokens to disable, this corresponds to {@link #setTokenIdBias(Map)} with a value of
  //	 * {@link Float#NEGATIVE_INFINITY}.
  /** @group param */
  val disableTokenIds =
    new IntArrayParam(this, "disableTokenIds", "Set the token ids to disable in the completion")

  /** @group param */
  val stopStrings = new StringArrayParam(
    this,
    "stopStrings",
    "Set strings upon seeing which token generation is stopped")

  // Set which samplers to use for token generation in the given order
  // val samplers = Sampler... samplers // either TOP_K, TFS_Z, TYPICAL_P, TOP_P, MIN_P, TEMPERATURE
  /** @group param */
//  val stream = new BooleanParam(this, "stream", "Whether to stream the output or not")
  /** @group param */
  val useChatTemplate = new BooleanParam(
    this,
    "useChatTemplate",
    "Set whether or not generate should apply a chat template (default: false)")

  /** Set the prompt to start generation with (default: "")
    *
    * @group setParam
    */
  def setInputPrefix(inputPrefix: String) = { set(this.inputPrefix, inputPrefix) }

  /** Set a suffix for infilling (default: "")
    *
    * @group setParam
    */
  def setInputSuffix(inputSuffix: String) = { set(this.inputSuffix, inputSuffix) }

  /** Whether to remember the prompt to avoid reprocessing it
    *
    * @group setParam
    */
  def setCachePrompt(cachePrompt: Boolean) = { set(this.cachePrompt, cachePrompt) }

  /** Set the number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
    *
    * @group setParam
    */
  def setNPredict(nPredict: Int) = { set(this.nPredict, nPredict) }

  /** Set top-k sampling (default: 40, 0 = disabled)
    *
    * @group setParam
    */
  def setTopK(topK: Int) = { set(this.topK, topK) }

  /** Set top-p sampling (default: 0.9, 1.0 = disabled)
    *
    * @group setParam
    */
  def setTopP(topP: Float) = { set(this.topP, topP) }

  /** Set min-p sampling (default: 0.1, 0.0 = disabled)
    *
    * @group setParam
    */
  def setMinP(minP: Float) = { set(this.minP, minP) }

  /** Set tail free sampling, parameter z (default: 1.0, 1.0 = disabled)
    * @group setParam
    */
  def setTfsZ(tfsZ: Float) = { set(this.tfsZ, tfsZ) }

  /** Set locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)
    *
    * @group setParam
    */
  def setTypicalP(typicalP: Float) = { set(this.typicalP, typicalP) }

  /** Set the temperature (default: 0.8)
    *
    * @group setParam
    */
  def setTemperature(temperature: Float) = { set(this.temperature, temperature) }

  /** Set the dynamic temperature range (default: 0.0, 0.0 = disabled)
    *
    * @group setParam
    */
  def setDynamicTemperatureRange(dynatempRange: Float) = {
    set(this.dynamicTemperatureRange, dynatempRange)
  }
  def setDynamicTemperatureExponent(dynatempExponent: Float) = {
    set(this.dynamicTemperatureExponent, dynatempExponent)
  }

  /** Set the last n tokens to consider for penalties (default: 64, 0 = disabled, -1 = ctx_size)
    *
    * @group setParam
    */
  def setRepeatLastN(repeatLastN: Int) = { set(this.repeatLastN, repeatLastN) }

  /** Set the penalty of repeated sequences of tokens (default: 1.0, 1.0 = disabled)
    *
    * @group setParam
    */
  def setRepeatPenalty(repeatPenalty: Float) = { set(this.repeatPenalty, repeatPenalty) }

  /** Set the repetition alpha frequency penalty (default: 0.0, 0.0 = disabled)
    *
    * @group setParam
    */
  def setFrequencyPenalty(frequencyPenalty: Float) = {
    set(this.frequencyPenalty, frequencyPenalty)
  }

  /** Set the repetition alpha presence penalty (default: 0.0, 0.0 = disabled)
    *
    * @group setParam
    */
  def setPresencePenalty(presencePenalty: Float) = { set(this.presencePenalty, presencePenalty) }
// def setMiroStat(mirostat: MiroStat ) =  {set(this.mirostat, mirostat)}

  /** Set the MiroStat target entropy, parameter tau (default: 5.0)
    *
    * @group setParam
    */
  def setMiroStatTau(mirostatTau: Float) = { set(this.miroStatTau, mirostatTau) }

  /** Set the MiroStat learning rate, parameter eta (default: 0.1)
    *
    * @group setParam
    */
  def setMiroStatEta(mirostatEta: Float) = { set(this.miroStatEta, mirostatEta) }

  /** Set whether to penalize newline tokens
    *
    * @group setParam
    */
  def setPenalizeNl(penalizeNl: Boolean) = { set(this.penalizeNl, penalizeNl) }

  /** Set the number of tokens to keep from the initial prompt (default: 0, -1 = all)
    *
    * @group setParam
    */
  def setNKeep(nKeep: Int) = { set(this.nKeep, nKeep) }

  /** Set the RNG seed (default: -1, use random seed for < 0)
    *
    * @group setParam
    */
  def setSeed(seed: Int) = { set(this.seed, seed) }

  /** Set the amount top tokens probabilities to output if greater than 0.
    *
    * @group setParam
    */
  def setNProbs(nProbs: Int) = { set(this.nProbs, nProbs) }

  /** Set the amount of tokens the samplers should return at least (0 = disabled)
    *
    * @group setParam
    */
  def setMinKeep(minKeep: Int) = { set(this.minKeep, minKeep) }

  /** Set BNF-like grammar to constrain generations
    *
    * @group setParam
    */
  def setGrammar(grammar: String) = { set(this.grammar, grammar) }

  /** Override which part of the prompt is penalized for repetition.
    *
    * @group setParam
    */
  def setPenaltyPrompt(penaltyPrompt: String) = { set(this.penaltyPrompt, penaltyPrompt) }

// TODO?  def setPenaltyPrompt(tokens: Array[Int] ) =  {set(this.penaltyPrompt, tokens)}

  /** Set whether to ignore end of stream token and continue generating (implies --logit-bias
    * 2-inf)
    *
    * @group setParam
    */
  def setIgnoreEos(ignoreEos: Boolean) = { set(this.ignoreEos, ignoreEos) }
// TODO: def setTokenIdBias(Float: Map<Integer, > logitBias) =  {set(this.Float, Float)}
// TODO: def setTokenBias(Float: Map<String, > logitBias) =  {set(this.Float, Float)}

  /** Set the token ids to disable in the completion
    *
    * @group setParam
    */
  def setDisableTokenIds(disableTokenIds: Array[Int]) = {
    set(this.disableTokenIds, disableTokenIds)
  }

  /** Set strings upon seeing which token generation is stopped
    *
    * @group setParam
    */
  def setStopStrings(stopStrings: Array[String]) = { set(this.stopStrings, stopStrings) }
//  def setSamplers(samplers: Sampler... ) =  {set(this.samplers, samplers)}

  /** Whether or not to stream the output
    *
    * @group setParam
    */
  def setUseChatTemplate(useChatTemplate: Boolean) = {
    set(this.useChatTemplate, useChatTemplate)
  }

  // ---------------- MODEL PARAMETERS ----------------

  setDefault(
    inputPrefix -> "",
    inputSuffix -> "",
    cachePrompt -> true,
    nPredict -> -1,
    topK -> 40,
    topP -> 0.9f,
    minP -> 0.1f,
    tfsZ -> 1.0f,
    typicalP -> 1.0f,
    temperature -> 0.8f,
    dynamicTemperatureRange -> 0.0f,
    dynamicTemperatureExponent -> 1.0f,
    repeatLastN -> 64,
    repeatPenalty -> 1.0f,
    frequencyPenalty -> 0.0f,
    presencePenalty -> 0.0f,
    miroStatTau -> 5.0f,
    miroStatEta -> 0.1f,
    penalizeNl -> false,
    nKeep -> 0,
    seed -> -1,
    nProbs -> 0,
    minKeep -> 0,
    grammar -> "",
    penaltyPrompt -> "",
    ignoreEos -> false,
    disableTokenIds -> Array(),
    stopStrings -> Array(),
    useChatTemplate -> false)
}
