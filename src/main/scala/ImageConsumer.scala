import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types._
import com.sksamuel.scrimage.ImmutableImage
import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import java.io.File
import scala.io.Source

// Singleton : charge le modele ONNX une seule fois par JVM (lazy = au 1er appel)
object OnnxPredictor {
  val MODEL_PATH  = "output/chestx6_model.onnx"
  val LABELS_PATH = "output/chestx6_labels.json"

  def isReady: Boolean = new File(MODEL_PATH).exists()

  lazy val classNames: Array[String] = {
    if (new File(LABELS_PATH).exists()) {
      val json = Source.fromFile(LABELS_PATH).mkString
      json.stripPrefix("[").stripSuffix("]")
        .split(",").map(_.trim.stripPrefix("\"").stripSuffix("\""))
    } else Array("class_0", "class_1", "class_2", "class_3")
  }

  lazy val env: OrtEnvironment = OrtEnvironment.getEnvironment()
  lazy val session: OrtSession =
    env.createSession(MODEL_PATH, new OrtSession.SessionOptions())

  // Predit le label (indice de la classe).
  def predict(features: Array[Float]): Int = {
    val tensor = OnnxTensor.createTensor(env, Array(features))
    val results = session.run(java.util.Collections.singletonMap("float_input", tensor))
    try {
      val labels = results.get(0).getValue.asInstanceOf[Array[Long]]
      labels(0).toInt
    } finally {
      results.close()
      tensor.close()
    }
  }

  // NOUVEAU (step SCORE) : predit le label ET le score de confiance (proba max).
  // La sortie "probabilities" (zipmap=False) est un tenseur [N, nbClasses].
  // On lit la ligne 0, on prend l'indice du max -> label, et la valeur max -> confiance.
  def predictWithScore(features: Array[Float]): (Int, Float) = {
    val tensor = OnnxTensor.createTensor(env, Array(features))
    val results = session.run(java.util.Collections.singletonMap("float_input", tensor))
    try {
      // results.get(1) = "probabilities" -> Array[Array[Float]] de forme [1, nbClasses]
      val probs = results.get(1).getValue.asInstanceOf[Array[Array[Float]]](0)
      var bestIdx = 0
      var bestVal = probs(0)
      var i = 1
      while (i < probs.length) {
        if (probs(i) > bestVal) { bestVal = probs(i); bestIdx = i }
        i += 1
      }
      (bestIdx, bestVal)
    } finally {
      results.close()
      tensor.close()
    }
  }

  def className(idx: Int): String =
    if (idx >= 0 && idx < classNames.length) classNames(idx) else "unknown"
}

object ImageConsumer {

  val TRAIN_INPUT       = "output/train"
  val VAL_INPUT         = "output/val"
  val UPLOAD_INPUT      = "output/upload"
  val TRAIN_OUTPUT      = "results/train.parquet"
  val VAL_OUTPUT        = "results/val.parquet"
  val UPLOAD_OUTPUT     = "results/upload.parquet"
  val TRAIN_CHECKPOINT  = "checkpoint/train"
  val VAL_CHECKPOINT    = "checkpoint/val"
  val UPLOAD_CHECKPOINT = "checkpoint/upload"

  val binarySchema: StructType = new StructType()
    .add("path",             StringType,    nullable = false)
    .add("modificationTime", TimestampType, nullable = false)
    .add("length",           LongType,      nullable = false)
    .add("content",          BinaryType,    nullable = true)

  // Resize 224x224 + normalisation RGB [0,1] + flatten -> Array[Float] (150 528 valeurs)
  // ROBUSTE : une image illisible renvoie null (pas de crash de la query).
  val transformUDF = udf((data: Array[Byte]) => {
    try {
      val resized = ImmutableImage.loader().fromBytes(data).scaleTo(224, 224)
      resized.pixels().flatMap { pixel =>
        Array(
          (pixel.red   & 0xFF).toFloat / 255.0f,
          (pixel.green & 0xFF).toFloat / 255.0f,
          (pixel.blue  & 0xFF).toFloat / 255.0f
        )
      }
    } catch {
      case _: Throwable => null
    }
  })

  // Prediction du label seul (val : on garde le comportement d'avant)
  val predictUDF = udf((features: Seq[Float]) => {
    if (OnnxPredictor.isReady) {
      try OnnxPredictor.predict(features.toArray) catch { case _: Throwable => -1 }
    } else -1
  })

  // NOUVEAU : prediction label + score, renvoyes ensemble sous forme (idx, score).
  // Struct Spark : on renvoie un tuple -> colonnes .prediction et .score.
  val predictScoreUDF = udf((features: Seq[Float]) => {
    if (OnnxPredictor.isReady) {
      try {
        val (idx, score) = OnnxPredictor.predictWithScore(features.toArray)
        (idx, score)
      } catch { case _: Throwable => (-1, 0.0f) }
    } else (-1, 0.0f)
  })

  val predictNameUDF = udf((idx: Int) => {
    if (idx >= 0 && OnnxPredictor.isReady) OnnxPredictor.className(idx) else "unknown"
  })

  // Pipeline TRAIN : READ -> PARSE -> TRANSFORM -> WRITE (features seulement)
  def buildTrainPipeline(spark: SparkSession): DataFrame = {
    val stream = spark.readStream
      .format("binaryFile")
      .schema(binarySchema)
      .option("pathGlobFilter",      "*.{jpg,jpeg,png}")
      .option("recursiveFileLookup", "true")
      .option("latestFirst",         "false")
      .option("maxFilesPerTrigger",  "200")
      .load(TRAIN_INPUT)

    val labelPattern = ".*/train/([^/]+)/[^/]+$"

    stream
      .withColumn("split",    lit("train"))
      .withColumn("label",    regexp_extract(col("path"), labelPattern, 1))
      .withColumn("features", transformUDF(col("content")))
      .filter(col("features").isNotNull)
      .select("path", "split", "label", "features")
  }

  // Pipeline VAL : READ -> PARSE -> TRANSFORM -> SCORE -> PREDICT -> WRITE
  def buildValPipeline(spark: SparkSession): DataFrame = {
    val stream = spark.readStream
      .format("binaryFile")
      .schema(binarySchema)
      .option("pathGlobFilter",      "*.{jpg,jpeg,png}")
      .option("recursiveFileLookup", "true")
      .option("latestFirst",         "false")
      .option("maxFilesPerTrigger",  "50")
      .load(VAL_INPUT)

    val labelPattern = ".*/val/([^/]+)/[^/]+$"

    stream
      .withColumn("split",           lit("val"))
      .withColumn("label",           regexp_extract(col("path"), labelPattern, 1))
      .withColumn("features",        transformUDF(col("content")))
      .filter(col("features").isNotNull)
      .withColumn("ps",              predictScoreUDF(col("features")))
      .withColumn("prediction",      col("ps._1"))
      .withColumn("score",           col("ps._2"))
      .withColumn("prediction_name", predictNameUDF(col("prediction")))
      .select("path", "split", "label", "prediction", "prediction_name", "score")
  }

  // Pipeline UPLOAD (medecin) : READ -> TRANSFORM -> SCORE -> PREDICT -> WRITE
  // Diagnostic a l'aveugle : pas de label, mais on garde le SCORE de confiance.
  def buildUploadPipeline(spark: SparkSession): DataFrame = {
    val stream = spark.readStream
      .format("binaryFile")
      .schema(binarySchema)
      .option("pathGlobFilter",      "*.{jpg,jpeg,png}")
      .option("recursiveFileLookup", "true")
      .option("latestFirst",         "false")
      .option("maxFilesPerTrigger",  "5")
      .load(UPLOAD_INPUT)

    stream
      .withColumn("split",           lit("upload"))
      .withColumn("features",        transformUDF(col("content")))
      .filter(col("features").isNotNull)
      .withColumn("ps",              predictScoreUDF(col("features")))
      .withColumn("prediction",      col("ps._1"))
      .withColumn("score",           col("ps._2"))
      .withColumn("prediction_name", predictNameUDF(col("prediction")))
      .select("path", "split", "prediction", "prediction_name", "score")
  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("ChestX6-Consumer")
      .master("local[2]")
      .config("spark.sql.legacy.allowUntypedScalaUDF", "true")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    Seq(TRAIN_INPUT, VAL_INPUT, UPLOAD_INPUT).foreach { d =>
      val dir = new File(d)
      if (!dir.exists()) dir.mkdirs()
    }

    println("=== Consumer demarre ===")
    println(s"  TRAIN  -> $TRAIN_OUTPUT   (features seulement)")
    println(s"  VAL    -> $VAL_OUTPUT     (prediction + score ONNX)")
    println(s"  UPLOAD -> $UPLOAD_OUTPUT  (prediction + score, diagnostic a l'aveugle)")
    if (OnnxPredictor.isReady) {
      println(s"  Modele ONNX charge : ${OnnxPredictor.MODEL_PATH}")
      println(s"  Classes            : ${OnnxPredictor.classNames.mkString(", ")}")
    } else {
      println(s"  Pas de modele ONNX (${OnnxPredictor.MODEL_PATH} absent)")
      println("  -> colonne 'prediction' remplie avec -1 en attendant l'entrainement")
    }
    println("========================\n")

    val queryTrain = buildTrainPipeline(spark).writeStream
      .trigger(Trigger.ProcessingTime("5 seconds"))
      .outputMode("append")
      .format("parquet")
      .option("path",               TRAIN_OUTPUT)
      .option("checkpointLocation", TRAIN_CHECKPOINT)
      .start()

    val queryVal = buildValPipeline(spark).writeStream
      .trigger(Trigger.ProcessingTime("5 seconds"))
      .outputMode("append")
      .format("parquet")
      .option("path",               VAL_OUTPUT)
      .option("checkpointLocation", VAL_CHECKPOINT)
      .start()

    val queryUpload = buildUploadPipeline(spark).writeStream
      .trigger(Trigger.ProcessingTime("5 seconds"))
      .outputMode("append")
      .format("parquet")
      .option("path",               UPLOAD_OUTPUT)
      .option("checkpointLocation", UPLOAD_CHECKPOINT)
      .start()

    spark.streams.awaitAnyTermination()
  }
}

