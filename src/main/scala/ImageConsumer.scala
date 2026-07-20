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

  def className(idx: Int): String =
    if (idx >= 0 && idx < classNames.length) classNames(idx) else "unknown"
}

object ImageConsumer {

  val TRAIN_INPUT      = "output/train"
  val VAL_INPUT        = "output/val"
  val TRAIN_OUTPUT     = "results/train.parquet"
  val VAL_OUTPUT       = "results/val.parquet"
  val TRAIN_CHECKPOINT = "checkpoint/train"
  val VAL_CHECKPOINT   = "checkpoint/val"

  val binarySchema: StructType = new StructType()
    .add("path",             StringType,    nullable = false)
    .add("modificationTime", TimestampType, nullable = false)
    .add("length",           LongType,      nullable = false)
    .add("content",          BinaryType,    nullable = true)

  // Resize 224x224 + normalisation RGB [0,1] + flatten -> Array[Float] (150 528 valeurs)
  val transformUDF = udf((data: Array[Byte]) => {
    val resized = ImmutableImage.loader().fromBytes(data).scaleTo(224, 224)
    resized.pixels().flatMap { pixel =>
      Array(
        ((pixel.red)   & 0xFF).toFloat / 255.0f,
        ((pixel.green) & 0xFF).toFloat / 255.0f,
        ((pixel.blue)  & 0xFF).toFloat / 255.0f
      )
    }
  })

  // UDFs de prediction ONNX (utilisees uniquement pour val)
  val predictUDF = udf((features: Seq[Float]) => {
    if (OnnxPredictor.isReady) {
      try OnnxPredictor.predict(features.toArray) catch { case _: Throwable => -1 }
    } else -1
  })

  val predictNameUDF = udf((idx: Int) => {
    if (idx >= 0 && OnnxPredictor.isReady) OnnxPredictor.className(idx) else "unknown"
  })

  // Pipeline TRAIN : READ -> PARSE -> TRANSFORM -> WRITE (pas de predict)
  def buildTrainPipeline(spark: SparkSession): DataFrame = {
    val stream = spark.readStream
      .format("binaryFile")
      .schema(binarySchema)
      .option("pathGlobFilter",     "*.{jpg,jpeg,png}")
      .option("recursiveFileLookup", "true")
      .option("latestFirst",         "false")
      .option("maxFilesPerTrigger",  "5")
      .load(TRAIN_INPUT)

    val labelPattern = ".*/train/([^/]+)/[^/]+$"

    stream
      .withColumn("split",    lit("train"))
      .withColumn("label",    regexp_extract(col("path"), labelPattern, 1))
      .withColumn("features", transformUDF(col("content")))
      .select("path", "split", "label", "features")
  }

  // Pipeline VAL : READ -> PARSE -> TRANSFORM -> SCORE -> PREDICT -> WRITE
  def buildValPipeline(spark: SparkSession): DataFrame = {
    val stream = spark.readStream
      .format("binaryFile")
      .schema(binarySchema)
      .option("pathGlobFilter",     "*.{jpg,jpeg,png}")
      .option("recursiveFileLookup", "true")
      .option("latestFirst",         "false")
      .option("maxFilesPerTrigger",  "5")
      .load(VAL_INPUT)

    val labelPattern = ".*/val/([^/]+)/[^/]+$"

    stream
      .withColumn("split",           lit("val"))
      .withColumn("label",           regexp_extract(col("path"), labelPattern, 1))
      .withColumn("features",        transformUDF(col("content")))
      .withColumn("prediction",      predictUDF(col("features")))
      .withColumn("prediction_name", predictNameUDF(col("prediction")))
      .select("path", "split", "label", "features", "prediction", "prediction_name")
  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("ChestX6-Consumer")
      .master("local[2]")
      .config("spark.sql.legacy.allowUntypedScalaUDF", "true")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    println("=== Consumer demarre ===")
    println(s"  TRAIN -> $TRAIN_OUTPUT   (features seulement)")
    println(s"  VAL   -> $VAL_OUTPUT     (features + prediction ONNX)")
    if (OnnxPredictor.isReady) {
      println(s"  Modele ONNX charge : ${OnnxPredictor.MODEL_PATH}")
      println(s"  Classes           : ${OnnxPredictor.classNames.mkString(", ")}")
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

    spark.streams.awaitAnyTermination()
  }
}
