import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types._
import com.sksamuel.scrimage.ImmutableImage

object ImageConsumer {

  val INPUT_PATH  = "output"
  val OUTPUT_PATH = "results/preprocessed.parquet"
  val CHECKPOINT  = "checkpoint/consumer"

  val binarySchema: StructType = new StructType()
    .add("path",             StringType,    nullable = false)
    .add("modificationTime", TimestampType, nullable = false)
    .add("length",           LongType,      nullable = false)
    .add("content",          BinaryType,    nullable = true)

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("ChestX6-Consumer")
      .master("local[2]")
      .config("spark.sql.legacy.allowUntypedScalaUDF", "true")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    // ── 1. readStream ─────────────────────────────────────────────
    val imageStream = spark.readStream
      .format("binaryFile")
      .schema(binarySchema)
      .option("pathGlobFilter", "*.jpg")
      .option("recursiveFileLookup", "true")
      .option("latestFirst", "false")
      .option("maxFilesPerTrigger", "100")
      .load(INPUT_PATH)

    // ── 2. PARSE — extrait split et label ────────────────────────
    val pattern = """.*output/([^/]+)/([^/]+)/[^/]+$"""

    val parsed = imageStream
      .withColumn("split", regexp_extract(col("path"), pattern, 1))
      .withColumn("label", regexp_extract(col("path"), pattern, 2))

    // ── 3. TRANSFORM — resize 224×224 + normalise + vectorise ─────
    val transformUDF = udf((data: Array[Byte]) => {
      val resized = ImmutableImage.loader()
        .fromBytes(data)
        .scaleTo(224, 224)

      val pixels = resized.pixels()
      val vector = pixels.flatMap { pixel =>
        val r = ((pixel.red)   & 0xFF).toFloat / 255.0f
        val g = ((pixel.green) & 0xFF).toFloat / 255.0f
        val b = ((pixel.blue)  & 0xFF).toFloat / 255.0f
        Array(r, g, b)
      }
      vector
    })

    val transformed = parsed
      .withColumn("features", transformUDF(col("content")))
      .select(
        col("path"),
        col("split"),
        col("label"),
        col("features")
      )

    // ── 4. writeStream — sauvegarde en Parquet ────────────────────
    val query = transformed.writeStream
      .trigger(Trigger.ProcessingTime("5 seconds"))
      .outputMode("append")
      .format("parquet")
      .option("path", OUTPUT_PATH)
      .option("checkpointLocation", CHECKPOINT)
      .start()

    println("=== Consumer démarré — écriture vers results/ ===")
    query.awaitTermination()
  }
}