import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.size

object CheckParquet {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("CheckParquet")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val df = spark.read.parquet("results/preprocessed.parquet")

    println(s"Nombre d'images : ${df.count()}")
    println(s"Colonnes : ${df.columns.mkString(", ")}")

    df.select("path", "split", "label").show(10, truncate = false)

    df.select(size(df("features")).alias("taille_vecteur")).show(5)

    spark.stop()
  }
}