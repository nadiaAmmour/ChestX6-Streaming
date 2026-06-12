import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import java.io.File
import java.nio.file.{Files, Paths, StandardCopyOption}

object ImageProducer {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("ImageProducer")
      .master("local[2]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    val trainInput  = "Data/train"
    val valInput    = "Data/val"
    val trainOutput = "output/train"
    val valOutput   = "output/val"
    val batchSize   = 5
    val trigger     = "15 seconds"

    // Extensions d'images acceptées
    val imageExtensions = Set("jpg", "jpeg", "png", "bmp", "tiff")

    // Charger les images d'un dossier avec ses sous-dossiers maladies
    // Retourne Array[(cheminAbsolu, nomMaladie, outputBase)]
    def loadImages(baseDir: String, outputBase: String): Array[(String, String, String)] =
      new File(baseDir)
        .listFiles()
        .filter(_.isDirectory)
        .flatMap { diseaseDir =>
          // Créer le sous-dossier maladie dans output automatiquement
          val outDisease = new File(s"$outputBase/${diseaseDir.getName}")
          if (!outDisease.exists()) outDisease.mkdirs()

          diseaseDir.listFiles()
            .filter(f => f.isFile && imageExtensions.contains(
              f.getName.split("\\.").last.toLowerCase
            ))
            .sortBy(_.getName)
            .map(f => (f.getAbsolutePath, diseaseDir.getName, outputBase))
        }

    // train en premier (une seule fois), val ensuite (attend nouvelles images)
    val trainImages = loadImages(trainInput, trainOutput)
    val valImages   = loadImages(valInput, valOutput)

    // train passe en premier, val ensuite
    val allImages = trainImages ++ valImages

    println("=== ImageProducer démarré ===")
    println(s"Train       : ${trainImages.length} images")
    println(s"Val         : ${valImages.length} images")
    println(s"Total       : ${allImages.length} images")
    println(s"Batch size  : $batchSize")
    println(s"Trigger     : $trigger")
    println("=============================\n")

    // Diviser en batches de batchSize
    val batches      = allImages.grouped(batchSize).toSeq
    val trainBatches = math.ceil(trainImages.length.toDouble / batchSize).toInt

    // Streaming DataFrame avec rate source
    val imageDF = spark.readStream
      .format("rate")
      .option("rowsPerSecond", 1)
      .load()

    val query = imageDF.writeStream
      .trigger(Trigger.ProcessingTime(trigger))
      .foreachBatch { (batchDF: Dataset[Row], batchId: Long) =>

        // train passe une seule fois, val attend les nouvelles images
        val idx = if (batchId < trainBatches) {
          batchId.toInt
        } else {
          trainBatches + ((batchId - trainBatches) % (batches.length - trainBatches)).toInt
        }

        val batch = batches(idx)
        val phase = if (batchId < trainBatches) "TRAIN" else "VAL"

        println(s"--- Batch $batchId [$phase] : ${batch.length} image(s) ---")

        batch.foreach { case (filePath, diseaseDir, outputBase) =>
          val srcPath  = Paths.get(filePath)
          val fileName = srcPath.getFileName
          val destPath = Paths.get(s"$outputBase/$diseaseDir").resolve(fileName)
          Files.copy(srcPath, destPath, StandardCopyOption.REPLACE_EXISTING)
          println(s"  Copié : [$phase/$diseaseDir] $fileName")
        }
      }
      .start()

    query.awaitTermination()
  }
}