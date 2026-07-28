import org.apache.spark.{SparkConf, SparkContext}
import org.apache.hadoop.fs.{FileSystem, Path}
import java.io.File

object ImageProducer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("ImageProducer")
      .setMaster("local[2]")

    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    val fs = FileSystem.get(sc.hadoopConfiguration)

    val trainInput  = "Data/train"
    val valInput    = "Data/val"
    val trainOutput = "output/train"
    val valOutput   = "output/val"
    val batchSize   = 500
    val interval    = 2000 // 5 secondes

    // Extensions d'images acceptées
    val imageExtensions = Set("jpg", "jpeg", "png")

    // Charger les images d'un dossier avec ses sous-dossiers maladies
    def loadImages(baseDir: String, outputBase: String): Array[(String, String, String)] =
      new File(baseDir)
        .listFiles()
        .filter(_.isDirectory)
        .flatMap { diseaseDir =>
          val outDisease = new Path(s"$outputBase/${diseaseDir.getName}")
          if (!fs.exists(outDisease)) fs.mkdirs(outDisease)

          diseaseDir.listFiles()
            .filter(f => f.isFile && imageExtensions.contains(
              f.getName.split("\\.").last.toLowerCase
            ))
            .sortBy(_.getName)
            .map(f => (f.getAbsolutePath, diseaseDir.getName, outputBase))
        }

    val trainImages = loadImages(trainInput, trainOutput)
    val valImages   = loadImages(valInput, valOutput)
    val allImages   = trainImages ++ valImages

    println("=== ImageProducer demarre ===")
    println(s"Train       : ${trainImages.length} images")
    println(s"Val         : ${valImages.length} images")
    println(s"Total       : ${allImages.length} images")
    println(s"Batch size  : $batchSize")
    println(s"Intervalle  : ${interval/1000} secondes")
    println("=============================\n")

    // Traiter train puis val en une seule passe
    allImages.grouped(batchSize).zipWithIndex.foreach { case (batch, idx) =>
      val phase = if (batch.head._3 == trainOutput) "TRAIN" else "VAL"

      val rdd = sc.parallelize(batch)

      println(s"--- Batch $idx [$phase] : ${batch.length} image(s) ---")

      rdd.foreach { case (filePath, diseaseDir, outputBase) =>
        val src        = new Path(filePath)
        val dest       = new Path(s"$outputBase/$diseaseDir/${src.getName}")
        val hadoopFs   = FileSystem.get(new org.apache.hadoop.conf.Configuration())
        hadoopFs.copyFromLocalFile(src, dest)
        println(s"  Copiee : [$outputBase/$diseaseDir] ${src.getName}")
      }

      Thread.sleep(interval)
    }

    println("\n=== Toutes les images ont ete traitees. Producer arrete. ===")
    sc.stop()
  }
}