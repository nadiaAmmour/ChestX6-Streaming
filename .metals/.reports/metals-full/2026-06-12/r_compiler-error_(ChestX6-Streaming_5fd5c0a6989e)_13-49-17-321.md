error id: 7B24F233620AC68017BC6B6C3469744D
file:///C:/Users/nadia/.git/ChestX6-Streaming/src/Consumer/Imageproducer.scala
### dotty.tools.dotc.core.UnpicklingError: Could not read definition method wrapDoubleArray in <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-lang\scala-library\3.8.2\scala-library-3.8.2.jar(scala/LowPriorityImplicits.tasty). Caused by the following exception:
java.lang.AssertionError: assertion failed: `-Xread-docs` enabled, but no `docCtx` is set.

Run with -Ydebug-unpickling to see full stack trace.

occurred in the presentation compiler.



action parameters:
offset: 3479
uri: file:///C:/Users/nadia/.git/ChestX6-Streaming/src/Consumer/Imageproducer.scala
text:
```scala
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

    val trainInput  = "Data/train"
    val valInput    = "Data/val"
    val trainOutput = "output/train"
    val valOutput   = "output/val"
    val batchSize   = 5
    val trigger     = "15 seconds"

    // Extensions d'images acceptées
    val imageExtensions = Set("jpg", "jpeg", "png")

    // Charger les images d'un dossier avec ses sous-dossiers maladies
    // Retourne Array[(cheminAbsolu, nomMaladie)]
    def loadImages(baseDir: String): Array[(String, String)] =
      new File(baseDir)
        .listFiles()
        .filter(_.isDirectory)
        .flatMap { diseaseDir =>
          // Créer le sous-dossier maladie dans output automatiquement
          val outBase = if (baseDir == trainInput) trainOutput else valOutput
          val outDisease = new File(s"$outBase/${diseaseDir.getName}")
          if (!outDisease.exists()) outDisease.mkdirs()

          diseaseDir.listFiles()
            .filter(f => f.isFile && imageExtensions.contains(
              f.getName.split("\\.").last.toLowerCase
            ))
            .sortBy(_.getName)
            .map(f => (f.getAbsolutePath, diseaseDir.getName))
        }

    // train en premier (une seule fois), val ensuite (boucle)
    val trainImages = loadImages(trainInput)
    val valImages   = loadImages(valInput)

    // (cheminAbsolu, nomMaladie, dossierOutput)
    val allImages = trainImages.map { case (p, d) => (p, d, trainOutput) } ++
                    valImages.map   { case (p, d) => (p, d, valOutput)   }

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

    val streamDF = spark.readStream
      .format("rate")
      .option("rowsPerSecond", 1)
      .load()

    val query = streamDF.writeStream
      .trigger(Trigger.ProcessingTime(trigger))
      .foreachBatch { (batchDF: Dataset[Row], batchId: Long) =>

        // train passe une seule fois, val boucle indéfiniment
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
          val destPath = Paths.get(s"$outputBase/$diseaseDir").res@@olve(fileName)
          Files.copy(srcPath, destPath, StandardCopyOption.REPLACE_EXISTING)
          println(s"  Copié : [$phase/$diseaseDir] $fileName")
        }
      }
      .start()

    query.awaitTermination()
  }
}
```


presentation compiler configuration:
Scala version: 3.8.2-bin-nonbootstrapped
Classpath:
<WORKSPACE>\.scala-build\ChestX6-Streaming_d5c0a6989e\classes\main [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-lang\scala3-library_3\3.8.2\scala3-library_3-3.8.2.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-lang\scala-library\3.8.2\scala-library-3.8.2.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\sourcegraph\semanticdb-javac\0.10.0\semanticdb-javac-0.10.0.jar [exists ], <WORKSPACE>\.scala-build\ChestX6-Streaming_d5c0a6989e\classes\main\META-INF\best-effort [missing ]
Options:
-Xsemanticdb -sourceroot <WORKSPACE> -release 11 -Ywith-best-effort-tasty




#### Error stacktrace:

```

```
#### Short summary: 

dotty.tools.dotc.core.UnpicklingError: Could not read definition method wrapDoubleArray in <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-lang\scala-library\3.8.2\scala-library-3.8.2.jar(scala/LowPriorityImplicits.tasty). Caused by the following exception:
java.lang.AssertionError: assertion failed: `-Xread-docs` enabled, but no `docCtx` is set.

Run with -Ydebug-unpickling to see full stack trace.