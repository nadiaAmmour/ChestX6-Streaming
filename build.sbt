name := "ChestX6-Streaming"
version := "1.0"
scalaVersion := "2.12.15"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"   % "3.3.2",
  "org.apache.spark" %% "spark-sql"    % "3.3.2",
  "org.apache.spark" %% "spark-mllib"  % "3.3.2",
  "com.sksamuel.scrimage" % "scrimage-core" % "4.0.31",
  "com.microsoft.onnxruntime" % "onnxruntime" % "1.16.3"
)