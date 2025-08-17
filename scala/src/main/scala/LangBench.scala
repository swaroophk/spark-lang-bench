import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object LangBench {
  case class Conf(rows: Long = 10_000_000L,
                  users: Long = 100_000L,
                  shuffle: Int = 400,
                  adaptive: Boolean = true,
                  seed: Int = 42)

  def main(args: Array[String]): Unit = {
    val argMap = args.sliding(2, 2).collect { case Array(k, v) => k.stripPrefix("--") -> v }.toMap
    val conf = Conf(
      rows = argMap.get("rows").map(_.toLong).getOrElse(10_000_000L),
      users = argMap.get("users").map(_.toLong).getOrElse(100_000L),
      shuffle = argMap.get("shuffle").map(_.toInt).getOrElse(400),
      adaptive = argMap.get("adaptive").forall(_.toBoolean),
      seed = argMap.get("seed").map(_.toInt).getOrElse(42)
    )

    val spark = SparkSession.builder()
      .appName("LangBench-Scala")
      .master("local[*]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "localhost")
      .config("spark.driver.port", "0")
      .config("spark.blockManager.port", "0")
      .config("spark.ui.port", "0")
      .config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
      .config("spark.executor.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
      .config("spark.sql.shuffle.partitions", conf.shuffle)
      .config("spark.sql.adaptive.enabled", conf.adaptive.toString)
      .getOrCreate()


    spark.sparkContext.setLogLevel("WARN")

    val (txns, users) = generateData(spark, conf.rows, conf.users, conf.seed)

    // Warm-up to trigger JIT
    txns.limit(1000).agg(sum("amount")).count()

    val tA = timeIt("WorkloadA_SQL") {
      workloadA(spark, txns, users).count()
    }
    val tB = timeIt("WorkloadB_UDF") {
      workloadB_UDF(spark, txns).count()
    }
    println(s"RESULT scala rows=${conf.rows} users=${conf.users} shuffle=${conf.shuffle} " +
      s"adaptive=${conf.adaptive} WorkloadA_ms=$tA WorkloadB_ms=$tB")

    spark.stop()
  }

  def generateData(spark: SparkSession, nRows: Long, nUsers: Long, seed: Int): (DataFrame, DataFrame) = {
    import spark.implicits._

    val baseDate = to_date(lit("2025-01-01"))
    val signup0 = to_date(lit("2020-01-01"))

    val txns = spark.range(nRows).select(
      col("id").as("txn_id"),
      floor(rand(seed) * nUsers.toDouble).cast("long").as("user_id"),
      (rand(seed + 1) * 1000d).cast("double").as("amount"),
      floor(rand(seed + 2) * 100).cast("int").as("category"),
      // Random day offset [0, 179] from 2025-01-01 @ 00:00:00
      to_timestamp(
        date_add(baseDate, floor(rand(seed + 3) * 180).cast("int"))
      ).as("ts")
    )

    val users = spark.range(nUsers).select(
      col("id").as("user_id"),
      when(rand(seed + 4) < 0.5, lit("US")).otherwise(lit("IN")).as("country"),
      // Random day offset [0, 1824] from 2020-01-01 @ 00:00:00
      to_timestamp(
        date_add(signup0, floor(rand(seed + 5) * 1825).cast("int"))
      ).as("signup_ts")
    )

    (txns, users)
  }


  def workloadA(spark: SparkSession, txns: DataFrame, users: DataFrame): DataFrame = {
    val joined = txns.join(users, "user_id")
    joined
      .groupBy(window(col("ts"), "7 days"), col("country"), col("category"))
      .agg(
        count(lit(1)).as("txn_count"),
        sum("amount").as("amount_sum"),
        avg("amount").as("amount_avg"),
        approx_count_distinct("user_id").as("unique_users")
      )
  }

  private def churn(s: String): Int = {
    var acc = 0
    var i = 0
    val n = s.length
    while (i < n) {
      acc = (acc * 31 + s.charAt(i)) ^ (acc >>> 13)
      i += 1
    }
    math.abs(acc)
  }

  def workloadB_UDF(spark: SparkSession, txns: DataFrame): DataFrame = {
    val churnUdf = udf((s: String) => churn(Option(s).getOrElse("")))

    val sigDf = txns.select(churnUdf(col("txn_id").cast("string")).as("sig"))
    sigDf.groupBy("sig").count()
  }

  private def timeIt[T](label: String)(thunk: => T): Long = {
    val t0 = System.nanoTime()
    val _ = thunk
    val t1 = System.nanoTime()
    val ms = ((t1 - t0) / 1e6).toLong
    println(s"$label took ${ms} ms")
    ms
  }
}
