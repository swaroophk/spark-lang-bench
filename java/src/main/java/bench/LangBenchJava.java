package bench;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.sql.*;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.*;

public class LangBenchJava {

    static class Conf {
        long rows = 10_000_000L;
        long users = 100_000L;
        int shuffle = 400;
        boolean adaptive = true;
        int seed = 42;
    }

    public static void main(String[] args) {
        Map<String, String> argMap = new HashMap<>();
        for (int i = 0; i + 1 < args.length; i += 2) {
            argMap.put(args[i].replaceFirst("^--", ""), args[i + 1]);
        }
        Conf conf = new Conf();
        if (argMap.containsKey("rows")) conf.rows = Long.parseLong(argMap.get("rows"));
        if (argMap.containsKey("users")) conf.users = Long.parseLong(argMap.get("users"));
        if (argMap.containsKey("shuffle")) conf.shuffle = Integer.parseInt(argMap.get("shuffle"));
        if (argMap.containsKey("adaptive")) conf.adaptive = Boolean.parseBoolean(argMap.get("adaptive"));
        if (argMap.containsKey("seed")) conf.seed = Integer.parseInt(argMap.get("seed"));

        SparkSession spark = SparkSession.builder()
                .appName("LangBench-Java")
                .master("local[*]")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .config("spark.driver.host", "localhost")
                .config("spark.driver.port", "0")
                .config("spark.blockManager.port", "0")
                .config("spark.ui.port", "0")
                .config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
                .config("spark.executor.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
                .config("spark.sql.shuffle.partitions", conf.shuffle)
                .config("spark.sql.adaptive.enabled", Boolean.toString(conf.adaptive))
                .getOrCreate();

        spark.sparkContext().setLogLevel("WARN");

        Dataset<Row> txns = generateTxns(spark, conf.rows, conf.users, conf.seed);
        Dataset<Row> users = generateUsers(spark, conf.users, conf.seed);

        System.out.println("Transactions: " + txns.count());
        System.out.println("Users: " + users.count());

        // Warm-up
        txns.limit(1000).agg(sum("amount")).count();

        long tA = timeIt("WorkloadA_SQL", () -> workloadA(spark, txns, users).count());
        long tB = timeIt("WorkloadB_UDF", () -> workloadB_UDF(spark, txns).count());

        System.out.printf("RESULT java rows=%d users=%d shuffle=%d adaptive=%s WorkloadA_ms=%d WorkloadB_ms=%d%n",
                conf.rows, conf.users, conf.shuffle, conf.adaptive, tA, tB);

        spark.stop();
    }

    static Dataset<Row> generateTxns(SparkSession spark, long nRows, long nUsers, int seed) {
        // random 0..179 days
        Column rndDays = floor(rand(seed + 3).multiply(180)).cast("int");

        return spark.range(nRows).select(
                col("id").alias("txn_id"),
                floor(rand(seed).multiply(nUsers)).cast("long").alias("user_id"),
                rand(seed + 1).multiply(1000.0).alias("amount"),
                floor(rand(seed + 2).multiply(100)).cast("int").alias("category"),
                to_timestamp(date_add(to_date(lit("2025-01-01")), rndDays)).alias("ts")
        );
    }

    static Dataset<Row> generateUsers(SparkSession spark, long nUsers, int seed) {
        Column rndDays = floor(rand(seed + 5).multiply(1825)).cast("int"); // ~5 years

        return spark.range(nUsers).select(
                col("id").alias("user_id"),
                when(rand(seed + 4).lt(0.5), lit("US")).otherwise(lit("IN")).alias("country"),
                to_timestamp(date_add(lit("2020-01-01").cast("date"), rndDays)).alias("signup_ts")
        );
    }

    static Dataset<Row> workloadA(SparkSession spark, Dataset<Row> txns, Dataset<Row> users) {
        Dataset<Row> joined = txns.join(users, "user_id");
        return joined.groupBy(window(col("ts"), "7 days"), col("country"), col("category"))
                .agg(
                        count(lit(1)).alias("txn_count"),
                        sum(col("amount")).alias("amount_sum"),
                        avg(col("amount")).alias("amount_avg"),
                        approx_count_distinct(col("user_id")).alias("unique_users")
                );
    }

    static int churnJava(String s) {
        int acc = 0;
        for (int i = 0; i < s.length(); i++) {
            acc = (acc * 31 + s.charAt(i)) ^ (acc >>> 13);
        }
        return Math.abs(acc);
    }

    static Dataset<Row> workloadB_UDF(SparkSession spark, Dataset<Row> txns) {
        UserDefinedFunction churn = udf((String s) -> churnJava(s), DataTypes.IntegerType);
        Dataset<Row> sigDf = txns.select(churn.apply(col("txn_id").cast("string")).alias("sig"));
        return sigDf.groupBy("sig").count();
    }

    interface Thunk {
        void run();
    }

    static long timeIt(String label, Runnable r) {
        long t0 = System.nanoTime();
        r.run();
        long t1 = System.nanoTime();
        long ms = (long) ((t1 - t0) / 1e6);
        System.out.printf("%s took %d ms%n", label, ms);
        return ms;
    }
}

