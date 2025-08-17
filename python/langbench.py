import argparse, time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=10_000_000)
    p.add_argument("--users", type=int, default=100_000)
    p.add_argument("--shuffle", type=int, default=400)
    p.add_argument("--adaptive", type=str, default="true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pandas_udf", action="store_true", help="also benchmark vectorized pandas UDF")
    return p.parse_args()

def build_spark(args):
    spark = (
        SparkSession.builder
        .appName("LangBench-PySpark")
        .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "localhost")
        .config("spark.driver.port", "0")
        .config("spark.blockManager.port", "0")
        .config("spark.ui.port", "0")
        .config("spark.sql.shuffle.partitions", args.shuffle)
        .config("spark.sql.adaptive.enabled", args.adaptive.lower() == "true")
        .config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
        .config("spark.executor.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    if args.pandas_udf:
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    return spark

def generate(spark, rows, users, seed):
    base_ts = F.to_timestamp(F.lit("2025-01-01"))

    rnd_days_txn = F.floor(F.rand(seed + 3) * 180).cast("int")
    ts = F.to_timestamp(F.date_add(F.to_date(F.lit("2025-01-01")), rnd_days_txn))

    txns = (spark.range(rows).select(
        F.col("id").alias("txn_id"),
        F.floor(F.rand(seed) * users).cast("long").alias("user_id"),
        (F.rand(seed + 1) * 1000.0).alias("amount"),
        F.floor(F.rand(seed + 2) * 100).cast("int").alias("category"),
        ts.alias("ts"),
    ))

    rnd_days_user = F.floor(F.rand(seed + 5) * 1825).cast("int")
    signup_ts = F.to_timestamp(F.date_add(F.lit("2020-01-01").cast("date"), rnd_days_user))

    users_df = (spark.range(users).select(
        F.col("id").alias("user_id"),
        F.when(F.rand(seed + 4) < 0.5, F.lit("US")).otherwise(F.lit("IN")).alias("country"),
        signup_ts.alias("signup_ts"),
    ))
    return txns, users_df

def workloadA(txns, users):
    joined = txns.join(users, "user_id")
    return (joined.groupBy(F.window(F.col("ts"), "7 days"), F.col("country"), F.col("category"))
            .agg(F.count(F.lit(1)).alias("txn_count"),
                 F.sum("amount").alias("amount_sum"),
                 F.avg("amount").alias("amount_avg"),
                 F.approx_count_distinct("user_id").alias("unique_users")))

def churn_py(s: str) -> int:
    acc = 0
    for ch in s:
        acc = (acc * 31 + ord(ch)) ^ (acc >> 13)
    return abs(acc)

def workloadB_udf(txns):
    churn_udf = F.udf(churn_py, IntegerType())
    sig = txns.select(churn_udf(F.col("txn_id").cast("string")).alias("sig"))
    return sig.groupBy("sig").count()

def workloadB_pandas_udf(txns):
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    @pandas_udf("int")
    def churn_pandas(col: pd.Series) -> pd.Series:
        out = []
        for s in col.astype(str):
            acc = 0
            for ch in s:
                acc = (acc * 31 + ord(ch)) ^ (acc >> 13)
            out.append(abs(acc))
        return pd.Series(out)
    sig = txns.select(churn_pandas(F.col("txn_id").cast("string")).alias("sig"))
    return sig.groupBy("sig").count()

def time_it(label, action):
    t0 = time.perf_counter()
    _ = action()
    t1 = time.perf_counter()
    ms = int((t1 - t0) * 1000)
    print(f"{label} took {ms} ms")
    return ms

if __name__ == "__main__":
    args = parse_args()
    spark = build_spark(args)
    txns, users = generate(spark, args.rows, args.users, args.seed)

    # warm-up
    txns.limit(1000).agg(F.sum("amount")).count()

    tA = time_it("WorkloadA_SQL", lambda: workloadA(txns, users).count())
    tB = time_it("WorkloadB_UDF(py)", lambda: workloadB_udf(txns).count())

    tBp = None
    if args.pandas_udf:
        tBp = time_it("WorkloadB_UDF(pandas)", lambda: workloadB_pandas_udf(txns).count())

    print(f"RESULT python rows={args.rows} users={args.users} shuffle={args.shuffle} "
          f"adaptive={args.adaptive} WorkloadA_ms={tA} WorkloadB_py_ms={tB}"
          + (f" WorkloadB_pandas_ms={tBp}" if tBp is not None else ""))

    spark.stop()

