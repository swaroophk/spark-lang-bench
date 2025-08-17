# Spark Language Benchmark

Compare Spark ETL performance across Scala, Java, and Python.

## Layout
- `scala/` → Spark app in Scala (SBT project).
- `java/` → Spark app in Java (Maven project).
- `python/` → Spark app in PySpark.
- `scripts/` → Convenience shell scripts for building/running all benchmarks.
- `results/` → Collected benchmark logs.

## Requirements
- Spark 3.5+
- Java 11+
- Scala 2.12.x
- Python 3.9+ with PySpark

## Usage
Run individually:
```bash
./scripts/run_scala.sh
./scripts/run_java.sh
./scripts/run_python.sh
````
## Analyze & Plot

After running one or more benchmarks, parse logs and create CSV + plots:

```bash
python3 scripts/parse_and_plot.py \
  --logs results/*.log \
  --out-csv results/bench.csv \
  --out-png results/bench.png
````

**Dependencies**
```bash
python3 -m pip install matplotlib
```

