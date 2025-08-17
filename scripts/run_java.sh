#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../java"
mvn -q -DskipTests package
spark-submit \
  --master local[*] \
  --class bench.LangBenchJava \
  target/langbench-java-1.0.0.jar \
  --rows 2000000 --users 200000 --shuffle 600 --adaptive true --seed 42 \
  | tee ../results/java.log

