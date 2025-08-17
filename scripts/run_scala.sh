#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../scala"
sbt clean package
spark-submit \
  --master local[*] \
  --class LangBench \
  target/scala-2.13/*.jar \
  --rows 2000000 --users 200000 --shuffle 600 --adaptive true --seed 42 \
  | tee ../results/scala.log

