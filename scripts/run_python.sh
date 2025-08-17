#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../python"
spark-submit \
  --master local[*] \
  langbench.py \
  --rows 2000000 --users 200000 --shuffle 600 --adaptive true --seed 42 --pandas_udf \
  | tee ../results/python.log

