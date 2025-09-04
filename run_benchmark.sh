#!/bin/bash

source .venv/bin/activate
python benchmark.py --csv data/muse.csv --embeddings data/embeddings.parquet --dbs qdrant weaviate postgres clickhouse-mergetree clickhouse-float16 clickhouse-float32 clickhouse-float64 --topk 10 --repetitions 5
