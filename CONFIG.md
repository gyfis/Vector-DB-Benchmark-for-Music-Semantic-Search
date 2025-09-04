# Database Configuration

This document describes how to configure the new PostgreSQL and ClickHouse databases for the benchmark.

## Environment Variables

Create a `.env` file in the project root with the following variables:

### PostgreSQL (pgvector)
```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=music_vectors
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
```

### ClickHouse
```bash
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=9000
CLICKHOUSE_DB=default
```

### Existing databases
```bash
QDRANT_URL=http://localhost:6333
MILVUS_HOST=localhost
MILVUS_PORT=19530
WEAVIATE_URL=http://localhost:8080
SQLITE_DB_PATH=music_vectors.db
```

## Running the Benchmark

To include the new databases in your benchmark, use these database names:

- `postgres` - PostgreSQL with pgvector and HNSW
- `clickhouse-mergetree` - ClickHouse with regular MergeTree table
- `clickhouse-float16` - ClickHouse with Float32 vectors in partitioned table (closest to Float16B)
- `clickhouse-float32` - ClickHouse with Float32 vectors in partitioned table
- `clickhouse-float64` - ClickHouse with Float64 vectors in partitioned table

Example command:
```bash
python benchmark.py --csv data/muse.csv --embeddings data/embeddings.parquet --dbs postgres clickhouse-mergetree clickhouse-float32 --topk 10
```

## Docker Setup

Start all services with:
```bash
cd scripts
docker-compose up -d
```

This will start:
- PostgreSQL with pgvector extension
- ClickHouse 25.8 with vector search capabilities
- All existing databases (Qdrant, Milvus, Weaviate, etc.)
