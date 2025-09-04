from typing import List, Dict, Any
from clickhouse_driver import Client
from .base import VectorDB
import json
import time
from datetime import datetime, timedelta
import uuid


class ClickHouse(VectorDB):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9009,
        database: str = "default",
        table: str = "music_embeddings",
        variant: str = "mergetree"  # mergetree, float16 (BFloat16), float32, float64, bf16 (BFloat16), bf32, bf64
    ):
        self.host = host
        self.port = port
        self.database = database
        self.table = table
        self.variant = variant
        self.client = None

    def close(self):
        if self.client:
            self.client.disconnect()
            self.client = None

    def _get_client(self):
        if self.client is None:
            self.client = Client(host=self.host, port=self.port, database=self.database)
        return self.client

    def _get_float_type(self):
        """Get the element float type for the vector column based on variant."""
        if self.variant in ("float16", "bf16"):
            return "BFloat16"  # Use native BFloat16 in ClickHouse
        elif self.variant in ("float32", "bf32"):
            return "Float32"
        elif self.variant in ("float64", "bf64"):
            return "Float64"
        else:
            return "Float32"  # Default for mergetree

    def _use_vector_index(self) -> bool:
        # Indexed variants (HNSW) vs brute-force (bf*)
        return self.variant in ("mergetree", "float16", "float32", "float64")

    def _unique_table_name(self, base_name: str) -> str:
        """Generate a unique table name to avoid overwriting previous runs."""
        client = self._get_client()
        # Try timestamp + short uuid suffix
        for _ in range(5):
            suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:6]
            candidate = f"{base_name}_{suffix}"
            exists = client.execute(
                "SELECT count() FROM system.tables WHERE database = currentDatabase() AND name = %(n)s",
                {"n": candidate},
            )[0][0]
            if exists == 0:
                return candidate
        # Fallback to random uuid
        return f"{base_name}_{uuid.uuid4().hex[:8]}"

    def _create_table_mergetree(self, dim: int):
        """Create MergeTree table. For bf* variants, use the requested element type; otherwise Float32."""
        client = self._get_client()

        element_type = self._get_float_type() if self.variant.startswith("bf") else "Float32"

        index_clause = (
            f"INDEX vec_idx vector TYPE vector_similarity('hnsw', 'cosineDistance', {dim}),"
            if self._use_vector_index()
            else ""
        )

        # Ensure unique table name per setup to avoid overwrites
        base = self.table
        self.table = self._unique_table_name(base)

        create_query = f"""
        CREATE TABLE {self.table} (
            id UInt32,
            row_id UInt32,
            vector Array({element_type}),
            track String,
            artist String,
            genre String,
            seeds String,
            text String{',' if index_clause else ''}
            {index_clause}
        ) ENGINE = MergeTree()
        ORDER BY id
        SETTINGS index_granularity = 8192
        """

        client.execute(create_query)

    def _create_table_partitioned(self, dim: int, float_type: str):
        """Create partitioned table with dt column for time-based partitioning."""
        client = self._get_client()

        index_clause = (
            f"INDEX vec_idx vector TYPE vector_similarity('hnsw', 'cosineDistance', {dim}),"
            if self._use_vector_index()
            else ""
        )

        # Ensure unique table name per setup to avoid overwrites
        base = self.table
        self.table = self._unique_table_name(base)

        create_query = f"""
        CREATE TABLE {self.table} (
            dt DateTime64(6),
            id UInt32,
            row_id UInt32,
            vector Array({float_type}),
            track String,
            artist String,
            genre String,
            seeds String,
            text String{',' if index_clause else ''}
            {index_clause}
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMMDD(dt)
        ORDER BY (dt, id)
        TTL dt + INTERVAL 30 DAY
        SETTINGS index_granularity = 8192
        """

        client.execute(create_query)

    def setup(self, dim: int):
        if self.variant == "mergetree" or self.variant.startswith("bf"):
            self._create_table_mergetree(dim)
        else:
            float_type = self._get_float_type()
            self._create_table_partitioned(dim, float_type)

    def _generate_timestamps(self, num_records: int):
        """Generate timestamps spread across 24 hours for partitioned variants"""
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        timestamps = []

        for i in range(num_records):
            # Spread records across 24 hours
            hours_offset = (i * 24 * 60 * 60) // num_records  # seconds
            dt = base_time + timedelta(seconds=hours_offset)
            timestamps.append(dt)

        return timestamps

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        client = self._get_client()

        if self.variant == "mergetree" or self.variant.startswith("bf"):
            # Regular MergeTree variant
            data = []
            for i, (vector, payload) in enumerate(zip(vectors, payloads)):
                data.append((
                    i,
                    int(payload.get("row_id", i)),
                    vector,
                    str(payload.get("track", "unknown")) if payload.get("track") is not None else "unknown",
                    str(payload.get("artist", "unknown")) if payload.get("artist") is not None else "unknown",
                    str(payload.get("genre", "unknown")) if payload.get("genre") is not None else "unknown",
                    json.dumps(payload.get("seeds", [])),
                    str(payload.get("text", "")) if payload.get("text") is not None else ""
                ))

            client.execute(f"""
                INSERT INTO {self.table} (id, row_id, vector, track, artist, genre, seeds, text)
                VALUES
            """, data)
        else:
            # Partitioned variants with timestamps
            timestamps = self._generate_timestamps(len(vectors))

            data = []
            for i, (vector, payload, dt) in enumerate(zip(vectors, payloads, timestamps)):
                data.append((
                    dt,
                    i,
                    int(payload.get("row_id", i)),
                    vector,
                    str(payload.get("track", "unknown")) if payload.get("track") is not None else "unknown",
                    str(payload.get("artist", "unknown")) if payload.get("artist") is not None else "unknown",
                    str(payload.get("genre", "unknown")) if payload.get("genre") is not None else "unknown",
                    json.dumps(payload.get("seeds", [])),
                    str(payload.get("text", "")) if payload.get("text") is not None else ""
                ))

            client.execute(f"""
                INSERT INTO {self.table} (dt, id, row_id, vector, track, artist, genre, seeds, text)
                VALUES
            """, data)

        # Materialize vector index for the inserted data (only if present)
        if self._use_vector_index():
            client.execute(f"ALTER TABLE {self.table} MATERIALIZE INDEX vec_idx SETTINGS mutations_sync = 2")

    def search(self, query: List[float], top_k: int) -> List[Dict[str, Any]]:
        client = self._get_client()

        # Search SQL is the same for both indexed and brute-force; ORDER BY computes distances
        query_sql = f"""
        SELECT id, row_id, cosineDistance(vector, %(q)s) as distance,
               track, artist, genre, seeds, text
        FROM {self.table}
        ORDER BY cosineDistance(vector, %(q)s)
        LIMIT %(k)s
        """

        results = client.execute(query_sql, {"q": query, "k": top_k})

        output = []
        for row in results:
            id_val, row_id, distance, track, artist, genre, seeds, text = row
            output.append({
                "id": id_val,
                "score": 1.0 - float(distance),
                "payload": {
                    "row_id": row_id,
                    "track": track,
                    "artist": artist,
                    "genre": genre,
                    "seeds": json.loads(seeds) if seeds else [],
                    "text": text
                }
            })

        return output

    def teardown(self):
        client = self._get_client()
        # Do not drop previous run tables implicitly; leave them for inspection
        self.close()


class ClickHouseMergeTree(ClickHouse):
    """Indexed: MergeTree with HNSW index, vectors stored as Array(Float32)."""
    def __init__(self, host: str = "localhost", port: int = 9009, database: str = "default", table: str = "music_embeddings_mergetree"):
        super().__init__(host, port, database, table, "mergetree")


class ClickHouseFloat16(ClickHouse):
    """Indexed: partitioned table with vectors stored as Array(BFloat16)."""
    def __init__(self, host: str = "localhost", port: int = 9009, database: str = "default", table: str = "music_embeddings_bfloat16"):
        super().__init__(host, port, database, table, "float16")


class ClickHouseFloat32(ClickHouse):
    """Indexed: partitioned table with vectors stored as Array(Float32)."""
    def __init__(self, host: str = "localhost", port: int = 9009, database: str = "default", table: str = "music_embeddings_float32"):
        super().__init__(host, port, database, table, "float32")


class ClickHouseFloat64(ClickHouse):
    """Indexed: partitioned table with vectors stored as Array(Float64)."""
    def __init__(self, host: str = "localhost", port: int = 9009, database: str = "default", table: str = "music_embeddings_float64"):
        super().__init__(host, port, database, table, "float64")


class ClickHouseBF16(ClickHouse):
    """Brute-force: MergeTree without index, vectors stored as Array(BFloat16)."""
    def __init__(self, host: str = "localhost", port: int = 9009, database: str = "default", table: str = "music_embeddings_bf16"):
        super().__init__(host, port, database, table, "bf16")


class ClickHouseBF32(ClickHouse):
    """Brute-force: MergeTree without index, vectors stored as Array(Float32)."""
    def __init__(self, host: str = "localhost", port: int = 9009, database: str = "default", table: str = "music_embeddings_bf32"):
        super().__init__(host, port, database, table, "bf32")


class ClickHouseBF64(ClickHouse):
    """Brute-force: MergeTree without index, vectors stored as Array(Float64)."""
    def __init__(self, host: str = "localhost", port: int = 9009, database: str = "default", table: str = "music_embeddings_bf64"):
        super().__init__(host, port, database, table, "bf64")
