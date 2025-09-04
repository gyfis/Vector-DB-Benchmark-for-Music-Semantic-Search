from typing import List, Dict, Any
import psycopg2
from pgvector.psycopg2 import register_vector
from .base import VectorDB
import json


class Postgres(VectorDB):
    def __init__(
        self,
        host: str = "localhost",
        port: str = "5432",
        database: str = "music_vectors",
        user: str = "postgres",
        password: str = "password",
        table: str = "music_embeddings"
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.table = table
        self.conn = None

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def _get_connection(self):
        if self.conn is None:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            register_vector(self.conn)
        return self.conn

    def setup(self, dim: int):
        conn = self._get_connection()
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Drop table if exists
            cur.execute(f"DROP TABLE IF EXISTS {self.table};")

            # Create table with HNSW index
            cur.execute(f"""
                CREATE TABLE {self.table} (
                    id SERIAL PRIMARY KEY,
                    row_id INTEGER,
                    vector vector({dim}),
                    track TEXT,
                    artist TEXT,
                    genre TEXT,
                    seeds TEXT,
                    text TEXT
                );
            """)

            # Create HNSW index for vector similarity search
            cur.execute(f"""
                CREATE INDEX ON {self.table}
                USING hnsw (vector vector_cosine_ops)
                WITH (m = 16, ef_construction = 128);
            """)

        conn.commit()

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        conn = self._get_connection()
        with conn.cursor() as cur:
            # Prepare data for batch insert
            data = []
            for i, (vector, payload) in enumerate(zip(vectors, payloads)):
                data.append((
                    int(payload.get("row_id", i)),
                    vector,
                    payload.get("track", "unknown"),
                    payload.get("artist", "unknown"),
                    payload.get("genre", "unknown"),
                    json.dumps(payload.get("seeds", [])),
                    payload.get("text", "")
                ))

            # Batch insert
            cur.executemany(f"""
                INSERT INTO {self.table} (row_id, vector, track, artist, genre, seeds, text)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, data)

        conn.commit()

    def search(self, query: List[float], top_k: int) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        with conn.cursor() as cur:
            # Perform vector similarity search
            cur.execute(f"""
                SELECT id, row_id, vector <=> %s::vector as distance,
                       track, artist, genre, seeds, text
                FROM {self.table}
                ORDER BY vector <=> %s::vector
                LIMIT %s
            """, (query, query, top_k))

            results = []
            for row in cur.fetchall():
                id_val, row_id, distance, track, artist, genre, seeds, text = row
                results.append({
                    "id": id_val,
                    "score": 1.0 - float(distance),  # Convert distance to similarity score
                    "payload": {
                        "row_id": row_id,
                        "track": track,
                        "artist": artist,
                        "genre": genre,
                        "seeds": json.loads(seeds) if seeds else [],
                        "text": text
                    }
                })

        return results

    def teardown(self):
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table};")
        conn.commit()
        self.close()
