import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

TOTAL_EMBEDDINGS = 1_000_000
NUM_FILES  = 10
D = 1024
rows_per_file = TOTAL_EMBEDDINGS // NUM_FILES

DATA_DIRECTORY = "../synthetic-embeddings/"

import os
os.makedirs(DATA_DIRECTORY, exist_ok=True)

print(f"Generating {TOTAL_EMBEDDINGS} embeddings and writing to {DATA_DIRECTORY}...")

for i in range(NUM_FILES):
    embeddings = np.random.randn(rows_per_file, D).astype("float32")
    table = pa.table({
        "embedding": embeddings.tolist()
    })
    pq.write_table(table, f"{DATA_DIRECTORY}/embeddings_{i}_base.parquet")

    print(f"Wrote {i+1} files")
