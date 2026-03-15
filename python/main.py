import pyarrow as pa
import pyarrow.parquet as pq

def main():
    # file_name = "../data/vldb_2025.parquet"

    file_name = "../synthetic-embeddings/embeddings_0_base.parquet"

    table = pq.read_table(
        file_name,
    )

    print(table.schema)
    print(table.num_rows)

    # file_name = "../synthetic-embeddings/"




if __name__ == "__main__":
    main()
