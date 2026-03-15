//! Run a top-k search against an indexed Parquet file.
//!
//! Usage:
//!   cargo run --example topk_search
//!
//! Optional env vars:
//! - PQ_VECTOR_SOURCE: source parquet file (default: data/vldb_2025.parquet)
//! - PQ_VECTOR_INDEXED: indexed parquet file (default: data/vldb_2025_indexed.parquet)
//! - PQ_VECTOR_QUERY_ROW: which row to use as the query (default: 0)

mod common;

use common::{ensure_indexed, read_embedding_at_row};
use pq_vector::MultiTopkBuilder;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {


    let sources = vec!["data/vldb_2025.parquet".to_string(), "data/vldb_2025-copy.parquet".to_string()];
    let indexed = vec!["data/vldb_2025_indexed.parquet".to_string(), "data/vldb_2025-copy_indexed.parquet".to_string()];

    let query_row: usize = std::env::var("PQ_VECTOR_QUERY_ROW")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);

    for (source_file, index_file) in sources.iter().zip(indexed.iter()) {
        ensure_indexed(&source_file, &index_file)?;
    }


    let query = read_embedding_at_row(Path::new(&indexed[0]), "embedding", query_row)?;
    let results = MultiTopkBuilder::new(&indexed, &query)
        .k(5)?
        .nprobe(5)?
        .search()
        .await?;

    println!("Top 5 neighbors for row {query_row}:");
    for (rank, result) in results.iter().enumerate() {
        println!(
            "{}. row {} distance {:.4}",
            rank + 1,
            result.row_idx,
            result.distance
        );
    }

    Ok(())
}
