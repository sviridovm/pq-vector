//! Build an IVF index and write it to a new Parquet file.
//!
//! Usage:
//!   cargo run --example build_index
//!
//! Optional env vars:
//! - PQ_VECTOR_SOURCE: source parquet file (default: data/vldb_2025.parquet)
//! - PQ_VECTOR_INDEXED: output parquet file (default: data/vldb_2025_indexed.parquet)

// mod common;
use pq_vector::MultiIndexBuilder;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source =
        std::env::var("PQ_VECTOR_SOURCE").unwrap_or_else(|_| "data/vldb_2025.parquet".to_string());
    let indexed = std::env::var("PQ_VECTOR_INDEXED")
        .unwrap_or_else(|_| "data/vldb_2025_indexed.parquet".to_string());

    let secondary_source = "data/vldb_2025-copy.parquet".to_string();

    let sources = std::vec![source, secondary_source];

    println!("Building IVF index from {:?}...", sources);
    MultiIndexBuilder::new(&sources, "embedding").build_all_inplace();
    println!("Wrote indexed parquet to {indexed}");

    Ok(())
}
