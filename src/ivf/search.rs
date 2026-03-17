use crate::ivf::index::squared_l2_distance;
use crate::ivf::{EmbeddingColumn, EmbeddingDim, IvfIndex, read_index_from_parquet};
use arrow::array::Array;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{ArrowReaderOptions, RowSelection, RowSelector};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use futures::future::join_all;

// For max-heap (we want to pop largest distances).
#[derive(Debug, Clone)]
struct HeapItem {
    row_idx: u32,
    distance: f32,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}


struct HeapItemWithSource {
    row_idx: u32,
    distance: f32,
    source_file: PathBuf,
}

impl PartialEq for HeapItemWithSource {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for HeapItemWithSource {}

impl PartialOrd for HeapItemWithSource {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItemWithSource {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}


/// Result item from top-k search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub row_idx: u32,
    pub distance: f32,
}

/// Builder for top-k nearest neighbor search.
#[derive(Debug, Clone)]
pub struct TopkBuilder<'a> {
    parquet_path: PathBuf,
    query: &'a [f32],
    k: Option<NonZeroUsize>,
    nprobe: Option<NonZeroUsize>,
}


impl<'a> TopkBuilder<'a> {
    pub fn new(parquet_path: impl AsRef<Path>, query: &'a [f32]) -> Self {
        Self {
            parquet_path: parquet_path.as_ref().to_path_buf(),
            query,
            k: None,
            nprobe: None,
        }
    }

    pub fn k(mut self, k: usize) -> Result<Self, Box<dyn std::error::Error>> {
        self.k = Some(NonZeroUsize::new(k).ok_or("k must be > 0")?);
        Ok(self)
    }

    pub fn nprobe(mut self, nprobe: usize) -> Result<Self, Box<dyn std::error::Error>> {
        self.nprobe = Some(NonZeroUsize::new(nprobe).ok_or("nprobe must be > 0")?);
        Ok(self)
    }

    pub async fn search(self) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let k = self.k.ok_or("k must be set")?;
        let nprobe = self.nprobe.ok_or("nprobe must be set")?;
        topk(self.parquet_path.as_path(), self.query, k, nprobe).await
    }
}


#[derive(Debug, Clone)]
pub struct MultiTopkBuilder<'a> {
    parquet_paths: Vec<PathBuf>,
    query: &'a [f32],
    k: Option<NonZeroUsize>,
    nprobe: Option<NonZeroUsize>,
}

impl<'a> MultiTopkBuilder<'a> {
    pub fn new<I, P>(parquet_paths: I, query: &'a [f32]) -> Self
    where
        I: IntoIterator<Item=P>,
        P: AsRef<Path>
    {
        let paths = parquet_paths
            .into_iter()
            .map(|p| p.as_ref().to_path_buf())
            .collect();

        Self {
            parquet_paths: paths,
            query,
            k: None,
            nprobe: None,
        }
    }

    pub fn k(mut self, k: usize) -> Result<Self, Box<dyn std::error::Error>> {
        self.k = Some(NonZeroUsize::new(k).ok_or("k must be > 0")?);
        Ok(self)
    }

    pub fn nprobe(mut self, nprobe: usize) -> Result<Self, Box<dyn std::error::Error>> {
        self.nprobe = Some(NonZeroUsize::new(nprobe).ok_or("nprobe must be > 0")?);
        Ok(self)
    }

    pub async fn search(self) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let k = self.k.ok_or("k must be set")?;
        let nprobe = self.nprobe.ok_or("nprobe must be set")?;
        let futures = self.parquet_paths.iter().map(|path| {
            topk(path, self.query, k, nprobe)
        });

        // let results: Vec<Result<Vec<SearchResult>, Box<dyn std::error::Error>>> = join_all(futures).await;

        let mut all_results: Vec<SearchResult> =
            join_all(futures)
                .await
                .into_iter()
                .filter_map(Result::ok)
                .flatten()
                .collect();

        all_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        Ok(all_results)
    }

    pub async fn nonuniform_search(self) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let k = self.k.ok_or("k must be set")?;
        let nprobe = self.nprobe.ok_or("nprobe must be set")?;
        let clusters_to_search = nprobe.get() * self.parquet_paths.len();
        let dim = self.query.len();


        let mut heap: BinaryHeap<HeapItemWithSource> = BinaryHeap::with_capacity(clusters_to_search);

        //     query all centroids from all files
        for parquet_path in self.parquet_paths {
            let parquet_path = parquet_path.as_path();
            let (index, embedding_column) = read_index_from_parquet(parquet_path)?;
            
            let centroid_distances = index.get_centroid_distances(self.query);

            centroid_distances.iter().enumerate().for_each(|(centroid_idx, distance)| {
                heap.push(HeapItemWithSource {
                    row_idx: centroid_idx as u32,
                    distance: *distance,
                    source_file: parquet_path.to_path_buf(),
                });
            });

        }

        
        // group by source file and read embeddings for all selected centroids in that file
        let files_to_search: Vec<(PathBuf, Vec<u32>)> = heap.into_sorted_vec()
            .into_iter()
            .take(clusters_to_search)
            .fold(std::collections::HashMap::<PathBuf, Vec<u32>>::new(), |mut acc, item| {
                acc.entry(item.source_file)
                    .or_default()
                    .push(item.row_idx);
                acc
            })
            .into_iter()
            .collect();


            
        // let rows_to_check_futures = files_to_search.iter().map(|(parquet_path, cluster_idxs)| async move {
        //         let (index, _) = read_index_from_parquet(parquet_path).ok()?;

        //         Some(
        //             cluster_idxs
        //                 .iter()
        //                 .flat_map(|&cluster_idx| index.get_rows_for_cluster(cluster_idx))
        //                 .collect::<Vec<u32>>(),
        //         )
        //     });

        let rows_to_check_futures = files_to_search.iter().map(|(parquet_path, cluster_idxs)| async move {
            read_index_from_parquet(parquet_path)
                .ok()
                .map(|(index, _)| {
                    cluster_idxs
                        .iter()
                        .flat_map(|&cluster_idx| index.get_rows_for_cluster(cluster_idx))
                        .collect::<Vec<u32>>()
                })
        });


        let rows_to_check: Vec<u32> = join_all(rows_to_check_futures)
            .await
            .into_iter()
            .flatten()      // remove None
            .flatten()      // flatten Vec<Vec<u32>>
            .collect();

        let files_to_search_ref = &files_to_search;

        let futures = rows_to_check
            .chunks(1000)
            .map(|chunk| {
                let files_to_search = files_to_search_ref;

                async move {
                    let parquet_path = files_to_search
                        .iter()
                        .find(|(_, cluster_idxs)| cluster_idxs.contains(&chunk[0]))
                        .map(|(path, _)| path)
                        .unwrap();

                    let (index, embedding_column) = read_index_from_parquet(parquet_path).ok()?;

                    read_embeddings_for_rows(
                        EmbeddingReadContext {
                            path: parquet_path,
                            embedding_column: &embedding_column,
                            dim: index_dim(&index),
                        },
                        chunk,
                    )
                    .await
                    .ok()
                }
            });

        let embeddings: Vec<f32> = join_all(futures).await.into_iter().filter(|x| x.is_some()).flatten().flatten().collect();
        

        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k.get() + 1);

        for (i, &row_idx) in rows_to_check.iter().enumerate() {
            let vec = &embeddings[i * dim..(i + 1) * dim];

            let distance = squared_l2_distance(self.query, vec);

            if heap.len() < k.get() {
                heap.push(HeapItem { row_idx, distance });
            } else if let Some(top) = heap.peek()
                && distance < top.distance
            {
                heap.pop();
                heap.push(HeapItem { row_idx, distance });
            }
        }

        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|item| SearchResult {
                row_idx: item.row_idx,
                distance: item.distance.sqrt(),
            })
            .collect();
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });



        Ok(results)

    }



}

async fn topk(
    parquet_path: &Path,
    query: &[f32],
    k: NonZeroUsize,
    nprobe: NonZeroUsize,
) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
    let (index, embedding_column) = read_index_from_parquet(parquet_path)?;

    if query.len() != index.dim() {
        return Err(format!(
            "Query dimension mismatch: expected {}, got {}",
            index.dim(),
            query.len()
        )
        .into());
    }

    let rows_to_check: Vec<u32> = index.candidate_rows(query, nprobe.get());

    let embeddings = read_embeddings_for_rows(
        EmbeddingReadContext {
            path: parquet_path,
            embedding_column: &embedding_column,
            dim: index_dim(&index),
        },
        &rows_to_check,
    )
    .await?;

    let k = k.get();
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k + 1);

    for (i, &row_idx) in rows_to_check.iter().enumerate() {
        let vec = &embeddings[i * index.dim()..(i + 1) * index.dim()];
        let distance = squared_l2_distance(query, vec);

        if heap.len() < k {
            heap.push(HeapItem { row_idx, distance });
        } else if let Some(top) = heap.peek()
            && distance < top.distance
        {
            heap.pop();
            heap.push(HeapItem { row_idx, distance });
        }
    }

    let mut results: Vec<SearchResult> = heap
        .into_iter()
        .map(|item| SearchResult {
            row_idx: item.row_idx,
            distance: item.distance.sqrt(),
        })
        .collect();
    results.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(Ordering::Equal)
    });
    Ok(results)
}

struct EmbeddingReadContext<'a> {
    path: &'a std::path::Path,
    embedding_column: &'a EmbeddingColumn,
    dim: EmbeddingDim,
}

fn index_dim(index: &IvfIndex) -> EmbeddingDim {
    EmbeddingDim::new(index.dim()).expect("IVF index dimension must be > 0")
}

/// Read embeddings for specific rows using direct page reads.
async fn read_embeddings_for_rows(
    context: EmbeddingReadContext<'_>,
    rows: &[u32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }

    let mut sorted_rows: Vec<u32> = rows.to_vec();
    sorted_rows.sort_unstable();

    let file = tokio::fs::File::open(context.path).await?;

    let options = ArrowReaderOptions::new().with_page_index(true);
    let builder = parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder::new_with_options(
        file, options,
    )
    .await?;

    let schema = builder.schema();
    let embedding_col_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == context.embedding_column.as_str())
        .ok_or_else(|| format!("Column '{}' not found", context.embedding_column.as_str()))?;
    let projection = ProjectionMask::roots(builder.parquet_schema(), [embedding_col_idx]);

    let total_rows = builder.metadata().file_metadata().num_rows() as usize;
    let mut selectors = Vec::new();
    let mut current_pos = 0;

    for &row in &sorted_rows {
        let row = row as usize;
        if row > current_pos {
            selectors.push(RowSelector::skip(row - current_pos));
        }
        selectors.push(RowSelector::select(1));
        current_pos = row + 1;
    }
    if current_pos < total_rows {
        selectors.push(RowSelector::skip(total_rows - current_pos));
    }

    let selection = RowSelection::from(selectors);

    let mut stream = builder
        .with_projection(projection)
        .with_row_selection(selection)
        .build()?;

    use arrow::array::{Float32Array, ListArray};
    use futures::StreamExt;
    let mut sorted_embeddings = Vec::with_capacity(sorted_rows.len() * context.dim.as_usize());
    while let Some(batch) = stream.next().await {
        let batch: arrow::record_batch::RecordBatch = batch?;
        let embedding_col = batch.column(0);
        let list_array = embedding_col.as_any().downcast_ref::<ListArray>().unwrap();
        if list_array.null_count() > 0 {
            return Err("Embedding column contains null rows".into());
        }
        let values = list_array.values();
        let float_array = values.as_any().downcast_ref::<Float32Array>().unwrap();
        if float_array.null_count() > 0 {
            return Err("Embedding values contain nulls".into());
        }

        for i in 0..float_array.len() {
            sorted_embeddings.push(float_array.value(i));
        }
    }

    if sorted_embeddings.len() != sorted_rows.len() * context.dim.as_usize() {
        return Err("Selected embeddings do not match expected dimensions".into());
    }

    let row_to_sorted_idx: std::collections::HashMap<u32, usize> = sorted_rows
        .iter()
        .enumerate()
        .map(|(i, &r)| (r, i))
        .collect();

    let mut result = Vec::with_capacity(rows.len() * context.dim.as_usize());
    for &row in rows {
        let sorted_idx = row_to_sorted_idx[&row];
        let start = sorted_idx * context.dim.as_usize();
        result.extend_from_slice(&sorted_embeddings[start..start + context.dim.as_usize()]);
    }

    Ok(result)
}
