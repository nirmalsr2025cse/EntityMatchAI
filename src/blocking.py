import argparse
from collections import defaultdict
import gc
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DATA_DIR, K
from src.io_utils import load_source_records, write_candidate_pairs
from src.normalize import normalize_records

# Ensure Unicode characters print cleanly on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_mem_mb() -> Tuple[float, float]:
    """Return (current_rss_mb, peak_wset_mb)."""
    p = psutil.Process()
    info = p.memory_info()
    rss = info.rss / (1024 * 1024)
    peak = getattr(info, "peak_wset", info.rss) / (1024 * 1024)
    return rss, peak


def extract_sparse_topk(
    sim_chunk: csr_matrix,
    top_k: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract top-K non-zero columns and values row-by-row directly from a CSR matrix.
    
    Zero dense matrix allocation, zero .toarray() calls.
    Returns a list of (col_indices, sim_values) for each row in the chunk.
    """
    results: List[Tuple[np.ndarray, np.ndarray]] = []
    indptr = sim_chunk.indptr
    indices = sim_chunk.indices
    data = sim_chunk.data

    for r in range(sim_chunk.shape[0]):
        start = indptr[r]
        end = indptr[r + 1]
        n_nz = end - start
        if n_nz == 0:
            results.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)))
            continue

        cols = indices[start:end]
        vals = data[start:end]

        if n_nz > top_k:
            part = np.argpartition(-vals, top_k - 1)[:top_k]
            cols = cols[part]
            vals = vals[part]

        order = np.argsort(-vals)
        results.append((cols[order], vals[order]))

    return results


def block_country_source(
    s1_texts_a: List[str],
    s1_texts_b: List[str],
    s1_ids: np.ndarray,
    cand_texts_a: List[str],
    cand_texts_b: List[str],
    cand_ids: np.ndarray,
    top_k: int = 30,
    max_df: float = 0.3,
    chunk_size: int = 1000,
    progress_label: str = "",
) -> Dict[str, Dict[str, List[float]]]:
    """Compute sparse top-K candidates for both views (name and combo).
    
    Logs throughput (time per 10k queries) and peak RAM throughout.
    Returns: {s1_id: {cand_id: [sim_name, sim_combo]}}
    """
    n_s1 = len(s1_ids)
    n_cand = len(cand_ids)
    if n_s1 == 0 or n_cand == 0:
        return {}

    s1_cands: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    t_start = time.time()

    # --- VIEW A: name_core ---
    vec_a = TfidfVectorizer(
        max_df=max_df,
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b\w+\b",
    )
    vec_a.fit(s1_texts_a + cand_texts_a)
    Xq_a = vec_a.transform(s1_texts_a)
    Xc_a = vec_a.transform(cand_texts_a)
    XcT_a = Xc_a.T.tocsr()

    # --- VIEW B: combo (name_core + addr_norm) ---
    vec_b = TfidfVectorizer(
        max_df=max_df,
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b\w+\b",
    )
    vec_b.fit(s1_texts_b + cand_texts_b)
    Xq_b = vec_b.transform(s1_texts_b)
    Xc_b = vec_b.transform(cand_texts_b)
    XcT_b = Xc_b.T.tocsr()

    # Query loop in chunks
    last_log_q = 0
    t_chunk_start = time.time()

    for start in range(0, n_s1, chunk_size):
        end = min(start + chunk_size, n_s1)

        # View A chunk
        sim_a = (Xq_a[start:end] @ XcT_a).tocsr()
        topk_a = extract_sparse_topk(sim_a, top_k)
        for offset, (cols, vals) in enumerate(topk_a):
            s1 = s1_ids[start + offset]
            for c_idx, sim_v in zip(cols, vals):
                s1_cands[s1][cand_ids[c_idx]][0] = float(sim_v)

        # View B chunk
        sim_b = (Xq_b[start:end] @ XcT_b).tocsr()
        topk_b = extract_sparse_topk(sim_b, top_k)
        for offset, (cols, vals) in enumerate(topk_b):
            s1 = s1_ids[start + offset]
            for c_idx, sim_v in zip(cols, vals):
                s1_cands[s1][cand_ids[c_idx]][1] = float(sim_v)

        # Log every 10k queries
        if end - last_log_q >= 10000 or end == n_s1:
            q_done = end
            elapsed = time.time() - t_start
            time_per_10k = (elapsed / max(q_done, 1)) * 10000
            rss_mb, peak_mb = get_mem_mb()
            print(
                f"    [{progress_label}] Queries: {q_done:,}/{n_s1:,} "
                f"({q_done/n_s1:.1%}) | {time_per_10k:.2f}s/10k queries | "
                f"RAM: {rss_mb:.0f} MB (Peak: {peak_mb:.0f} MB)",
                flush=True,
            )
            last_log_q = end

    return s1_cands


def run_blocking(
    data_dir: Path,
    out_dir: Path,
    pool_dir: Optional[Path] = None,
    query_frac: float = 1.0,
    query_seed: int = 42,
    top_k: int = K,
    max_df: float = 0.3,
    chunk_size: int = 1000,
    split: str = "train",
) -> Tuple[pd.DataFrame, int]:
    """Execute memory-efficient blocking with support for query subsampling and full candidate pools.
    
    1. Loads and normalizes query S1 records (subsampling via query_frac if < 1.0).
    2. Saves query records in records.parquet.
    3. Streams candidate pools from pool_dir per country and per source.
    4. Computes sparse top-K for View A and View B, writing pairs.parquet incrementally in chunks.
    5. Writes candidate_pairs.tsv.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = Path(pool_dir) if pool_dir else Path(data_dir)
    t0_total = time.time()

    print("=" * 80)
    print("BLOCKING PIPELINE EXECUTION")
    print(f"  Query Source 1 dir : {data_dir}")
    print(f"  Candidate Pool dir : {pool_dir}")
    print(f"  Query fraction     : {query_frac:.1%} (seed={query_seed})")
    print(f"  Artifacts output   : {out_dir}")
    print(f"  Top-K per view     : {top_k}")
    print("=" * 80)

    # Step 1: Load S1 Query Records
    print(f"\n[Step 1] Loading S1 query records from: {data_dir}")
    t0 = time.time()
    s1_raw = load_source_records(data_dir, source="S1", split=split)
    n_s1_full = len(s1_raw)
    print(f"  Loaded {n_s1_full:,} raw S1 records in {time.time()-t0:.2f}s")

    # Apply query fraction if requested
    if query_frac < 1.0:
        rng = np.random.default_rng(query_seed)
        mask = rng.random(n_s1_full) < query_frac
        s1_raw = s1_raw[mask].reset_index(drop=True)
        print(f"  Subsampled {len(s1_raw):,} query S1 records ({len(s1_raw)/n_s1_full:.2%})")

    # Normalize S1 query records
    print("  Normalizing query S1 records...")
    t0 = time.time()
    s1_norm = normalize_records(s1_raw)
    del s1_raw
    gc.collect()
    print(f"  Normalized {len(s1_norm):,} query S1 records in {time.time()-t0:.2f}s")

    # Save normalized query records
    rec_path = out_dir / "records.parquet"
    s1_norm.to_parquet(rec_path, index=False)
    print(f"  Saved query records to: {rec_path}")

    # Prepare combo text for View B
    s1_norm["combo"] = (
        s1_norm["name_core"].fillna("") + " " + s1_norm["addr_norm"].fillna("")
    ).str.strip()

    all_query_s1_ids = s1_norm["id"].tolist()
    country_groups = sorted(s1_norm["country"].unique())
    print(f"\nQuery S1 records span {len(country_groups)} country groups: {country_groups}")

    # Step 2: Setup ParquetWriter for streaming pairs.parquet in chunks
    pairs_path = out_dir / "pairs.parquet"
    cand_tsv_path = out_dir / "candidate_pairs.tsv"

    pairs_schema = pa.schema([
        ("s1", pa.string()),
        ("cand", pa.string()),
        ("block_rank", pa.int64()),
        ("block_sim_name", pa.float64()),
        ("block_sim_combo", pa.float64()),
    ])
    parquet_writer = pq.ParquetWriter(pairs_path, schema=pairs_schema, compression="snappy")
    tsv_file = open(cand_tsv_path, "w", encoding="utf-8", newline="\n")
    tsv_file.write("source1_entity_id\tcandidate_entity_ids\n")

    total_candidate_pairs = 0

    # Step 3: Process per country group
    for country in country_groups:
        t_country = time.time()
        s1_c = s1_norm[s1_norm["country"] == country]
        n_queries = len(s1_c)
        print(f"\n=======================================================")
        print(f"[Country: {country.upper()}] Query S1 count: {n_queries:,}")
        print(f"=======================================================")

        s1_ids = s1_c["id"].values
        s1_texts_a = s1_c["name_core"].tolist()
        s1_texts_b = s1_c["combo"].tolist()

        # Track candidate dictionary for this country: s1 -> cand -> (sim_name, sim_combo)
        country_cands: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)

        for cand_src in ("S2", "S3"):
            t_src = time.time()
            print(f"\n  Loading {cand_src} candidate pool from {pool_dir} for country '{country}'...")
            try:
                cand_raw = load_source_records(pool_dir, source=cand_src, split=split, country=country)
            except Exception as e:
                print(f"  Could not load {cand_src} for country {country}: {e}")
                continue

            n_cands = len(cand_raw)
            rss_mb, peak_mb = get_mem_mb()
            print(f"  Loaded {n_cands:,} {cand_src} raw candidate records in {time.time()-t_src:.2f}s | RAM: {rss_mb:.0f} MB")

            if n_cands == 0:
                del cand_raw
                gc.collect()
                continue

            # Normalize candidate records
            t_norm = time.time()
            cand_norm = normalize_records(cand_raw)
            del cand_raw
            gc.collect()
            print(f"  Normalized {n_cands:,} {cand_src} records in {time.time()-t_norm:.2f}s")

            cand_norm["combo"] = (
                cand_norm["name_core"].fillna("") + " " + cand_norm["addr_norm"].fillna("")
            ).str.strip()

            cand_ids = cand_norm["id"].values
            cand_texts_a = cand_norm["name_core"].tolist()
            cand_texts_b = cand_norm["combo"].tolist()
            del cand_norm
            gc.collect()

            # Execute blocking
            print(f"  Blocking {n_queries:,} S1 queries against {n_cands:,} {cand_src} pool...")
            src_results = block_country_source(
                s1_texts_a=s1_texts_a,
                s1_texts_b=s1_texts_b,
                s1_ids=s1_ids,
                cand_texts_a=cand_texts_a,
                cand_texts_b=cand_texts_b,
                cand_ids=cand_ids,
                top_k=top_k,
                max_df=max_df,
                chunk_size=chunk_size,
                progress_label=f"{country}-{cand_src}",
            )

            # Merge results into country_cands
            pairs_found = 0
            for s1, c_dict in src_results.items():
                for c_id, scores in c_dict.items():
                    country_cands[s1][c_id] = (scores[0], scores[1])
                    pairs_found += 1

            print(f"  Retrieved {pairs_found:,} candidate pairs from {cand_src} in {time.time()-t_src:.2f}s")
            del cand_ids, cand_texts_a, cand_texts_b, src_results
            gc.collect()

        # Assemble and write this country's candidate pairs in chunks
        print(f"\n  Writing candidate pairs for country '{country}'...")
        t_write = time.time()
        c_s1_list: List[str] = []
        c_cand_list: List[str] = []
        c_rank_list: List[int] = []
        c_sim_name_list: List[float] = []
        c_sim_combo_list: List[float] = []

        for s1 in s1_ids:
            cands_dict = country_cands.get(s1, {})
            if not cands_dict:
                tsv_file.write(f"{s1}\t\n")
                continue

            # Sort candidates by max similarity (combo, then name)
            sorted_cands = sorted(
                cands_dict.items(),
                key=lambda item: (max(item[1][1], item[1][0]), item[1][1], item[1][0]),
                reverse=True,
            )

            ranked_cand_ids = []
            for rank, (cand, (sim_name, sim_combo)) in enumerate(sorted_cands, start=1):
                c_s1_list.append(s1)
                c_cand_list.append(cand)
                c_rank_list.append(rank)
                c_sim_name_list.append(round(sim_name, 4))
                c_sim_combo_list.append(round(sim_combo, 4))
                ranked_cand_ids.append(cand)

            tsv_file.write(f"{s1}\t{','.join(ranked_cand_ids)}\n")

        # Stream chunk to pairs.parquet
        if c_s1_list:
            chunk_df = pd.DataFrame({
                "s1": c_s1_list,
                "cand": c_cand_list,
                "block_rank": c_rank_list,
                "block_sim_name": c_sim_name_list,
                "block_sim_combo": c_sim_combo_list,
            })
            table_chunk = pa.Table.from_pandas(chunk_df, schema=pairs_schema)
            parquet_writer.write_table(table_chunk)
            total_candidate_pairs += len(c_s1_list)
            print(f"  Wrote {len(c_s1_list):,} pairs to parquet chunk in {time.time()-t_write:.2f}s")
            del chunk_df, table_chunk, c_s1_list, c_cand_list, c_rank_list, c_sim_name_list, c_sim_combo_list

        del country_cands
        gc.collect()
        rss_mb, peak_mb = get_mem_mb()
        print(f"  Country '{country}' finished in {time.time()-t_country:.2f}s | Peak RAM: {peak_mb:.0f} MB")

    # Close writers
    parquet_writer.close()
    tsv_file.close()

    total_time = time.time() - t0_total
    avg_cands = total_candidate_pairs / max(len(all_query_s1_ids), 1)
    rss_mb, peak_mb = get_mem_mb()

    print("\n" + "=" * 80)
    print("BLOCKING SUMMARY")
    print("=" * 80)
    print(f"  Total S1 queries processed : {len(all_query_s1_ids):,}")
    print(f"  Total candidate pairs      : {total_candidate_pairs:,}")
    print(f"  Average candidates per S1  : {avg_cands:.1f}")
    print(f"  Total runtime              : {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  Overall Time per 10k S1    : {(total_time/max(len(all_query_s1_ids),1))*10000:.2f}s")
    print(f"  Peak RAM used              : {peak_mb:.0f} MB")
    print(f"  Saved pairs to             : {pairs_path}")
    print(f"  Saved candidate TSV to     : {cand_tsv_path}")
    print("=" * 80)

    return s1_norm, total_candidate_pairs


def main():
    parser = argparse.ArgumentParser(description="Blocking Pipeline for Business Entity Resolution")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to query dataset directory")
    parser.add_argument("--pool-dir", default=None, help="Path to full candidate pool directory (e.g. dataset/train)")
    parser.add_argument("--query-frac", type=float, default=1.0, help="Fraction of S1 records to use as queries")
    parser.add_argument("--query-seed", type=int, default=42, help="Random seed for query subsampling")
    parser.add_argument("--out", default="artifacts/sample_1pct", help="Path to output artifacts directory")
    parser.add_argument("--k", type=int, default=K, help="Top-K neighbours per S1 per view")
    parser.add_argument("--max-df", type=float, default=0.3, help="Max document frequency for TF-IDF")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for sparse queries")
    args = parser.parse_args()

    run_blocking(
        data_dir=Path(args.data_dir),
        pool_dir=Path(args.pool_dir) if args.pool_dir else None,
        query_frac=args.query_frac,
        query_seed=args.query_seed,
        out_dir=Path(args.out),
        top_k=args.k,
        max_df=args.max_df,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
