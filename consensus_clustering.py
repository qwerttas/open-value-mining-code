#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score

DIM_FIELDS = {
    "goal": "goal",
    "final_goal": "final_goal",
    "principle": "principle",
    "tradeoff": "tradeoff",
    "trade_off": "trade_off",
    "beneficiaries": "beneficiaries",
    "harmed_groups": "harmed_groups",
    "norm": "norm",
    "perspective": "perspective",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def parse_list(value: str, cast=str) -> list[Any]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(cast(item))
    return out


def dimension_text(row: dict[str, Any], dim: str) -> str:
    attrs = row.get("attributes") or {}
    field = DIM_FIELDS.get(dim, dim)
    value = str(attrs.get(field) or "").strip()
    if not value and dim == "goal":
        value = str(attrs.get("final_goal") or "").strip()
    if not value and dim == "tradeoff":
        value = str(attrs.get("trade_off") or "").strip()
    return value


def build_embeddings(rows: list[dict[str, Any]], dims: list[str], model_name: str, out_dir: Path, batch_size: int) -> dict[str, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    out_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)
    vecs = {}
    for dim in dims:
        path = out_dir / f"{dim}.npy"
        if path.exists():
            vecs[dim] = np.load(path)
            continue
        texts = [dimension_text(row, dim) for row in rows]
        arr = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        arr = np.asarray(arr, dtype=np.float32)
        np.save(path, arr)
        vecs[dim] = arr
    return vecs


def load_embeddings(embedding_dir: Path, dims: list[str]) -> dict[str, np.ndarray]:
    return {dim: np.load(embedding_dir / f"{dim}.npy") for dim in dims}


def make_view_embedding(dim_vecs: dict[str, np.ndarray], dims: list[str]) -> np.ndarray:
    mats = [dim_vecs[dim] for dim in dims]
    return l2_normalize(np.concatenate(mats, axis=1).astype(np.float32))


def knn_candidates(vecs: np.ndarray, k: int, batch_size: int) -> list[tuple[int, int]]:
    n = vecs.shape[0]
    k = min(k, max(n - 1, 1))
    edges = set()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = vecs[start:end] @ vecs.T
        for r, i in enumerate(range(start, end)):
            sims[r, i] = -np.inf
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        for r, i in enumerate(range(start, end)):
            for j in idx[r]:
                a, b = (i, int(j)) if i < int(j) else (int(j), i)
                if a != b:
                    edges.add((a, b))
    return sorted(edges)


def kmeans_partition(vecs: np.ndarray, k: int, seed: int, subset: Optional[np.ndarray]) -> np.ndarray:
    labels = np.full(vecs.shape[0], -1, dtype=np.int32)
    data = vecs if subset is None else vecs[subset]
    if data.shape[0] < k:
        return labels
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    part = model.fit_predict(data)
    if subset is None:
        labels[:] = part
    else:
        labels[subset] = part
    return labels


def build_partitions(vecs: np.ndarray, k_values: list[int], seeds: list[int], bootstrap: int, bootstrap_frac: float) -> list[np.ndarray]:
    rng = np.random.default_rng(12345)
    n = vecs.shape[0]
    partitions = []
    for k in k_values:
        for seed in seeds:
            partitions.append(kmeans_partition(vecs, k, seed, None))
            for _ in range(bootstrap):
                size = max(k + 1, int(n * bootstrap_frac))
                size = min(size, n)
                subset = np.sort(rng.choice(n, size=size, replace=False))
                partitions.append(kmeans_partition(vecs, k, seed + int(rng.integers(1, 1_000_000)), subset))
    return partitions


def consensus_matrix(partitions: list[np.ndarray], edges: list[tuple[int, int]], n: int, min_observed: int) -> sparse.csr_matrix:
    votes = np.zeros(len(edges), dtype=np.float32)
    denom = np.zeros(len(edges), dtype=np.float32)
    a = np.array([e[0] for e in edges], dtype=np.int32)
    b = np.array([e[1] for e in edges], dtype=np.int32)
    for labels in partitions:
        observed = (labels[a] >= 0) & (labels[b] >= 0)
        denom[observed] += 1.0
        votes[observed & (labels[a] == labels[b])] += 1.0
    keep = denom >= float(min_observed)
    weights = np.zeros_like(votes)
    weights[keep] = votes[keep] / np.maximum(denom[keep], 1.0)
    diag = np.arange(n, dtype=np.int32)
    rows = np.concatenate([a[keep], b[keep], diag])
    cols = np.concatenate([b[keep], a[keep], diag])
    vals = np.concatenate([weights[keep], weights[keep], np.ones(n, dtype=np.float32)])
    mat = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return mat


def confidence_scores(affinity: sparse.csr_matrix, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = labels.shape[0]
    conf_intra = np.zeros(n, dtype=np.float32)
    conf_ratio = np.zeros(n, dtype=np.float32)
    for i in range(n):
        row = affinity.getrow(i)
        js = row.indices
        ws = row.data
        mask_self = js != i
        js = js[mask_self]
        ws = ws[mask_self]
        if len(js) == 0:
            continue
        own = ws[labels[js] == labels[i]]
        other = ws[labels[js] != labels[i]]
        own_mean = float(own.mean()) if len(own) else 0.0
        other_mean = float(other.mean()) if len(other) else 0.0
        conf_intra[i] = own_mean
        conf_ratio[i] = own_mean / (own_mean + other_mean + 1e-8)
    return conf_intra, conf_ratio


def cluster_summary(labels: np.ndarray, vecs: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(labels)
    summary = {
        "n_samples": int(labels.shape[0]),
        "n_clusters": int(len(counts)),
        "max_cluster_ratio": float(counts.max() / labels.shape[0]),
        "cluster_sizes": {str(i): int(v) for i, v in enumerate(counts)},
    }
    if len(counts) > 1 and min(counts) > 1:
        summary["silhouette"] = float(silhouette_score(vecs, labels, metric="cosine"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse consensus clustering over value-semantic vectors.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dims", default="goal,principle,tradeoff,perspective")
    parser.add_argument("--k-final", type=int, required=True)
    parser.add_argument("--k-base", default=None, help="Comma-separated K values for base partitions.")
    parser.add_argument("--seeds", default="11,23,37,51,67")
    parser.add_argument("--bootstrap", type=int, default=2)
    parser.add_argument("--bootstrap-frac", type=float, default=0.8)
    parser.add_argument("--knn", type=int, default=50)
    parser.add_argument("--min-observed", type=int, default=2)
    parser.add_argument("--embedding-dir", default=None)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_jsonl))
    dims = parse_list(args.dims)
    out_dir = Path(args.output_dir)
    emb_dir = Path(args.embedding_dir) if args.embedding_dir else out_dir / "embeddings"
    if emb_dir.exists() and all((emb_dir / f"{dim}.npy").exists() for dim in dims):
        dim_vecs = load_embeddings(emb_dir, dims)
    else:
        dim_vecs = build_embeddings(rows, dims, args.embedding_model, emb_dir, args.batch_size)

    vecs = make_view_embedding(dim_vecs, dims)
    k_values = parse_list(args.k_base, int) if args.k_base else sorted(set([args.k_final - 2, args.k_final, args.k_final + 2]))
    k_values = [k for k in k_values if k > 1]
    seeds = parse_list(args.seeds, int)

    edges = knn_candidates(vecs, args.knn, args.batch_size)
    partitions = build_partitions(vecs, k_values, seeds, args.bootstrap, args.bootstrap_frac)
    affinity = consensus_matrix(partitions, edges, vecs.shape[0], args.min_observed)
    labels = SpectralClustering(
        n_clusters=args.k_final,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=0,
    ).fit_predict(affinity)
    conf_intra, conf_ratio = confidence_scores(affinity, labels)

    out_rows = []
    for i, row in enumerate(rows):
        out = {
            "pack_id": row.get("pack_id"),
            "edge_id": row.get("edge_id"),
            "cluster_id": int(labels[i]),
            "conf_intra": float(conf_intra[i]),
            "conf_ratio": float(conf_ratio[i]),
        }
        out_rows.append(out)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "final_clusters.jsonl", out_rows)
    np.save(out_dir / "view_embedding.npy", vecs)
    sparse.save_npz(out_dir / "consensus_affinity.npz", affinity)
    with (out_dir / "cluster_summary.json").open("w", encoding="utf-8") as f:
        json.dump(cluster_summary(labels, vecs), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
