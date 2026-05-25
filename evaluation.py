#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score, silhouette_score


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def normalize_stance(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"support", "supports", "in favor", "in favour", "in favor of", "favor"}:
        return "in favor of"
    if text in {"attack", "attacks", "against", "oppose", "opposes"}:
        return "against"
    return text


def extract_edges(rows: list[dict[str, Any]], edge_key: str) -> list[dict[str, Any]]:
    edges = []
    for row in rows:
        pack_id = str(row.get("pack_id") or row.get("id") or "")
        units = {
            str(u.get("unit_id")): str(u.get("text") or "")
            for u in row.get("units") or row.get("units_pred") or []
            if isinstance(u, dict)
        }
        pair_list = row.get(edge_key) or row.get("pairs") or row.get("pairs_pred") or row.get("gold_edges")
        if isinstance(pair_list, list):
            for idx, edge in enumerate(pair_list, 1):
                if not isinstance(edge, dict):
                    continue
                src = str(edge.get("src_unit_id") or edge.get("premise_unit_id") or "").strip()
                tgt = str(edge.get("tgt_unit_id") or edge.get("conclusion_unit_id") or "").strip()
                premise = str(edge.get("premise_text") or units.get(src) or "").strip()
                conclusion = str(edge.get("conclusion_text") or units.get(tgt) or "").strip()
                if premise and conclusion:
                    edges.append({
                        "pack_id": pack_id,
                        "edge_id": str(edge.get("edge_id") or f"e{idx}"),
                        "premise_text": premise,
                        "conclusion_text": conclusion,
                        "stance": normalize_stance(edge.get("stance") or edge.get("relation")),
                    })
        elif row.get("premise_text") and row.get("conclusion_text"):
            edges.append({
                "pack_id": pack_id,
                "edge_id": str(row.get("edge_id") or len(edges)),
                "premise_text": str(row.get("premise_text") or ""),
                "conclusion_text": str(row.get("conclusion_text") or ""),
                "stance": normalize_stance(row.get("stance") or row.get("relation")),
            })
    return edges


def load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def encode_texts(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    return np.asarray(
        model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True),
        dtype=np.float32,
    )


def edge_scores(
    pred: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    model: Any,
    batch_size: int,
    ignore_stance: bool,
) -> np.ndarray:
    texts = (
        [e["premise_text"] for e in pred]
        + [e["conclusion_text"] for e in pred]
        + [e["premise_text"] for e in gold]
        + [e["conclusion_text"] for e in gold]
    )
    emb = encode_texts(model, texts, batch_size)
    n_pred = len(pred)
    n_gold = len(gold)
    p_pre = emb[:n_pred]
    p_con = emb[n_pred:2 * n_pred]
    g_pre = emb[2 * n_pred:2 * n_pred + n_gold]
    g_con = emb[2 * n_pred + n_gold:]
    sim = 0.5 * (p_pre @ g_pre.T + p_con @ g_con.T)
    if ignore_stance:
        return sim
    stance_ok = np.ones_like(sim, dtype=bool)
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            stance_ok[i, j] = p.get("stance") == g.get("stance")
    return np.where(stance_ok, sim, 0.0)


def evaluate_edges(args: argparse.Namespace) -> None:
    pred = extract_edges(read_jsonl(Path(args.pred_jsonl)), args.pred_key)
    gold = extract_edges(read_jsonl(Path(args.gold_jsonl)), args.gold_key)
    model = load_encoder(args.embedding_model)
    pred_by_pack = defaultdict(list)
    gold_by_pack = defaultdict(list)
    for edge in pred:
        pred_by_pack[edge["pack_id"]].append(edge)
    for edge in gold:
        gold_by_pack[edge["pack_id"]].append(edge)

    tp = 0
    soft_tp = 0.0
    matched_pairs = 0
    for pack_id in sorted(set(pred_by_pack) | set(gold_by_pack)):
        p_edges = pred_by_pack.get(pack_id, [])
        g_edges = gold_by_pack.get(pack_id, [])
        if not p_edges or not g_edges:
            continue
        sim = edge_scores(p_edges, g_edges, model, args.batch_size, args.ignore_stance)
        row_ind, col_ind = linear_sum_assignment(-sim)
        for i, j in zip(row_ind, col_ind):
            score = float(sim[i, j])
            if score > 0:
                soft_tp += score
                matched_pairs += 1
            if score >= args.threshold:
                tp += 1

    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    soft_precision = soft_tp / len(pred) if pred else 0.0
    soft_recall = soft_tp / len(gold) if gold else 0.0
    soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall) if soft_precision + soft_recall else 0.0

    result = {
        "pred_edges": len(pred),
        "gold_edges": len(gold),
        "matched_pairs": matched_pairs,
        "threshold": args.threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "soft_f1": soft_f1,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def load_cluster_labels(path: Path) -> np.ndarray:
    rows = read_jsonl(path)
    return np.array([int(r["cluster_id"]) for r in rows], dtype=np.int32)


def evaluate_cluster_summary(args: argparse.Namespace) -> None:
    labels = load_cluster_labels(Path(args.clusters_jsonl))
    vecs = l2_normalize(np.load(args.embedding_npy).astype(np.float32))
    counts = np.bincount(labels)
    out = {
        "n_samples": int(labels.shape[0]),
        "n_clusters": int(len(counts)),
        "max_cluster_ratio": float(counts.max() / labels.shape[0]),
        "cluster_sizes": {str(i): int(v) for i, v in enumerate(counts)},
    }
    if len(counts) > 1 and min(counts) > 1:
        out["silhouette_cosine"] = float(silhouette_score(vecs, labels, metric="cosine"))
    print(json.dumps(out, indent=2, ensure_ascii=False))


def parse_label_vector(value: Any) -> Optional[np.ndarray]:
    if isinstance(value, list):
        try:
            return np.array(value, dtype=np.float32)
        except Exception:
            return None
    if isinstance(value, str):
        try:
            obj = json.loads(value)
            if isinstance(obj, list):
                return np.array(obj, dtype=np.float32)
        except Exception:
            return None
    return None


def evaluate_external_alignment(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.rows_jsonl))
    labels = load_cluster_labels(Path(args.clusters_jsonl))
    if len(rows) != len(labels):
        raise ValueError("rows-jsonl and clusters-jsonl must have the same number of rows.")
    y = []
    for row in rows:
        vec = parse_label_vector(row.get(args.label_field))
        if vec is None:
            raise ValueError(f"Missing numeric label vector field: {args.label_field}")
        y.append(vec)
    y = l2_normalize(np.vstack(y))

    same_scores = []
    diff_scores = []
    auc_y = []
    auc_s = []
    rng = np.random.default_rng(args.seed)
    pairs = set()
    pair_limit = min(args.max_pairs, labels.shape[0] * (labels.shape[0] - 1) // 2)
    while len(pairs) < pair_limit:
        i, j = sorted(rng.choice(labels.shape[0], size=2, replace=False))
        pairs.add((int(i), int(j)))
    for i, j in pairs:
        score = float(y[i] @ y[j])
        same = labels[i] == labels[j]
        if same:
            same_scores.append(score)
        else:
            diff_scores.append(score)
        auc_y.append(1 if same else 0)
        auc_s.append(score)
    if not same_scores or not diff_scores:
        raise ValueError("External alignment requires at least one sampled within-cluster and one between-cluster pair.")
    out = {
        "cos_gap": float(np.mean(same_scores) - np.mean(diff_scores)),
        "same_mean_cos": float(np.mean(same_scores)),
        "diff_mean_cos": float(np.mean(diff_scores)),
        "auc_cos": float(roc_auc_score(auc_y, auc_s)) if len(set(auc_y)) == 2 else None,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation utilities for the value-mining pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    edge = sub.add_parser("edge")
    edge.add_argument("--pred-jsonl", required=True)
    edge.add_argument("--gold-jsonl", required=True)
    edge.add_argument("--pred-key", default="pairs_pred")
    edge.add_argument("--gold-key", default="pairs")
    edge.add_argument("--threshold", type=float, default=0.7)
    edge.add_argument("--embedding-model", default="BAAI/bge-m3")
    edge.add_argument("--batch-size", type=int, default=64)
    edge.add_argument("--ignore-stance", action="store_true")
    edge.set_defaults(func=evaluate_edges)

    cluster = sub.add_parser("cluster-summary")
    cluster.add_argument("--clusters-jsonl", required=True)
    cluster.add_argument("--embedding-npy", required=True)
    cluster.set_defaults(func=evaluate_cluster_summary)

    align = sub.add_parser("external-alignment")
    align.add_argument("--rows-jsonl", required=True)
    align.add_argument("--clusters-jsonl", required=True)
    align.add_argument("--label-field", default="label_vector")
    align.add_argument("--max-pairs", type=int, default=200000)
    align.add_argument("--seed", type=int, default=13)
    align.set_defaults(func=evaluate_external_alignment)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
