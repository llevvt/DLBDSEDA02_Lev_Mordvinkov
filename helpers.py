from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from scipy.sparse import save_npz
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


def _ensure_dir(path: str) -> Path:
    """
    Checking the existence of the directory, which is used to save the results and outputs

    :param path: string, which contains the resulting directory path
    :return p: ready-to-use directory, created with path
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_vectorization(method: str,
                       X,
                       feature_names: list[str],
                       vectorizer,
                       outdir: str) -> dict:
    """
    Save artifacts of the vectorization:
      - matrix X (NPZ)
      - the list of features (CSV)
      - short description (JSON)
    Returns resume dictionary for further processing.

    :param method: method of vectorization
    :param X: results matrix
    :param feature_names: features from the matrix (rows)
    :param vectorizer: the object, which was used for vectorization
    :param outdir: the directory for results saving
    :returns meta: a dictionary with brief description of the saved vectorization data
    """

    #Ensures, that the path for the results directory exists
    outdir = _ensure_dir(outdir)

    stamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    #Saves matrix
    save_npz(outdir / f"X_{method}.npz", X)

    #Saves features
    pd.DataFrame({"feature": feature_names}).to_csv(outdir / f"features_{method}.csv", index=False)

    # Serialize vectorization parameters
    params = {}
    try:
        for k, v in vectorizer.get_params().items():
            if isinstance(v, (int, float, str, bool, list, dict, tuple, type(None))):
                params[k] = v
            else:
                params[k] = str(v)
    except Exception:
        params = {}

    meta = {
        "method": method,
        "shape": list(X.shape),
        "nnz": int(X.nnz),
        "n_features": len(feature_names),
        "vectorizer_type": type(vectorizer).__name__,
        "vectorizer_params": params,
        "created_utc": stamp,
    }

    #Saves meta data of the vectorization
    with open(outdir / f"vectorization_{method}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def _to_term_weight_pairs(items) -> List[Tuple[str, float|None]]:
    """
    Normalizes the list of term-weight pairs to unified form.

    :param items: a list of terms, which can contain different data types
    :returns pairs: a tuple of pairs of term-weight values
    """

    pairs = []

    #Goes through each row in terms, forms pairs and adds them to the output variable
    for it in items:
        if isinstance(it, (list, tuple)) and len(it) >= 2:
            term, weight = it[0], it[1]
        else:
            term, weight = it, None
        try:
            w = None if weight is None else float(weight)
        except Exception:
            w = None
        pairs.append((str(term), w))
    return pairs


def normalize_topics_input(topics: List[Dict] | List[Tuple[int, List[Tuple[str, float]]]] | List[List[Tuple[str, float]]]) -> List[Dict]:
    """
    Normalizes the diverse topics representation formats to provide unified format for further processing

    :param topics: this is a list of different objects, that contains topics descriptions
    :returns normalized: a normalized list of topics
    """
    normalized = []

    #Checks that topics is a list
    if not isinstance(topics, list):
        return normalized

    # Processes topics in case if topics is a List[Dict]
    if topics and isinstance(topics[0], dict):
        for i, t in enumerate(topics):
            terms = t.get("terms", t.get("top_terms", t.get("words", [])))
            topic_id = t.get("topic_id", t.get("id", i))
            normalized.append({
                "topic_id": int(topic_id),
                "terms": _to_term_weight_pairs(terms),
            })
        return normalized

    # Processes topics in case if topics is a List[Tuple[int, List[...]]
    if topics and isinstance(topics[0], (list, tuple)) and len(topics[0]) == 2 \
       and isinstance(topics[0][1], (list, tuple)):
        for pair in topics:
            topic_id, terms = pair
            normalized.append({
                "topic_id": int(topic_id),
                "terms": _to_term_weight_pairs(terms),
            })
        return normalized

    # Processes topics in case if each element does not have an ID (topics is a List[List[...]])
    if topics and isinstance(topics[0], (list, tuple)):
        for i, terms in enumerate(topics):
            normalized.append({
                "topic_id": i,
                "terms": _to_term_weight_pairs(terms),
            })
        return normalized

    #Returns empty list in case if topics does not suit for any of listed cases
    return normalized


def save_topics(method: str,
                topics,
                doc_topic,
                outdir: str,
                top_k_overview: int = 10) -> dict:
    """
    Processes, normalizes, and saves artefacts

    :param method: the name of topic extraction method
    :param doc_topic: raw topics list in one of the supported formats
    :param doc_topic: distribution of topics among documents matrix
    :param outdir: path to the output directory
    :param top_k_overview: terms per topic rate
    :returns meta: short resume (also saved in JSON file)
    """

    #Preparation for saving
    outdir = _ensure_dir(outdir)
    stamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    #Normalizes topics
    norm_topics = normalize_topics_input(topics)
    if not norm_topics:
        raise ValueError(
            f"Unsupported topics structure for method '{method}'. "
            f"Expected list of dicts/tuples/lists; got: {type(topics).__name__}"
        )

    #Long-list of topics with weights and ranks
    rows = []
    for t in norm_topics:
        tid = int(t["topic_id"])
        for rank, (term, weight) in enumerate(t["terms"], start=1):
            rows.append({
                "method": method,
                "topic_id": tid,
                "rank": rank,
                "term": term,
                "weight": (None if weight is None else float(weight)),
            })
    df_terms = pd.DataFrame(rows)
    df_terms.to_csv(outdir / f"topics_terms_{method}.csv", index=False)

    #Forms overview topic representation with topic id and best k terms for each topic
    overview = []
    for t in norm_topics:
        top_terms = [w[0] for w in t["terms"][:top_k_overview]]
        overview.append({
            "method": method,
            "topic_id": int(t["topic_id"]),
            f"top_{top_k_overview}_terms": ", ".join(top_terms)
        })
    df_overview = pd.DataFrame(overview).sort_values(["method", "topic_id"])
    df_overview.to_csv(outdir / f"topics_overview_{method}.csv", index=False)

    #Saves doc-topic matrix
    dt = doc_topic.toarray() if hasattr(doc_topic, "toarray") else np.asarray(doc_topic)
    dt_df = pd.DataFrame(dt, columns=[f"topic_{i:02d}" for i in range(dt.shape[1])])
    dt_df.index.name = "doc_id"
    try:
        dt_df.to_parquet(outdir / f"doc_topic_{method}.parquet", index=True)
        doc_topic_path = str(outdir / f"doc_topic_{method}.parquet")
    except Exception:
        dt_df.to_csv(outdir / f"doc_topic_{method}.csv", index=True)
        doc_topic_path = str(outdir / f"doc_topic_{method}.csv")

    #Forms and saves meta-data to JSON file
    meta = {
        "method": method,
        "n_topics": int(dt.shape[1]),
        "n_docs": int(dt.shape[0]),
        "doc_topic_path": doc_topic_path,
        "created_utc": stamp,
    }
    with open(outdir / f"topics_{method}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def save_comparison_overview(methods: list[str],
                             outdir: str,
                             filename: str = "topics_overview__comparison.csv") -> None:
    """
    Function joins all individual topics overview tables and saves them into one CSV file

    :param methods: list of topics extraction methods, used for analysis
    :param outdir: path to the output directory
    :param filename: output file for comparison CSV saving
    :returns None
    """
    #Ensures that output directory exists
    outdir = _ensure_dir(outdir)

    frames = []

    #Checks, if for ech method in list an overview file exists
    for m in methods:
        path = outdir / f"topics_overview_{m}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))

    #Joins all detected overview files into one CSV file
    if frames:
        pd.concat(frames, axis=0, ignore_index=True).to_csv(outdir / filename, index=False)