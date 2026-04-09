"""
RQ3 — Pulumi Text Alignment with Quality Signals
=================================================
Methodology:
    1. Filter Pulumi programs to those with substantive descriptions.
    2. TF-IDF vectorisation → identify the most discriminative vocabulary.
    3. LDA topic modelling → discover latent topic clusters.
    4. Compute per-document topic distributions.
    5. Correlate dominant topic assignments with quality signals:
       - has_tests   (binary: any test files present)
       - description_len  (description character length — richness proxy)
       - n_testing_fws    (number of distinct testing frameworks)
    6. Visualise: top terms per topic, topic size distribution, correlation.
"""

import ast
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

_STOP_WORDS = (
    "a an the and or of in to for with on at by from this that is are was were "
    "be been being have has had do does did will would could should may might "
    "it its this that these those minimal simple example basic program using "
    "aws pulumi infrastructure cloud deploys deploy deployment deploys creates "
    "create creates set sets up setting get gets demo sample shows shows "
    "project resource resources stack stacks app apps application applications"
).split()

# Generic boilerplate phrases that carry no topical signal
_BOILERPLATE_RE = re.compile(
    r"^a (minimal|simple|basic)\s+(aws|pulumi|cloud)?\s*(typescript|python|go|javascript|c#)?\s*"
    r"(pulumi|aws)?\s*program\.?$",
    re.IGNORECASE,
)


def _parse_list_len(val) -> int:
    """Return len() of a stringified Python list, 0 on failure."""
    if pd.isna(val) or str(val).strip() in ("", "[]"):
        return 0
    try:
        lst = ast.literal_eval(str(val))
        return len(lst) if isinstance(lst, list) else 0
    except Exception:
        return 0


def preprocess_descriptions(
    programs: pd.DataFrame,
    min_len: int = 15,
) -> pd.DataFrame:
    """
    Filter and clean Pulumi program descriptions.

    Parameters
    ----------
    programs  : full programs DataFrame (all solutions)
    min_len   : minimum character length to retain a description

    Returns
    -------
    DataFrame of Pulumi programs with quality-signal columns added and
    boilerplate descriptions removed.  Only rows with substantive
    descriptions are retained.
    """
    df = programs[programs["solution"] == "Pulumi"].copy()

    # ── Quality signals ──────────────────────────────────────────────────
    df["has_tests"] = df["tests"].apply(lambda x: 1 if _parse_list_len(x) > 0 else 0)
    df["n_test_files"] = df["tests"].apply(_parse_list_len)
    df["n_testing_fws"] = df["testing"].apply(_parse_list_len)
    df["description_len"] = df["description"].fillna("").str.len()

    # ── Quality tier ─────────────────────────────────────────────────────
    # high: has tests + description_len >= median
    med_len = df.loc[df["description_len"] > 0, "description_len"].median()
    df["quality_tier"] = "low"
    df.loc[df["description_len"] >= med_len, "quality_tier"] = "medium"
    df.loc[(df["has_tests"] == 1) & (df["description_len"] >= med_len), "quality_tier"] = "high"

    # ── Filter to substantive descriptions ───────────────────────────────
    desc = df["description"].fillna("")
    mask = (
        (desc.str.len() >= min_len)
        & (~desc.apply(lambda s: bool(_BOILERPLATE_RE.match(s.strip()))))
    )
    df = df[mask].copy()
    df["clean_description"] = desc[mask].str.lower().str.strip()

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# TF-IDF analysis
# ---------------------------------------------------------------------------

def run_tfidf(
    docs: list[str],
    max_features: int = 300,
    top_n: int = 20,
) -> tuple[TfidfVectorizer, np.ndarray, pd.DataFrame]:
    """
    Fit TF-IDF on *docs* and return top discriminative terms.

    Returns
    -------
    vectorizer  : fitted TfidfVectorizer
    X_tfidf     : (n_docs, max_features) TF-IDF matrix
    top_terms   : DataFrame with columns [term, mean_tfidf] sorted descending
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=_STOP_WORDS,
        min_df=5,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
    )
    X = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()
    mean_tfidf = np.asarray(X.mean(axis=0)).flatten()
    top_terms = (
        pd.DataFrame({"term": terms, "mean_tfidf": mean_tfidf})
        .sort_values("mean_tfidf", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return vectorizer, X, top_terms


# ---------------------------------------------------------------------------
# LDA topic modelling
# ---------------------------------------------------------------------------

def run_lda(
    docs: list[str],
    n_topics: int = 8,
    max_features: int = 300,
    random_state: int = 42,
) -> tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray]:
    """
    Fit LDA on *docs*.

    Returns
    -------
    lda         : fitted LatentDirichletAllocation
    vectorizer  : fitted CountVectorizer
    doc_topics  : (n_docs, n_topics) document-topic probability matrix
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words=_STOP_WORDS,
        min_df=5,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
    )
    X = vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        max_iter=20,
        learning_method="online",
    )
    doc_topics = lda.fit_transform(X)
    return lda, vectorizer, doc_topics


def top_words_per_topic(
    lda: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    n_words: int = 10,
) -> pd.DataFrame:
    """
    Return a DataFrame with one column per topic listing the top *n_words*.

    Columns are named 'Topic N — label' where the label is auto-assigned
    from the top word.
    """
    terms = vectorizer.get_feature_names_out()
    records = {}
    for tid, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[::-1][:n_words]
        top = [terms[i] for i in top_idx]
        records[f"Topic {tid}"] = top
    return pd.DataFrame(records)


def assign_topic_labels(top_words_df: pd.DataFrame) -> dict[str, str]:
    """
    Auto-assign a short descriptive label to each topic from its top words.
    Maps 'Topic N' → human-readable name.
    """
    # Simple heuristic: first distinctive word in the topic list
    _keyword_map = {
        # Kubernetes / container orchestration
        "kubernetes": "Kubernetes / Clusters",
        "k8s":        "Kubernetes / Clusters",
        "eks":        "Kubernetes / Clusters",
        "aks":        "Kubernetes / Clusters",
        "gke":        "GCP / Google Cloud",
        "cluster":    "Kubernetes / Clusters",
        # Serverless
        "serverless": "Serverless",
        "lambda":     "Serverless",
        "function":   "Serverless",
        # Container workloads
        "container":  "Containers / ECS",
        "docker":     "Containers / ECS",
        "ecs":        "Containers / ECS",
        # Databases
        "database":   "Databases",
        "rds":        "Databases",
        "postgres":   "Databases",
        "mysql":      "Databases",
        # Networking
        "vpc":        "Networking",
        "network":    "Networking",
        "subnet":     "Networking",
        # Storage
        "storage":    "Storage / S3",
        "bucket":     "Storage / S3",
        "s3":         "Storage / S3",
        # CI/CD
        "pipeline":   "CI/CD Pipelines",
        "cicd":       "CI/CD Pipelines",
        # Monitoring
        "monitoring":  "Monitoring / Observability",
        "cloudwatch":  "Monitoring / Observability",
        # Security
        "certificate": "Security / IAM",
        "ssl":         "Security / IAM",
        "iam":         "Security / IAM",
        "secret":      "Security / IAM",
        # Web / static hosting
        "web":         "Web / Static Sites",
        "static":      "Web / Static Sites",
        "website":     "Web / Static Sites",
        "http":        "Web / Static Sites",
        "server":      "Web / Static Sites",
        # ML / data
        "ml":          "ML / Data Engineering",
        "machine":     "ML / Data Engineering",
        "sagemaker":   "ML / Data Engineering",
        "data":        "ML / Data Engineering",
        # API / backend
        "api":         "API / Backend Services",
        "backend":     "API / Backend Services",
        "rest":        "API / Backend Services",
        "gateway":     "API / Backend Services",
        # Azure
        "azure":       "Azure / Multi-Cloud",
        "native":      "Azure / Multi-Cloud",
        # GCP / Google
        "google":      "GCP / Google Cloud",
        "gcp":         "GCP / Google Cloud",
        # Testing / demo
        "test":        "Testing & Demo",
        "demonstrate": "Testing & Demo",
        # Component resources
        "component":   "Component Resources",
        "constructs":  "Component Resources",
        "remote":      "Component Resources",
        # Provider / naming
        "alias":       "Provider Config & Naming",
        "provider":    "Provider Config & Naming",
    }
    labels = {}
    for col in top_words_df.columns:
        words = [w.lower() for w in top_words_df[col].tolist()]
        label = col  # fallback: "Topic N"
        for w in words:
            for kw, lbl in _keyword_map.items():
                if kw in w:
                    label = lbl
                    break
            else:
                continue
            break
        labels[col] = label
    return labels


# ---------------------------------------------------------------------------
# Correlation with quality signals
# ---------------------------------------------------------------------------

def topic_quality_correlation(
    pulumi_df: pd.DataFrame,
    doc_topics: np.ndarray,
    topic_labels: dict[str, str],
) -> pd.DataFrame:
    """
    Compute per-topic statistics correlated with quality signals.

    Parameters
    ----------
    pulumi_df    : filtered Pulumi DataFrame (from preprocess_descriptions)
    doc_topics   : (n_docs, n_topics) probability matrix
    topic_labels : mapping 'Topic N' → label string

    Returns
    -------
    DataFrame with one row per topic:
        topic, label, n_docs, pct_tested, avg_desc_len,
        avg_testing_fws, dominant_quality_tier
    """
    dominant = np.argmax(doc_topics, axis=1)
    rows = []
    n_topics = doc_topics.shape[1]
    for tid in range(n_topics):
        mask = dominant == tid
        sub = pulumi_df.loc[mask]
        n = mask.sum()
        rows.append({
            "topic": f"Topic {tid}",
            "label": topic_labels.get(f"Topic {tid}", f"Topic {tid}"),
            "n_docs": int(n),
            "pct_tested": round(sub["has_tests"].mean() * 100, 1) if n > 0 else 0.0,
            "avg_desc_len": round(sub["description_len"].mean(), 1) if n > 0 else 0.0,
            "avg_testing_fws": round(sub["n_testing_fws"].mean(), 2) if n > 0 else 0.0,
            "dominant_quality_tier": sub["quality_tier"].mode()[0] if n > 0 else "N/A",
        })
    return pd.DataFrame(rows).sort_values("n_docs", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience: full RQ3 pipeline
# ---------------------------------------------------------------------------

def run_rq3(
    programs: pd.DataFrame,
    n_topics: int = 8,
    max_features: int = 300,
    random_state: int = 42,
    min_desc_len: int = 15,
) -> dict:
    """
    End-to-end RQ3 pipeline.

    Returns a dict with keys:
        pulumi_df    : filtered Pulumi DataFrame with quality signals
        docs         : cleaned description strings
        tfidf_vec    : fitted TfidfVectorizer
        X_tfidf      : TF-IDF matrix
        top_terms    : top TF-IDF terms DataFrame
        lda          : fitted LDA model
        count_vec    : fitted CountVectorizer
        doc_topics   : document-topic probability matrix
        topic_words  : top-words-per-topic DataFrame
        topic_labels : dict mapping 'Topic N' → label
        corr_df      : topic-quality correlation DataFrame
    """
    print("Preprocessing Pulumi descriptions...")
    pulumi_df = preprocess_descriptions(programs, min_len=min_desc_len)
    docs = pulumi_df["clean_description"].tolist()
    print(f"  {len(docs):,} substantive Pulumi descriptions retained")

    print("Fitting TF-IDF...")
    tfidf_vec, X_tfidf, top_terms = run_tfidf(docs, max_features=max_features)
    print(f"  Vocabulary size: {len(tfidf_vec.get_feature_names_out()):,}")

    print(f"Fitting LDA (n_topics={n_topics})...")
    lda, count_vec, doc_topics = run_lda(
        docs, n_topics=n_topics, max_features=max_features, random_state=random_state
    )

    topic_words = top_words_per_topic(lda, count_vec)
    topic_labels = assign_topic_labels(topic_words)
    print("  Topics discovered:")
    for k, v in topic_labels.items():
        print(f"    {k}: {v}")

    print("Computing topic-quality correlations...")
    corr_df = topic_quality_correlation(pulumi_df, doc_topics, topic_labels)

    return {
        "pulumi_df": pulumi_df,
        "docs": docs,
        "tfidf_vec": tfidf_vec,
        "X_tfidf": X_tfidf,
        "top_terms": top_terms,
        "lda": lda,
        "count_vec": count_vec,
        "doc_topics": doc_topics,
        "topic_words": topic_words,
        "topic_labels": topic_labels,
        "corr_df": corr_df,
    }
