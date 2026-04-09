"""
RQ4 — Dependency Graph Topology as a Quality Signal
=====================================================
Methodology:
    For each IaC program, interpret the detected cloud services as nodes in a
    *service dependency graph* and connect every co-occurring service pair with
    an edge (fully-connected subgraph on detected services).  This models the
    intuition that services appearing together in one program are structurally
    dependent on each other.

    We then extract graph-level topology features — size, density, clustering
    coefficient, etc. — and test whether these features carry discriminative
    signal for predicting program quality (testing adoption) *beyond* the
    simpler n_services count that RQ2 already uses.

    Quality tiers are defined from the metadata testing signals:
        High   : ≥ 2 test files present
        Medium : exactly 1 test file present
        Low    : no test files

    A Random Forest and a Logistic Regression are evaluated with 5-fold
    stratified cross-validation (AUC metric) to quantify how well graph
    topology predicts tested vs. untested programs.
"""

import ast
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

try:
    import networkx as nx
    HAS_NX = True
except ImportError:  # pragma: no cover
    HAS_NX = False

# ---------------------------------------------------------------------------
# Service detection patterns (mirrors clustering.py for self-containment)
# ---------------------------------------------------------------------------
_SERVICE_PATTERNS: dict[str, str] = {
    "S3":             r"\bs3\b|s3[-_]bucket|aws[-_]s3|simple[-_]storage",
    "Lambda":         r"\blambda\b|aws[-_]lambda",
    "EC2":            r"\bec2\b|ec2[-_]instance",
    "RDS":            r"\brds\b|aws[-_]rds|aurora[-_](mysql|postgres)|relational[-_]db",
    "DynamoDB":       r"\bdynamodb\b|\bdynamo[-_]db\b",
    "SQS":            r"\bsqs\b",
    "SNS":            r"\bsns\b",
    "VPC":            r"\bvpc\b|virtual[-_]private[-_]cloud",
    "CloudWatch":     r"\bcloudwatch\b|aws[-_]cloudwatch",
    "IAM":            r"\biam\b|aws[-_]iam",
    "APIGateway":     r"api[-_]?gateway|apigw|rest[-_]api|http[-_]api",
    "EKS":            r"\beks\b|kubernetes|k8s",
    "ECS":            r"\becs\b|\bfargate\b",
    "Kinesis":        r"\bkinesis\b|\bfirehose\b",
    "ElastiCache":    r"\belasticache\b|elasticache[-_]redis",
    "CloudFront":     r"\bcloudfront\b|aws[-_]cloudfront",
    "Route53":        r"route[-_]?53|aws[-_]route53",
    "Cognito":        r"\bcognito\b|user[-_]pool",
    "StepFunctions":  r"step[-_]?functions|state[-_]machine",
    "EventBridge":    r"\beventbridge\b|event[-_]bus",
    "Glue":           r"\bglue\b",
    "Athena":         r"\bathena\b",
    "Redshift":       r"\bredshift\b",
    "SageMaker":      r"\bsagemaker\b|sage[-_]maker",
    "CodePipeline":   r"codepipeline|code[-_]pipeline",
    "ALB":            r"\balb\b|\bnlb\b|\belb\b|load[-_]balanc",
    "SecretsManager": r"secrets[-_]?manager|secretsmanager",
}

_COMPILED = {
    svc: re.compile(pat, re.IGNORECASE)
    for svc, pat in _SERVICE_PATTERNS.items()
}


def _parse_list_len(val) -> int:
    if pd.isna(val) or str(val).strip() in ("", "[]"):
        return 0
    try:
        lst = ast.literal_eval(str(val))
        return len(lst) if isinstance(lst, list) else 0
    except Exception:
        return 0


def _detect_services(row: pd.Series) -> list[str]:
    """Return list of detected service names for a single program row."""
    parts = []
    for col in ("name", "description", "runtime"):
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            parts.append(str(v))
    text = " ".join(parts)
    return [svc for svc, pat in _COMPILED.items() if pat.search(text)]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_service_graphs(programs: pd.DataFrame) -> dict:
    """
    Build a per-program service dependency graph.

    For each program, detect which cloud services appear in its metadata
    text (name + description + runtime).  Create a fully-connected subgraph
    on those services — every pair of co-occurring services is connected by
    an edge, modelling structural dependency.

    Parameters
    ----------
    programs : programs.csv as a DataFrame (must have an 'ID' column)

    Returns
    -------
    dict mapping program ID → nx.Graph
    """
    if not HAS_NX:
        raise ImportError("networkx is required for RQ4: pip install networkx")

    graphs: dict = {}
    for _, row in programs.iterrows():
        prog_id = row.get("ID", row.name)
        services = _detect_services(row)
        G = nx.Graph()
        G.add_nodes_from(services)
        for i in range(len(services)):
            for j in range(i + 1, len(services)):
                G.add_edge(services[i], services[j])
        graphs[prog_id] = G

    return graphs


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

GRAPH_FEATURE_COLS = [
    "n_services",
    "n_edges",
    "graph_density",
    "avg_degree",
    "max_degree",
    "is_connected",
    "clustering_coeff",
    "graph_complexity",
]


def extract_graph_features(graphs: dict) -> pd.DataFrame:
    """
    Compute graph-level topology features for each program.

    Features
    --------
    n_services       : number of service nodes (graph order)
    n_edges          : number of co-occurrence edges (graph size)
    graph_density    : 2·|E| / (|V|·(|V|−1)), zero when |V| < 2
    avg_degree       : mean node degree
    max_degree       : maximum node degree
    is_connected     : 1 if non-empty graph is connected, else 0
    clustering_coeff : average clustering coefficient (triangle density)
    graph_complexity : |V| × density  (composite: rich topology)

    Returns
    -------
    DataFrame indexed by program ID with one row per program.
    """
    if not HAS_NX:
        raise ImportError("networkx is required for RQ4: pip install networkx")

    records = []
    for prog_id, G in graphs.items():
        n = G.number_of_nodes()
        e = G.number_of_edges()
        density   = nx.density(G) if n > 1 else 0.0
        degrees   = [d for _, d in G.degree()]
        avg_deg   = float(np.mean(degrees)) if degrees else 0.0
        max_deg   = int(max(degrees)) if degrees else 0
        is_conn   = int(nx.is_connected(G)) if n > 0 else 0
        clust     = nx.average_clustering(G) if n > 1 else 0.0
        complexity = n * density

        records.append({
            "ID":              prog_id,
            "n_services":      n,
            "n_edges":         e,
            "graph_density":   round(density, 4),
            "avg_degree":      round(avg_deg, 4),
            "max_degree":      max_deg,
            "is_connected":    is_conn,
            "clustering_coeff": round(clust, 4),
            "graph_complexity": round(complexity, 4),
        })

    return pd.DataFrame(records).set_index("ID")


# ---------------------------------------------------------------------------
# Quality tiers
# ---------------------------------------------------------------------------

def assign_quality_tiers(programs: pd.DataFrame) -> pd.Series:
    """
    Assign a quality tier to every program based on testing metadata.

    Tiers
    -----
    "High"   : ≥ 2 test files present
    "Medium" : exactly 1 test file present
    "Low"    : no test files

    Returns
    -------
    pd.Series indexed by program ID, values in {"High", "Medium", "Low"}.
    """
    df = programs.copy()
    df["_n_tests"] = df["tests"].apply(_parse_list_len)

    def _tier(row) -> str:
        if row["_n_tests"] == 0:
            return "Low"
        if row["_n_tests"] >= 2:
            return "High"
        return "Medium"

    tiers = df.apply(_tier, axis=1)
    tiers.index = df["ID"]
    return tiers


# ---------------------------------------------------------------------------
# Quality classification
# ---------------------------------------------------------------------------

def classify_quality(
    graph_feat_df: pd.DataFrame,
    quality_tiers: pd.Series,
    random_state: int = 42,
) -> dict:
    """
    Evaluate whether graph topology features predict testing adoption.

    Binary classification: "High" or "Medium" (any tests) → 1  vs  "Low" → 0.
    Classifiers: Random Forest and Logistic Regression.
    Evaluation: 5-fold stratified cross-validation, AUC metric.

    Parameters
    ----------
    graph_feat_df : output of extract_graph_features()
    quality_tiers : output of assign_quality_tiers()

    Returns
    -------
    dict with keys:
        feature_names, rf_auc_cv, rf_auc_std, lr_auc_cv, lr_auc_std,
        rf_importances, roc_rf (fpr, tpr), roc_lr (fpr, tpr),
        X, y, merged
    """
    merged = graph_feat_df[GRAPH_FEATURE_COLS].join(
        quality_tiers.rename("tier"), how="inner"
    ).dropna()

    y = (merged["tier"] != "Low").astype(int)
    X = merged[GRAPH_FEATURE_COLS].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    lr = LogisticRegression(max_iter=1000, random_state=random_state)

    rf_aucs = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    lr_aucs = cross_val_score(lr, X_scaled, y, cv=cv, scoring="roc_auc")

    # Refit on full data for ROC curves and importances
    rf.fit(X, y)
    lr.fit(X_scaled, y)

    fpr_rf, tpr_rf, _ = roc_curve(y, rf.predict_proba(X)[:, 1])
    fpr_lr, tpr_lr, _ = roc_curve(y, lr.predict_proba(X_scaled)[:, 1])

    return {
        "feature_names": GRAPH_FEATURE_COLS,
        "rf_auc_cv":     float(rf_aucs.mean()),
        "rf_auc_std":    float(rf_aucs.std()),
        "lr_auc_cv":     float(lr_aucs.mean()),
        "lr_auc_std":    float(lr_aucs.std()),
        "rf_importances": dict(zip(GRAPH_FEATURE_COLS, rf.feature_importances_)),
        "roc_rf":        (fpr_rf, tpr_rf),
        "roc_lr":        (fpr_lr, tpr_lr),
        "X":             X,
        "y":             y,
        "merged":        merged,
    }


# ---------------------------------------------------------------------------
# Convenience: end-to-end pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    programs: pd.DataFrame,
    random_state: int = 42,
) -> dict:
    """
    End-to-end RQ4 pipeline.

    Returns
    -------
    dict with keys:
        graphs, graph_feat_df, quality_tiers, clf_results
    """
    print("Building per-program service dependency graphs …")
    graphs = build_service_graphs(programs)
    n_nontrivial = sum(1 for G in graphs.values() if G.number_of_nodes() > 0)
    print(f"  {len(graphs):,} total graphs  |  {n_nontrivial:,} with ≥ 1 detected service")

    print("Extracting graph topology features …")
    graph_feat_df = extract_graph_features(graphs)
    print(f"  Feature matrix: {graph_feat_df.shape[0]:,} programs × {graph_feat_df.shape[1]} features")

    print("Assigning quality tiers …")
    quality_tiers = assign_quality_tiers(programs)
    for tier, n in quality_tiers.value_counts().items():
        print(f"  {tier:8s}: {n:,}")

    print("Classifying quality from topology features (5-fold CV) …")
    clf_results = classify_quality(graph_feat_df, quality_tiers, random_state=random_state)
    print(f"  Random Forest       AUC = {clf_results['rf_auc_cv']:.3f} ± {clf_results['rf_auc_std']:.3f}")
    print(f"  Logistic Regression AUC = {clf_results['lr_auc_cv']:.3f} ± {clf_results['lr_auc_std']:.3f}")

    return {
        "graphs":        graphs,
        "graph_feat_df": graph_feat_df,
        "quality_tiers": quality_tiers,
        "clf_results":   clf_results,
    }
