"""
RQ2 — Structural Archetype Clustering & Coverage Gap Analysis
=============================================================
Methodology:
    Engineer a numeric feature vector for every IaC program from PIPr metadata.
    Apply K-Means clustering to discover structural archetypes (stable,
    interpretable clusters).  Map each cluster to a human-readable archetype
    label based on its centroid profile.  Analyse coverage gaps: archetypes
    that are small, solution-skewed, or quality-poor are underrepresented in
    the current dataset and should be oversampled when curating an LLM
    fine-tuning dataset.

Feature groups
--------------
1. Solution type     one-hot: CDK / Pulumi / CDKTF
2. Language          one-hot: typescript / python / csharp / go / javascript / java
3. Quality signals   has_tests, n_test_files, n_testing_frameworks
4. Text richness     description_len (Pulumi-only)
5. Complexity proxy  runtime_len (length of the runtime/entry-point string)
6. Repo signals      log_forks, is_fork, programs_per_repo
7. Service presence  one binary flag per detected cloud service (27 services)
"""

import ast
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Cloud-service keyword patterns (mirrors RQ1 — kept here for self-containment)
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

SERVICE_COLS = [f"svc_{s}" for s in _SERVICE_PATTERNS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_list_len(val) -> int:
    """Return len() of a stringified Python list, 0 on failure."""
    if pd.isna(val) or str(val).strip() in ("", "[]"):
        return 0
    try:
        lst = ast.literal_eval(str(val))
        return len(lst) if isinstance(lst, list) else 0
    except Exception:
        return 0


def _runtime_filename(runtime: str) -> str:
    """Extract the filename component of a runtime command string."""
    tokens = str(runtime).strip().split()
    for tok in reversed(tokens):
        if "/" in tok:
            return tok.rsplit("/", 1)[-1]
    return tokens[-1] if tokens else ""


def _build_corpus(row: pd.Series) -> str:
    parts = []
    for col in ("name", "description"):
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            parts.append(str(v))
    rt = row.get("runtime")
    if pd.notna(rt):
        parts.append(_runtime_filename(str(rt)))
    return " ".join(parts)


def _detect_services(text: str) -> dict[str, int]:
    return {f"svc_{svc}": int(bool(pat.search(text))) for svc, pat in _COMPILED.items()}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(
    programs: pd.DataFrame,
    repos: pd.DataFrame,
    testing: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the full feature matrix for all programs.

    Parameters
    ----------
    programs : programs.csv as a DataFrame
    repos    : repositories.csv as a DataFrame
    testing  : testing-files.csv as a DataFrame

    Returns
    -------
    DataFrame indexed by program ID with one row per program and one column
    per feature.  Categorical columns are already one-hot encoded.  All
    numeric columns are raw (un-scaled); call build_feature_matrix() for the
    scaled version ready for clustering.
    """
    df = programs.copy()

    # ── Quality signals ──────────────────────────────────────────────────
    df["has_tests"] = df["tests"].apply(_parse_list_len).clip(upper=1)
    df["n_test_files"] = df["tests"].apply(_parse_list_len)
    df["n_testing_fws"] = df["testing"].apply(_parse_list_len)

    # ── Text richness ────────────────────────────────────────────────────
    df["description_len"] = df["description"].fillna("").str.len()

    # ── Complexity proxy ─────────────────────────────────────────────────
    df["runtime_len"] = df["runtime"].fillna("").str.len()

    # ── Service detection ────────────────────────────────────────────────
    corpus = df.apply(_build_corpus, axis=1)
    service_rows = corpus.apply(_detect_services)
    svc_df = pd.DataFrame(list(service_rows), index=df.index)
    df = pd.concat([df, svc_df], axis=1)
    df["n_services"] = svc_df.sum(axis=1)

    # ── Repo-level signals ───────────────────────────────────────────────
    prog_per_repo = (
        programs.groupby("repository").size().rename("programs_per_repo")
    )
    df = df.join(prog_per_repo, on="repository")

    repo_sub = repos[["ID", "forks", "fork"]].copy()
    repo_sub.columns = ["repository", "forks", "is_fork"]
    df = df.merge(repo_sub, on="repository", how="left")
    df["forks"] = df["forks"].fillna(0)
    df["is_fork"] = df["is_fork"].fillna(False).astype(int)
    df["log_forks"] = np.log1p(df["forks"])

    # ── One-hot: solution ────────────────────────────────────────────────
    sol_dummies = pd.get_dummies(df["solution"], prefix="sol").astype(int)
    # normalise column names
    sol_dummies.columns = [c.replace(" ", "_") for c in sol_dummies.columns]
    df = pd.concat([df, sol_dummies], axis=1)

    # ── One-hot: language (top 6 + "other") ─────────────────────────────
    top_langs = ["typescript", "python", "csharp", "go", "javascript", "java"]
    _lang_bucket = df["language"].where(df["language"].isin(top_langs), "other")
    lang_dummies = pd.get_dummies(_lang_bucket, prefix="lang").astype(int)
    df = pd.concat([df, lang_dummies], axis=1)

    df = df.set_index("ID")
    return df


def build_feature_matrix(feature_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Select the numeric clustering columns and return a StandardScaler-scaled
    NumPy array plus the ordered list of column names.

    Returns
    -------
    (X_scaled, feature_names)
    """
    numeric_cols = (
        # solution
        [c for c in feature_df.columns if c.startswith("sol_")]
        # language
        + [c for c in feature_df.columns if c.startswith("lang_")]
        # quality
        + ["has_tests", "n_test_files", "n_testing_fws"]
        # complexity
        + ["description_len", "runtime_len", "n_services"]
        # repo
        + ["log_forks", "is_fork", "programs_per_repo"]
        # services
        + SERVICE_COLS
    )
    # keep only columns that actually exist (guard against edge cases)
    numeric_cols = [c for c in numeric_cols if c in feature_df.columns]
    X = feature_df[numeric_cols].fillna(0).values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, numeric_cols


# ---------------------------------------------------------------------------
# Elbow / silhouette search
# ---------------------------------------------------------------------------

def elbow_silhouette(
    X: np.ndarray,
    k_range: range = range(2, 13),
    random_state: int = 42,
    sample_size: int = 5000,
) -> pd.DataFrame:
    """
    Compute inertia and silhouette scores for a range of k values.

    Uses a random sample of *sample_size* rows for silhouette (O(n²) metric)
    while running K-Means on the full matrix.

    Returns a DataFrame with columns: k, inertia, silhouette.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    sil_idx = rng.choice(n, size=min(sample_size, n), replace=False)

    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X[sil_idx], labels[sil_idx])
        records.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"  k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# K-Means clustering
# ---------------------------------------------------------------------------

# Archetype label assignment rules: ordered list of (label, required_features).
# The first rule whose *required* features all score above the cluster mean
# in the centroid is selected.  Rules are ordered from most-specific to
# least-specific so that high-precision service combos fire before broad
# solution-level rules.  The empty-list rule at the end is a catch-all.
#
# Calibrated for k=10 (global silhouette maximum in the PIPr dataset sweep).
# At k=10 the Pulumi programs split into three structurally distinct sub-
# clusters (by language, description richness, and repo structure), and the
# Serverless archetype splits into a Lambda+S3+CloudFront static-hosting
# variant and a complex CDK-heavy multi-service variant.
_ARCHETYPE_RULES: list[tuple[str, list[str]]] = [
    # ── High-precision service combinations (checked first) ──────────────
    ("Serverless + Static Hosting", ["svc_Lambda", "svc_S3", "svc_CloudFront"]),
    ("Complex Serverless",          ["svc_Lambda", "n_services"]),
    ("Kubernetes Workload",         ["svc_EKS"]),
    # Multi-cloud high-complexity (captured before solution-level rules
    # so that mixed-solution, service-rich clusters aren't mis-labelled
    # as CDK or Pulumi on a marginal solution signal alone)
    ("Multi-Cloud High-Complexity", ["n_services", "description_len"]),
    ("Data Pipeline",               ["svc_S3", "svc_Kinesis"]),
    ("CI/CD Automation",            ["svc_CodePipeline", "svc_S3"]),
    # ── Solution-specific (ordered high → low specificity) ───────────────
    ("CDKTF Infrastructure",        ["sol_CDKTF"]),
    ("Tested CDK Deployments",      ["has_tests", "sol_AWS_CDK"]),
    ("CDK General Infrastructure",  ["sol_AWS_CDK"]),
    ("Pulumi Multi-Language Repos", ["sol_Pulumi", "programs_per_repo"]),
    ("Pulumi General Cloud",        ["sol_Pulumi", "description_len"]),
    # ── Catch-all ────────────────────────────────────────────────────────
    ("Minimal / Untested",          []),
]


def _assign_archetype(centroid: dict[str, float], global_mean: dict[str, float]) -> str:
    """
    Assign a human-readable archetype label to a cluster centroid.

    A rule fires when *all* of its required features have a centroid value
    strictly above the global feature mean.
    """
    for label, required in _ARCHETYPE_RULES:
        if not required:
            return label
        if all(centroid.get(f, 0) > global_mean.get(f, 0) for f in required):
            return label
    return "General IaC"


def run_kmeans(
    X: np.ndarray,
    feature_names: list[str],
    k: int = 7,
    random_state: int = 42,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Fit K-Means and assign archetype labels.

    Returns
    -------
    labels       : cluster label array (len = n_programs)
    archetypes   : list of archetype name strings (len = k)
    centroid_df  : DataFrame of unscaled centroids (rows = clusters)
    """
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    centroid_df = pd.DataFrame(km.cluster_centers_, columns=feature_names)
    global_mean = {f: centroid_df[f].mean() for f in feature_names}

    archetypes: list[str] = []
    for idx in range(k):
        centroid = centroid_df.iloc[idx].to_dict()
        archetypes.append(_assign_archetype(centroid, global_mean))

    return labels, archetypes, centroid_df


# ---------------------------------------------------------------------------
# PCA projection for visualisation
# ---------------------------------------------------------------------------

def pca_projection(X: np.ndarray, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """Return a 2-D PCA projection of *X*."""
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X)


# ---------------------------------------------------------------------------
# Coverage-gap analysis
# ---------------------------------------------------------------------------

def coverage_gap_analysis(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    archetypes: list[str],
) -> pd.DataFrame:
    """
    Compute per-archetype statistics relevant to LLM fine-tuning coverage.

    Returns a DataFrame with one row per archetype cluster containing:
        archetype, n_programs, solution_diversity (Shannon entropy),
        pct_tested, dominant_solution, pct_dominant,
        coverage_gap_score  (higher → more underrepresented)
    """
    df = feature_df.copy()
    df["cluster"] = labels
    df["archetype"] = [archetypes[lbl] for lbl in labels]

    solution_col = None
    for col in df.columns:
        if col == "solution":
            solution_col = "solution"
            break

    rows = []
    total_n = len(df)

    for cid, arch in enumerate(archetypes):
        sub = df[df["cluster"] == cid]
        n = len(sub)

        # solution distribution
        sol_col = "sol_AWS_CDK"  # fallback to dummy if 'solution' missing
        sol_counts: dict[str, int] = {}
        for sol_dummy, sol_name in [
            ("sol_AWS_CDK", "AWS CDK"),
            ("sol_Pulumi", "Pulumi"),
            ("sol_CDKTF", "CDKTF"),
        ]:
            if sol_dummy in sub.columns:
                sol_counts[sol_name] = int(sub[sol_dummy].sum())

        total_sol = sum(sol_counts.values()) or 1
        # Shannon entropy (diversity)
        entropy = 0.0
        for cnt in sol_counts.values():
            p = cnt / total_sol
            if p > 0:
                entropy -= p * np.log2(p)
        max_entropy = np.log2(3)  # 3 solutions

        dominant = max(sol_counts, key=sol_counts.get) if sol_counts else "N/A"
        pct_dominant = sol_counts.get(dominant, 0) / total_sol * 100

        pct_tested = sub["has_tests"].mean() * 100 if "has_tests" in sub.columns else 0.0
        pct_dataset = n / total_n * 100

        # Coverage gap score: low diversity + small size + low testing → high gap
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0
        size_score = 1 - min(n / (total_n / len(archetypes)), 1.0)  # 1 = tiny cluster
        quality_score = 1 - (pct_tested / 100)
        gap_score = round((size_score * 0.5 + (1 - diversity_score) * 0.3 + quality_score * 0.2), 3)

        rows.append(
            {
                "cluster_id": cid,
                "archetype": arch,
                "n_programs": n,
                "pct_dataset": round(pct_dataset, 1),
                "pct_tested": round(pct_tested, 1),
                "dominant_solution": dominant,
                "pct_dominant": round(pct_dominant, 1),
                "solution_diversity": round(entropy, 3),
                "coverage_gap_score": gap_score,
            }
        )

    return pd.DataFrame(rows).sort_values("coverage_gap_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience: build everything in one call
# ---------------------------------------------------------------------------

def run_full_pipeline(
    programs: pd.DataFrame,
    repos: pd.DataFrame,
    testing: pd.DataFrame,
    k: int = 7,
    random_state: int = 42,
) -> dict:
    """
    End-to-end RQ2 pipeline.

    Returns a dict with keys:
        feature_df   : enriched programs DataFrame with all features
        X_scaled     : scaled feature matrix
        feature_names: list of feature column names
        labels       : cluster label array
        archetypes   : list of archetype name strings
        centroid_df  : cluster centroid DataFrame
        gap_df       : coverage gap analysis DataFrame
        X_pca        : 2-D PCA projection
    """
    print("Building feature matrix...")
    feature_df = build_features(programs, repos, testing)
    X_scaled, feature_names = build_feature_matrix(feature_df)
    print(f"  {len(feature_df):,} programs × {len(feature_names)} features")

    print(f"Fitting K-Means (k={k})...")
    labels, archetypes, centroid_df = run_kmeans(X_scaled, feature_names, k=k, random_state=random_state)
    for i, name in enumerate(archetypes):
        print(f"  Cluster {i}: {name}  (n={int((labels == i).sum()):,})")

    print("Computing 2-D PCA projection...")
    X_pca = pca_projection(X_scaled, random_state=random_state)

    print("Coverage gap analysis...")
    gap_df = coverage_gap_analysis(feature_df, labels, archetypes)

    return {
        "feature_df": feature_df,
        "X_scaled": X_scaled,
        "feature_names": feature_names,
        "labels": labels,
        "archetypes": archetypes,
        "centroid_df": centroid_df,
        "gap_df": gap_df,
        "X_pca": X_pca,
    }
