"""
RQ1 — Frequent Resource Co-occurrence Pattern Mining
=====================================================
Methodology:
    Parse available text metadata (program names, Pulumi descriptions, and
    CDK/CDKTF entry-point filenames extracted from the `runtime` field) to
    detect mentions of known cloud-service identifiers.  Each program whose
    metadata yields ≥ 2 detected services becomes a *transaction* in the
    frequent-pattern mining sense.  FP-Growth is then run per-solution
    (AWS CDK, Pulumi, CDKTF) and association rules are extracted.

Limitation note:
    The PIPr raw source-code corpus (58 GB) is not stored locally.  This
    module therefore relies on metadata-derived service signals.  The
    extraction has high *precision* (we only fire on unambiguous service
    identifiers) at the cost of *recall* (programs with no informative text
    metadata are omitted).  Coverage statistics are printed by
    build_transactions() so the analyst can quantify the tractable subset.
"""

import re
import ast
from typing import Optional

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ---------------------------------------------------------------------------
# Cloud-service keyword patterns (AWS-centric, conservative / high-precision)
# ---------------------------------------------------------------------------
SERVICE_PATTERNS: dict[str, str] = {
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
    "CodeBuild":      r"codebuild|code[-_]build",
    "ALB":            r"\balb\b|\bnlb\b|\belb\b|load[-_]balanc",
    "SES":            r"\bses\b|simple[-_]email[-_]service",
    "WAF":            r"\bwaf\b",
    "SecretsManager": r"secrets[-_]?manager|secretsmanager",
    "EFS":            r"\befs\b|elastic[-_]file",
    "Rekognition":    r"\brekognition\b",
    "Polly":          r"\bpolly\b",
    "GCS":            r"\bgcs\b|google[-_]cloud[-_]storage",
    "GKE":            r"\bgke\b|google[-_]kubernetes",
    "BigQuery":       r"\bbigquery\b|big[-_]query",
    "AzureBlob":      r"azure[-_]blob|blob[-_]storage",
    "AKS":            r"\baks\b|azure[-_]kubernetes",
    "ServiceBus":     r"service[-_]bus",
}

# Compile patterns once for performance
_COMPILED: dict[str, re.Pattern] = {
    svc: re.compile(pattern, re.IGNORECASE)
    for svc, pattern in SERVICE_PATTERNS.items()
}


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _entry_point_filename(runtime: str) -> str:
    """
    Extract just the filename from a runtime string.

    Examples
    --------
    'npx ts-node --prefer-ts-exts bin/s3-lambda.ts'  →  's3-lambda.ts'
    'python3 app.py'                                  →  'app.py'
    'nodejs'                                          →  'nodejs'
    """
    # grab the last path component (after any slash), strip CLI flags
    tokens = runtime.strip().split()
    for token in reversed(tokens):
        if "/" in token:
            return token.rsplit("/", 1)[-1]
    return tokens[-1] if tokens else ""


def build_corpus(row: pd.Series) -> str:
    """
    Concatenate all available text signals for a single program row.

    Text sources (in order of priority):
        - `name`        — program name (mostly Pulumi)
        - `description` — natural-language description (98.5% Pulumi, ~0% others)
        - `runtime`     — entry-point filename extracted from the runtime string
                          (CDK TypeScript files often encode the stack name)
    """
    parts: list[str] = []
    for col in ("name", "description"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val))
    runtime = row.get("runtime")
    if pd.notna(runtime):
        parts.append(_entry_point_filename(str(runtime)))
    return " ".join(parts)


def extract_services(text: str) -> frozenset[str]:
    """Return the set of detected cloud-service names in *text*."""
    if not text:
        return frozenset()
    found: set[str] = set()
    for svc, pattern in _COMPILED.items():
        if pattern.search(text):
            found.add(svc)
    return frozenset(found)


# ---------------------------------------------------------------------------
# Transaction building
# ---------------------------------------------------------------------------

def build_transactions(
    programs: pd.DataFrame,
    min_services: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build a transaction table from the programs metadata.

    Parameters
    ----------
    programs      : programs.csv loaded as a DataFrame (must contain columns:
                    solution, name, description, runtime).
    min_services  : minimum number of distinct services that must be detected
                    for a program to be included as a transaction (default 2).
    verbose       : print coverage statistics per solution.

    Returns
    -------
    DataFrame with columns: program_id, solution, language, has_tests, services
        where `services` is a frozenset of detected service names.
    """
    corpus = programs.apply(build_corpus, axis=1)
    services = corpus.apply(extract_services)

    result = programs[["ID", "solution", "language", "has_tests"]].copy()
    result["services"] = services
    result["n_services"] = services.apply(len)

    if verbose:
        print("Service-detection coverage (programs with detected services):")
        print(f"  {'Solution':<12}  {'Total':>7}  {'≥1 svc':>8}  {'≥2 svc':>8}  {'≥1 %':>7}  {'≥2 %':>7}")
        print("  " + "─" * 58)
        for sol in ["AWS CDK", "Pulumi", "CDKTF"]:
            sub = result[result["solution"] == sol]
            n1 = (sub["n_services"] >= 1).sum()
            n2 = (sub["n_services"] >= min_services).sum()
            n = len(sub)
            print(
                f"  {sol:<12}  {n:>7,}  {n1:>8,}  {n2:>8,}"
                f"  {n1/n*100:>6.1f}%  {n2/n*100:>6.1f}%"
            )
        print()

    return result[result["n_services"] >= min_services].reset_index(drop=True)


# ---------------------------------------------------------------------------
# FP-Growth + association rules
# ---------------------------------------------------------------------------

def _encode_transactions(transaction_series: pd.Series) -> pd.DataFrame:
    """One-hot encode a Series of frozensets for mlxtend."""
    te = TransactionEncoder()
    te_array = te.fit_transform([list(s) for s in transaction_series])
    return pd.DataFrame(te_array, columns=te.columns_)


def run_fp_growth(
    transactions: pd.DataFrame,
    solution: str,
    min_support: float = 0.05,
    min_confidence: float = 0.30,
    min_lift: float = 1.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run FP-Growth + association-rule extraction for one solution.

    Parameters
    ----------
    transactions   : output of build_transactions() (already filtered to ≥2 svc)
    solution       : one of 'AWS CDK', 'Pulumi', 'CDKTF'
    min_support    : minimum support threshold (fraction of transactions)
    min_confidence : minimum confidence threshold
    min_lift       : minimum lift threshold for returned rules
    verbose        : print summary statistics

    Returns
    -------
    (freq_itemsets, rules) — both as DataFrames, or (empty, empty) if the
    solution has too few transactions for the given support threshold.
    """
    subset = transactions[transactions["solution"] == solution]
    n = len(subset)
    empty = pd.DataFrame(), pd.DataFrame()

    if n < 10:
        if verbose:
            print(f"  [{solution}] Skipped — only {n} transactions (need ≥ 10).")
        return empty

    encoded = _encode_transactions(subset["services"])

    try:
        freq_items = fpgrowth(
            encoded,
            min_support=min_support,
            use_colnames=True,
            max_len=None,
        )
    except ValueError as exc:
        if verbose:
            print(f"  [{solution}] FP-Growth error: {exc}")
        return empty

    if freq_items.empty:
        if verbose:
            print(
                f"  [{solution}] No frequent itemsets at support ≥ {min_support:.0%} "
                f"(n={n:,}). Try lowering min_support."
            )
        return empty

    rules = association_rules(
        freq_items,
        metric="confidence",
        min_threshold=min_confidence,
        num_itemsets=len(freq_items),
    )
    rules = rules[rules["lift"] >= min_lift].copy()

    # Add human-readable columns
    rules["antecedent_str"] = rules["antecedents"].apply(
        lambda x: " + ".join(sorted(x))
    )
    rules["consequent_str"] = rules["consequents"].apply(
        lambda x: " + ".join(sorted(x))
    )
    rules["rule"] = rules["antecedent_str"] + "  →  " + rules["consequent_str"]
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    if verbose:
        print(
            f"  [{solution}]  n={n:,}  |  "
            f"freq_itemsets={len(freq_items)}  |  "
            f"rules={len(rules)}  |  "
            f"support≥{min_support:.0%}  confidence≥{min_confidence:.0%}"
        )
    return freq_items, rules


def run_all_solutions(
    transactions: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.30,
    min_lift: float = 1.0,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Convenience wrapper: run FP-Growth for all three solutions.

    Returns
    -------
    dict mapping solution name → (freq_itemsets_df, rules_df)
    """
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    print("Running FP-Growth per solution:")
    for sol in ["AWS CDK", "Pulumi", "CDKTF"]:
        fi, rules = run_fp_growth(
            transactions, sol,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
        )
        results[sol] = (fi, rules)
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def top_rules_table(rules: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Return the top-n rules by lift, with key metrics formatted."""
    if rules.empty:
        return pd.DataFrame()
    cols = ["rule", "support", "confidence", "lift"]
    top = rules[cols].head(n).copy()
    top["support"]    = top["support"].map("{:.1%}".format)
    top["confidence"] = top["confidence"].map("{:.1%}".format)
    top["lift"]       = top["lift"].map("{:.2f}".format)
    return top.reset_index(drop=True)


def save_rules_csv(
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    output_path: str,
) -> None:
    """Concatenate rules from all solutions and save to CSV."""
    frames: list[pd.DataFrame] = []
    for sol, (_, rules) in results.items():
        if not rules.empty:
            rules = rules.copy()
            rules.insert(0, "solution", sol)
            frames.append(rules)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)
        print(f"Saved {sum(len(f) for f in frames):,} rules to {output_path}")
    else:
        print("No rules to save.")
