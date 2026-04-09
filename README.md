# IaC Pattern Mining

**Course**: CSCE 676 — Data Mining and Analysis  
**Semester**: Spring 2025  
**Student**: Divyanshu Singh

## Project Overview

This project applies data mining techniques to Infrastructure as Code (IaC) programs to uncover architectural patterns, identify structural archetypes, and discover quality signals across cloud ecosystems (AWS CDK and Pulumi). The analysis pipeline covers frequent pattern mining, clustering, text/topic modeling, and graph-based anomaly detection.

## Research Questions

| # | Research Question |
|---|-------------------|
| RQ1 | What service co-occurrence patterns are most frequent across IaC ecosystems? |
| RQ2 | What structural archetypes exist in IaC programs, and which are underrepresented in existing datasets? |
| RQ3 | Do program descriptions (text signals) correlate with quality indicators? |
| RQ4 | Can service dependency graph topology predict program quality? |

## Dataset

**Selected Dataset: The PIPr Dataset (Public Infrastructure as Code Programs)**

37,712 real-world IaC programs (AWS CDK, Pulumi, CDKTF) across TypeScript, Python, and Go.

- **AWS CDK**: ~23,940 programs; ~38% have tests (used as a proxy for production-grade quality)
- **Pulumi**: remaining programs; ~1.3% have tests
- **Curated subset**: 3,987 high-quality programs used for deep analysis ([results/curated_dataset.csv](results/curated_dataset.csv))

Source: Daniel Sokolowski et al., *The PIPr Dataset of Public Infrastructure as Code Programs*, MSR '24. [https://doi.org/10.1145/3643991.3644888](https://doi.org/10.1145/3643991.3644888)

## Analysis Pipeline

### Source Scripts ([src/](src/))

| Script | Purpose |
|--------|---------|
| [src/fp_growth.py](src/fp_growth.py) | FP-Growth frequent pattern mining — transaction construction from metadata text signals, per-ecosystem association rule extraction |
| [src/clustering.py](src/clustering.py) | K-Means clustering with PCA projection, feature engineering, and coverage gap evaluation |
| [src/text_analysis.py](src/text_analysis.py) | TF-IDF top-term extraction and LDA topic modeling on program descriptions, correlated with quality signals |
| [src/graph_features.py](src/graph_features.py) | Service dependency graph construction, topology feature extraction, and quality prediction via ROC analysis |

### Final Notebook

All analyses are integrated in [final_notebook.ipynb](final_notebook.ipynb), structured as:

1. Introduction & Motivation
2. Dataset Overview
3. EDA — Distribution of Tools, Languages, Testing
4. Data Curation — Filtering for high-quality programs
5. **RQ1** — FP-Growth Co-occurrence Analysis
6. **RQ2** — K-Means Clustering & Archetype Discovery
7. **RQ3** — Text Mining & Topic Modeling
8. **RQ4** — Graph Topology & Quality Prediction
9. Curated Dataset Composition
10. Conclusions, Limitations & Future Work

## Key Findings

### RQ1 — Frequent Service Patterns (FP-Growth)
- **AWS CDK**: ECS ↔ ALB co-occurrence lift = **3.87**
- **Pulumi**: Lambda ↔ APIGateway co-occurrence lift = **4.37**
- Association rules reveal canonical cloud architecture patterns per ecosystem

### RQ2 — Structural Archetypes (K-Means Clustering)
- Identified distinct archetypes: serverless, containerized, storage-centric, networking-heavy
- Coverage gap analysis surfaces underrepresented archetypes in the curated dataset
- Results: [results/rq2_cluster_labels.csv](results/rq2_cluster_labels.csv)

### RQ3 — Text Signals & Quality (TF-IDF + LDA)
- LDA topics align with architectural domains (web services, data pipelines, ML infrastructure)
- Certain topic distributions correlate positively with testing adoption (quality proxy)
- Results: [results/rq3_topics.csv](results/rq3_topics.csv)

### RQ4 — Graph Topology & Quality (NetworkX + ROC)
- Service dependency graph features (degree, clustering coefficient, connected components) are predictive of program quality
- ROC-AUC analysis identifies the most discriminative graph topology features
- Results visualized in [results/figures/rq4_roc_importance.png](results/figures/rq4_roc_importance.png)

## Results & Figures

| Figure | Description |
|--------|-------------|
| [results/figures/s5_rq1_association_rules.png](results/figures/s5_rq1_association_rules.png) | Top association rules per ecosystem (RQ1) |
| [results/figures/rq2_pca_archetypes.png](results/figures/rq2_pca_archetypes.png) | PCA projection of K-Means archetypes (RQ2) |
| [results/figures/rq2_elbow_silhouette.png](results/figures/rq2_elbow_silhouette.png) | Elbow & silhouette plots for K selection (RQ2) |
| [results/figures/rq2_centroid_heatmap.png](results/figures/rq2_centroid_heatmap.png) | Cluster centroid feature heatmap (RQ2) |
| [results/figures/rq2_coverage_gap.png](results/figures/rq2_coverage_gap.png) | Underrepresented archetype coverage gap (RQ2) |
| [results/figures/rq3_tfidf_top_terms.png](results/figures/rq3_tfidf_top_terms.png) | TF-IDF top terms per quality tier (RQ3) |
| [results/figures/rq3_lda_topics.png](results/figures/rq3_lda_topics.png) | LDA topic distributions (RQ3) |
| [results/figures/rq3_topic_quality_correlation.png](results/figures/rq3_topic_quality_correlation.png) | Topic–quality correlation heatmap (RQ3) |
| [results/figures/rq3_quality_heatmap.png](results/figures/rq3_quality_heatmap.png) | Quality signal heatmap (RQ3) |
| [results/figures/rq4_graph_distributions.png](results/figures/rq4_graph_distributions.png) | Graph topology feature distributions (RQ4) |
| [results/figures/rq4_roc_importance.png](results/figures/rq4_roc_importance.png) | ROC curves and feature importance (RQ4) |
| [results/figures/s9_curated_dataset_composition.png](results/figures/s9_curated_dataset_composition.png) | Curated dataset composition breakdown |

## Project Timeline

| Date | Milestone |
|------|-----------|
| Feb 12 | Dataset Selection and EDA (Checkpoint 1) |
| Mar 5 | Initial Research Questions (Checkpoint 2) |
| Apr 2 | Deep Dive Analysis (Checkpoint 3) |
| Apr 8 | Full analysis pipeline complete (RQ1–RQ4), notebook finalized |
| Apr 20 | Project Showcase |
| Apr 27 | Final Deliverable |

## References

- Daniel Sokolowski et al. *The PIPr Dataset of Public Infrastructure as Code Programs.* MSR '24. [https://doi.org/10.1145/3643991.3644888](https://doi.org/10.1145/3643991.3644888)
- IaC-Eval: A Code Generation Benchmark for Cloud Infrastructure-as-Code Programs. [https://doi.org/10.52202/079017-4273](https://doi.org/10.52202/079017-4273)
- Multi-IaC-Eval: Benchmarking Cloud Infrastructure as Code Across Multiple Formats. [https://doi.org/10.48550/arXiv.2509.05303](https://doi.org/10.48550/arXiv.2509.05303)

## Contact

**Divyanshu Singh**  
Texas A&M University  
Email: [divyanshu@tamu.edu](mailto:divyanshu@tamu.edu)
