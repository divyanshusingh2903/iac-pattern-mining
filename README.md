# IaC Pattern Mining

**Course**: CSCE 676 - Data Mining and Analysis  
**Semester**: Spring 2025  
**Student**: Divyanshu Singh

## Project Overview

This project explores patterns, structures, and anomalies in Infrastructure as Code (IaC) datasets, with a focus on cloud infrastructure configurations across AWS, GCP, and Azure. The goal is to apply data mining techniques to understand common architectural patterns, identify gaps in existing training datasets, and discover insights about how infrastructure code is written and structured.

## Research Questions

- What are the most common infrastructure patterns across different cloud providers?
- How do IaC code structures differ between Terraform, Pulumi, and other IaC languages?
- What gaps exist in current IaC datasets for LLM training purposes?
- Can we identify anomalous or potentially problematic infrastructure configurations?
- How do infrastructure patterns cluster by domain (e.g., web apps, data pipelines, ML systems)?

## Datasets

### Primary Dataset Candidates

1. **AutoIaC Evaluation Dataset**
   - Source: [Hugging Face - autoiac-project/iac-eval](https://huggingface.co/datasets/autoiac-project/iac-eval)
   - Description: Evaluation dataset for Infrastructure as Code generation

2. **Infrastructure as Code Dataset (Zenodo)**
   - Source: [Zenodo - IaC Dataset](https://zenodo.org/records/10173400)
   - Description: Large-scale collection of real-world IaC files

3. **Multi-IaC-Eval**
   - Source: [Hugging Face - AmazonScience/Multi-IaC-Eval](https://huggingface.co/datasets/AmazonScience/Multi-IaC-Eval)
   - Description: 2M code-comment pairs from open source repositories

## Dataset Selection

**Selected Dataset: The PIPr Dataset (Public Infrastructure as Code Programs)**

After evaluating candidates including IaC-Eval and Multi-IaC-Eval, we selected **PIPr** for the following reasons:
1.  **Scale**: With 37,712 real-world IaC programs, it offers volume essential for frequent pattern mining and graph neural networks.
2.  **Real-World Quality**: Unlike synthetic datasets, PIPr contains real user code with bugs and variety, making it ideal for anomaly detection.
3.  **Diversity**: It covers AWS CDK, Pulumi, and CDKTF across multiple languages (TypeScript, Python, Go), supporting robust clustering tasks.

## Data Mining Techniques

### Course-Related Techniques
- **Text Mining**: Analyzing IaC code structure and documentation
- **Clustering**: Grouping similar infrastructure patterns
- **Anomaly Detection**: Identifying unusual or potentially problematic configurations
- **Pattern Mining**: Discovering frequent code patterns and anti-patterns

### Beyond-Course Techniques
- **Graph Neural Networks**: Modeling infrastructure dependency graphs
- **Code Embeddings**: Using transformer-based models for code representation
- **Sequence Mining**: Analyzing temporal evolution of infrastructure code
- **Topic Modeling**: Identifying common infrastructure themes and use cases

## Project Timeline

- **Feb 12**: Dataset Selection and EDA (Checkpoint 1)
- **Mar 5**: Initial Research Question (Checkpoint 2)
- **Apr 2**: Deep Dive Analysis (Checkpoint 3)
- **Apr 20**: Project Showcase
- **Apr 27**: Final Deliverable

## Exploratory Data Analysis (EDA)

Initial analysis of the PIPr dataset (conducted in `01_dataset_exploration.ipynb`) reveals:

-   **Tool Dominance**: The dataset is heavily skewed towards **AWS CDK** (23,940 programs), with **TypeScript** as the dominant language.
-   **Testing Disparity**: Testing adoption is uneven. AWS CDK programs have a **~38% testing rate**, while Pulumi programs are largely untested (~1.3%).
-   **Implication**: The "tested" subset of AWS CDK programs likely represents "production-grade" infrastructure and will be a key focus for mining high-quality patterns.

## References

- IaC-Eval: A Code Generation Benchmark for Cloud Infrastructure-as-Code Programs[https://neurips.cc/virtual/2024/poster/97835](https://doi.org/10.52202/079017-4273)
- Zenodo IaC Dataset: Daniel Sokolowski, David Spielmann, and Guido Salvaneschi. 2024. The PIPr Dataset of Public Infrastructure as Code Programs. In Proceedings of the 21st International Conference on Mining Software Repositories (MSR '24). Association for Computing Machinery, New York, NY, USA, 498–503. [https://doi.org/10.1145/3643991.3644888](https://doi.org/10.1145/3643991.3644888)
- Multi-IaC-Eval: Benchmarking Cloud Infrastructure as Code Across Multiple Formats. [https://doi.org/10.48550/arXiv.2509.05303](https://doi.org/10.48550/arXiv.2509.05303)

## Contact

**Divyanshu Singh**  
Texas A&M University  
Email: [divyanshu@tamu.edu]
