# Association Rule Mining on Healthcare Symptom Data
### Using FP-Growth Algorithm

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![mlxtend](https://img.shields.io/badge/mlxtend-0.23-orange?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat&logo=pandas&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> **Frequent Itemsets Assignment** — Discovering meaningful symptom co-occurrence patterns from 246,945 patient records using the FP-Growth algorithm.



## Overview

This project implements **Association Rule Mining (ARM)** on a large-scale binary healthcare dataset to automatically discover statistically significant symptom co-occurrence patterns.

**Why FP-Growth over Apriori?**
| | Apriori | FP-Growth |
|---|---|---|
| Candidate generation | Required (slow) | Not needed |
| Database scans | Multiple per level | 2 total |
| Memory usage | High | Efficient |
| Best for | Small datasets | Large sparse datasets |

FP-Growth was chosen because our dataset has 246,945 rows and 377 symptom columns — Apriori would be computationally infeasible at this scale.

---

## Dataset

| Property | Value |
|---|---|
| Total patient records | 246,945 |
| Total columns | 378 |
| Disease label column | `diseases` (773 unique values) |
| Symptom columns | 377 (binary: 1 = present, 0 = absent) |
| Avg symptoms per patient | 5.3 |
| Max symptoms per patient | 12 |
| Matrix density | ~1.4% (highly sparse) |

**Top 10 Most Frequent Symptoms**

| Rank | Symptom | Support |
|---|---|---|
| 1 | Sharp abdominal pain | 13.08% |
| 2 | Vomiting | 11.29% |
| 3 | Headache | 10.01% |
| 4 | Cough | 9.84% |
| 5 | Sharp chest pain | 9.73% |
| 6 | Nausea | 9.59% |
| 7 | Back pain | 8.83% |
| 8 | Shortness of breath | 8.64% |
| 9 | Fever | 8.26% |
| 10 | Dizziness | 6.99% |

---

## Results

### Summary Statistics

| Metric | Value |
|---|---|
| Symptoms after preprocessing | 149 / 377 |
| Frequent itemsets found | 358 |
| 1-item | 149 |
| 2-item (pairs) | 204 |
| 3-item (triplets) | 5 |
| **Valid association rules** | **214** |
| Rules passing all filters | 214 / 214 (100%) |
| Average support | 0.0149 (1.49%) |
| Average confidence | 0.400 (40.0%) |
| **Average lift** | **7.90** |
| **Maximum lift** | **15.43** |
| All Zhang metric values | > 0.94 |

### FP-Growth Parameters

```python
MIN_SUPPORT        = 0.01   # 1% at least 2,469 patients
MIN_CONFIDENCE     = 0.30   # 30%
MIN_LIFT           = 1.20
MAX_ITEMSET_LENGTH = 3
MIN_SUPPORT_FILTER = 0.01   # symptom selection threshold
MAX_SUPPORT_FILTER = 0.95   # remove near-constant symptoms
```

---

## Top Association Rules

| # | IF (Antecedent) | THEN (Consequent) | Support | Confidence | Lift | Zhang |
|---|---|---|---|---|---|---|
| 1 | lacrimation | itchiness of eye | 1.28% | 43.1% | **15.43** | 0.964 |
| 2 | itchiness of eye | lacrimation | 1.28% | 45.8% | **15.43** | 0.962 |
| 3 | foreign body sensation in eye | symptoms of eye | 1.15% | 51.6% | 15.21 | 0.956 |
| 4 | symptoms of eye | foreign body sensation in eye | 1.15% | 33.9% | 15.21 | 0.967 |
| 5 | blood in urine | frequent urination | 1.17% | 43.5% | 14.94 | 0.959 |
| 6 | frequent urination | blood in urine | 1.17% | 40.0% | 14.94 | 0.961 |
| 7 | diminished vision | foreign body sensation in eye | 1.16% | 32.6% | 14.63 | 0.966 |
| 8 | foreign body sensation in eye | diminished vision | 1.16% | 52.0% | 14.63 | 0.953 |
| 9 | eye redness | lacrimation | 1.03% | 41.7% | 14.07 | 0.952 |
| 10 | lacrimation | eye redness | 1.03% | 34.6% | 14.07 | 0.957 |

> **How to read Rule 1:** When a patient has *lacrimation* (excessive tearing), there is a **43.1% chance** they also have *itchiness of eye*. This co-occurrence is **15.43× stronger** than random chance.

---

## Clinical Clusters Discovered

FP-Growth automatically grouped symptoms into three coherent clinical clusters — purely from co-occurrence patterns, without using any disease labels.

### Cluster 1 — Ocular (Eye) Diseases
`lacrimation` · `itchiness of eye` · `eye redness` · `foreign body sensation in eye` · `diminished vision` · `pain in eye`

| Rule | Lift | Clinical Meaning |
|---|---|---|
| lacrimation ↔ itchiness of eye | 15.43 | Classic allergic conjunctivitis — histamine causes both tearing and itching simultaneously |
| foreign body sensation ↔ diminished vision | 14.63 | Hallmark of corneal conditions (keratitis, corneal abrasion) |
| eye redness ↔ lacrimation | 14.07 | Key signs of conjunctivitis and dry eye syndrome |

### Cluster 2 — Urinary Tract Conditions
| Rule | Lift | Clinical Meaning |
|---|---|---|
| blood in urine ↔ frequent urination | 14.94 | Classic UTI / kidney stone / bladder pathology presentation |

### Cluster 3 — Psychiatric Disorders
| Rule | Lift | Clinical Meaning |
|---|---|---|
| excessive anger ↔ delusions or hallucinations | 13.27 | Characteristic of acute psychosis, bipolar mania, schizophrenia — aligns with DSM-5 criteria |

---

## Pipeline

```
Raw CSV (246,945 × 378)
        │
        ▼
┌─────────────────────┐
│  Cell 1: Load &     │  Shape check, symptom frequency,
│  Explore            │  support distribution analysis
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Cell 2:            │  Remove rare (<1%) and common (>95%)
│  Preprocessing      │  symptoms, drop duplicates, convert bool
└────────┬────────────┘
         │  377 → 149 symptoms
         ▼
┌─────────────────────┐
│  Cell 3:            │  min_support=0.01, max_len=3
│  FP-Growth          │  → 358 frequent itemsets
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Cell 4:            │  confidence ≥ 0.30, lift ≥ 1.20
│  Association Rules  │  Zhang's metric > 0
└────────┬────────────┘
         │  → 214 valid rules
         ▼
┌─────────────────────┐
│  Cell 5:            │  Auto-select top diseases,
│  Per-Disease Mining │  disease-specific symptom rules
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Cell 6:            │  Scatter plot, co-occurrence heatmap,
│  Visualise & Export │  bar chart, rule narratives, CSV export
└─────────────────────┘
```

---

## Metrics Reference

| Metric | Formula | Threshold Used | Meaning |
|---|---|---|---|
| **Support** | P(A ∪ B) | ≥ 0.01 | How often the pattern appears in all patients |
| **Confidence** | P(B\|A) | ≥ 0.30 | Given A, how often does B appear |
| **Lift** | Conf / P(B) | ≥ 1.20 | Strength vs random chance. >10 = very strong |
| **Conviction** | (1−P(B)) / (1−Conf) | — | Directional dependency strength |
| **Zhang's Metric** | Range: −1 to +1 | > 0 | Confirms genuine positive correlation |

---

## Team

| Name                           | NRP        |
| ------------------------------ | ---------- |
| Syarif Sanad                   | 5025221257 |
| Mirza Syahrizal Fathir         | 5025231151 |
| Aditya Fieansyah Putra Pratama | 5025231309 |
| Felda Ega Fadhila              | 5025231199 |

---

