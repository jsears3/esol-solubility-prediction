# ESOL Aqueous Solubility Prediction

Predicting aqueous solubility from molecular structure using RDKit and scikit-learn — an exploratory starter project marking a process development chemist's first steps into applied machine learning.

## Problem

Can molecular structure alone predict how soluble a compound is in water? This project uses the ESOL (Delaney) dataset — 1,128 real compounds with measured aqueous solubility — as a standard cheminformatics benchmark to build, validate, and stress-test a regression model end to end.

## Approach

1. Parsed each compound's SMILES string into a molecular structure with RDKit and computed structural descriptors, starting with the Lipinski Rule-of-Five set (molecular weight, LogP, H-bond donors/acceptors, polar surface area).
2. Trained a Random Forest regressor against a linear regression baseline to test whether the structure–solubility relationship is meaningfully nonlinear.
3. Expanded to 10 descriptors (adding ring count, rotatable bonds, aromaticity, and more).
4. Ran a feature-by-feature ablation study to attribute performance gains correctly, rather than trusting built-in feature importance scores at face value.
5. Cross-validated (with explicit shuffling — the scikit-learn default does not shuffle) to get a trustworthy performance estimate instead of relying on a single train/test split.

## Key results

| Model | R² | RMSE |
| :--- | ---: | ---: |
| Linear regression (5 descriptors) | 0.740 | 1.109 |
| Random Forest (5 descriptors) | 0.851 | 0.838 |
| Random Forest (10 descriptors) | 0.867 | 0.794 |
| Random Forest (LogP + RingCount + MolWt only) | 0.856 | 0.826 |
| Random Forest, 5-fold CV (10 descriptors, shuffled) | 0.888 ± 0.015 | — |

- The Random Forest's clear edge over linear regression is evidence of real nonlinearity/interaction effects in the structure–solubility relationship.
- An ablation study showed **RingCount** — not the descriptor ranked highest by built-in feature importance — was actually carrying most of the gain from the expanded descriptor set. Feature importance measures credit *within* a model fit, not a feature's true standalone contribution.
- A compact 3-descriptor model (LogP + RingCount + MolWt) reaches within 0.011 R² of the full 10-descriptor model using well under a third of the inputs.
- Shuffled 5-fold cross-validation (mean R² = 0.888, std = 0.015) is the more defensible performance estimate than any single train/test split, and caught a real gotcha: `cross_val_score(cv=5)` does not shuffle by default.

## Tools

Python, RDKit (SMILES parsing, molecular descriptor generation), pandas, scikit-learn (Random Forest, linear regression, cross-validation), Jupyter, in a dedicated conda environment (`chem-ai`).

## Why it matters

The value here isn't the R² number — it's the validation discipline behind it: distinguishing correlation from causation in feature importance, catching a cross-validation shuffling gotcha that would have understated model reliability, and finding that a smaller, more interpretable model performs almost as well as a larger one.

## Repo contents

```
esol-solubility-prediction/
├── README.md
├── LICENSE
├── environment.yml
├── notebooks/
│   └── esol_solubility_model.ipynb   — the full modeling notebook
└── notes/
    ├── running-notes.md              — detailed log: code, results, concepts, lessons learned
    └── portfolio-writeup.md          — condensed portfolio summary
```

## Setup

```bash
conda env create -f environment.yml
conda activate chem-ai
jupyter notebook notebooks/esol_solubility_model.ipynb
```

## Status

Core modeling complete: baseline comparison, feature expansion, ablation study, and cross-validation are all done and documented. Possible extensions (not required): gradient boosting, molecular fingerprints, SHAP-based interpretability.

## License

MIT — see [LICENSE](LICENSE).

---
*Last updated: 2026-09-27*
