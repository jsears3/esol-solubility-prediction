# Portfolio Write-Up: Predicting Aqueous Solubility from Molecular Structure

*RDKit + scikit-learn — ESOL/Delaney solubility dataset*

## The problem
My first portfolio project (ChemCAD → Excel automation) demonstrated workflow
automation — moving data faster between existing systems. The next capability
gap was predictive modeling: could I actually build and validate a model from
data, not just automate a spreadsheet? I also wanted to be certain the
underlying coding skill was real rather than only AI-mediated, so this project
was run with every line written and executed by me, with Claude limited to
explaining concepts rather than producing finished code.

## The approach
Built a solubility-prediction model using the ESOL (Delaney) dataset — 1,128
real compounds with measured aqueous solubility, a standard cheminformatics
benchmark. Used RDKit to parse each compound's SMILES string into a molecular
structure and compute descriptors (starting with the Lipinski Rule-of-Five
set — molecular weight, LogP, H-bond donors/acceptors, polar surface area),
then trained a Random Forest against a linear regression baseline to test
whether the underlying structure–solubility relationship is meaningfully
nonlinear. From there the work went past a single model fit into genuine
model validation: expanding the descriptor set, running a feature-by-feature
ablation study to attribute performance gains correctly rather than trusting
feature-importance scores at face value, and cross-validating to get a
trustworthy performance estimate instead of relying on one train/test split.

## Key results
- Random Forest (5 Lipinski descriptors): R² = 0.851, RMSE = 0.838 — vs.
  linear regression's R² = 0.740, evidence the true structure–solubility
  relationship involves real nonlinearity/interactions a linear model can't
  capture.
- Expanding to 10 descriptors (adding ring count, rotatable bonds,
  aromaticity, and more) raised Random Forest performance to R² = 0.867.
- An ablation study — refitting with each new descriptor added individually —
  showed RingCount, not the descriptor built-in feature importance ranked
  highest, was actually carrying most of that gain: a concrete lesson that
  feature importance measures credit within one model fit, not a feature's
  true standalone contribution.
- A compact 3-descriptor model (LogP + RingCount + MolWt) reached R² = 0.856 —
  within 0.011 of the full 10-descriptor model, using well under a third of
  the inputs.
- 5-fold cross-validation (explicitly shuffled — the scikit-learn default
  does not shuffle) gave mean R² = 0.888, std = 0.015, the more defensible
  characterization of the model's real performance than any single split.

## Tools used
Python, RDKit (SMILES parsing, molecular descriptor generation), pandas,
scikit-learn (Random Forest and linear regression, cross-validation),
Jupyter, in a dedicated conda environment (`chem-ai`).

## Why it matters
The value here isn't the R² number — it's the validation discipline behind
it: distinguishing correlation from causation in feature importance, catching
a cross-validation shuffling gotcha that would have understated model
reliability, and finding that a smaller, more interpretable model performs
almost as well as a larger one. That's the same rigor process-development
work already requires — not mistaking a correlation for a mechanism, not
trusting one experimental run — applied to a modeling context, and it's the
direct bridge between existing chemistry domain expertise and applied ML.

## Skills demonstrated
Python/pandas data wrangling, RDKit cheminformatics (SMILES-to-structure
parsing, descriptor computation), scikit-learn regression modeling, and
applied ML methodology — ablation studies, cross-validation design, and
correctly interpreting (and not over-trusting) feature importance.

## Status
Core modeling complete — baseline comparison, feature expansion, ablation
study, and cross-validation all done and documented. Optional extensions
identified (gradient boosting, molecular fingerprints, SHAP-based
interpretability) but not required for this write-up.

---
*Last updated: 2026-09-25*
