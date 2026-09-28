# ESOL Solubility Project — Running Notes

## Goal
Predict a real molecular property (aqueous solubility) from molecular structure, using a
real, freely available dataset, as a first hands-on ML project — with the explicit aim of
understanding every line of code well enough to write it independently later, not just
running AI-generated code.

## Dataset
ESOL / Delaney solubility dataset, 1,128 compounds, loaded directly from
`https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv`.
Columns of note: `smiles` (structure), `measured log solubility in mols per litre`
(prediction target).

## Stage 1 — Data loading and descriptor generation
1. Loaded the CSV into a pandas DataFrame (`pd.read_csv`), previewed with `.head()`.
2. Used RDKit to convert each SMILES string into a molecule object (`Chem.MolFromSmiles`)
   and computed five structural descriptors per compound: MolWt, LogP, NumHDonors,
   NumHAcceptors, TPSA — the same set behind Lipinski's Rule of Five. Applied across the
   whole dataset and joined back onto the DataFrame.
   - RDKit *computes* these from molecular structure (the SMILES string) — they are not
     looked up from a table. LogP, for example, is a computed estimate of how strongly a
     molecule partitions between an oily (octanol) phase and water.

**Actual code pattern used:** a `compute_descriptors(smiles)` function that parses
`mol = Chem.MolFromSmiles(smiles)` once, then returns all descriptors together as a
single `pd.Series`, applied across the DataFrame via multi-column assignment:
`df[[...]] = df['smiles'].apply(compute_descriptors)`. There is no separate stored `mol`
column reused elsewhere — `mol` only exists inside that one function call.

## Stage 2 — First model: Random Forest
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

X = df[['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA']]
y = df['measured log solubility in mols per litre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"R²: {r2_score(y_test, predictions):.3f}")
print(f"RMSE: {root_mean_squared_error(y_test, predictions):.3f}")
```

**Result:** R² = 0.851, RMSE = 0.838 (log solubility units)

**Key concepts:**
- `X` (features) uses double-bracket selection (`df[[...]]`) to get a 2-D DataFrame;
  `y` (target) uses single-bracket (`df[...]`) to get a 1-D Series. Models expect `X` to
  be 2-D even with one feature.
- `train_test_split` holds out 20% of the data the model never trains on, so R²/RMSE
  measure genuine generalization rather than memorization. `random_state=42` seeds the
  split for reproducibility.
- Random Forest fits many decision trees (`n_estimators=100`) on random subsets of data
  and features, then averages their predictions — this averaging reduces the overfitting
  a single tree is prone to.
- Interpretation: the model explains ~85% of the variance in solubility, with a typical
  prediction error under 1 log unit — a solid first result given the target spans roughly
  -12 to +2 log units in this dataset.

**Gotcha hit:** `mean_squared_error(..., squared=False)` throws `TypeError` on recent
scikit-learn (the `squared` kwarg was removed; use `root_mean_squared_error` instead, or
`np.sqrt(mean_squared_error(...))`).

## Stage 3 — Linear regression baseline
```python
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)
print(f"Linear Regression R²: {r2_score(y_test, lr_predictions):.3f}")
print(f"Linear Regression RMSE: {root_mean_squared_error(y_test, lr_predictions):.3f}")
```

**Result:** R² = 0.740, RMSE = 1.109 — clearly worse than the Random Forest (0.851 / 0.838).

**Interpretation:** Linear regression assumes each descriptor's effect on solubility is a
straight, additive line. The gap versus Random Forest is evidence the true relationship
involves nonlinearity and/or interactions between descriptors (e.g. TPSA's effect likely
depends on molecule size too) — something only the tree-based model can capture. The
*comparison itself* is the useful result here, not just "RF wins."

## Stage 4 — Feature importance (Random Forest, 5 descriptors)
```python
importances = model.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance:.3f}")
```

**Clean result** (after a Jupyter reproducibility hiccup — see Lessons below):

| Feature | Importance |
| :--- | ---: |
| LogP | 0.824 |
| MolWt | 0.110 |
| TPSA | 0.046 |
| NumHAcceptors | 0.012 |
| NumHDonors | 0.008 |

**Interpretation:** LogP alone accounts for over 80% of the model's predictive weight.
This tracks with chemical intuition — LogP measures octanol/water partitioning, which is
essentially asking the same underlying question as aqueous solubility from the opposite
direction. MolWt and TPSA contribute modestly; H-bond donor/acceptor counts contribute
almost nothing in this model.

## Stage 5 — Expanded to 10 descriptors
Added five more RDKit descriptors to `compute_descriptors`: NumRotatableBonds, RingCount,
HeavyAtomCount, FractionCSP3, and AromaticProportion (the last computed manually — RDKit
has no single built-in function for it — as `aromatic_atom_count / heavy_atom_count`).

**Results (same 80/20 split, `random_state=42`):**

| Model | R² (5 features) | R² (10 features) | RMSE (5) | RMSE (10) |
| :--- | ---: | ---: | ---: | ---: |
| Random Forest | 0.851 | 0.867 | 0.838 | 0.794 |
| Linear Regression | 0.740 | 0.758 | 1.109 | 1.069 |

**Feature importance (Random Forest, 10 descriptors):**

| Feature | Importance |
| :--- | ---: |
| LogP | 0.808 |
| MolWt | 0.087 |
| TPSA | 0.035 |
| HeavyAtomCount | 0.018 |
| FractionCSP3 | 0.012 |
| AromaticProportion | 0.010 |
| NumHAcceptors | 0.009 |
| NumRotatableBonds | 0.008 |
| RingCount | 0.007 |
| NumHDonors | 0.006 |

**Interpretation:**
- Both models improved modestly and in the same direction (RF +0.016 R², LR +0.018 R²) —
  consistent, real signal from the new descriptors, not just noise.
- The five new descriptors combined account for only ~5.5% of total importance, roughly
  matching the modest R² gain — though this is a loose, informal observation, not a
  mechanical relationship (see "On feature importance vs. R²" below).
- LogP, MolWt, and TPSA importance all *dropped* slightly even though nothing changed
  about how they're computed — because `HeavyAtomCount` is correlated with `MolWt` (both
  are size proxies), so the Random Forest now splits "credit" for explaining size-related
  variance across two correlated columns instead of concentrating it in one. Feature
  importance measures credit *in this model*, not unique information content.
- `HeavyAtomCount` (0.018) is the standout among the five new descriptors — more than
  double any of the others.

**Rigorous next step identified (not yet done):** an ablation study — refit with each new
descriptor added/removed one at a time and measure the actual test-set R² change — would
give a defensible per-feature attribution, rather than reading it off feature importance.

## Stage 6 — Ablation study, feature combinations, and cross-validation

**Ablation study** — refit with each new descriptor added individually to the original
5-descriptor baseline (R² = 0.851, RMSE = 0.838), same split/seed throughout:

| Addition | R² | RMSE |
| :--- | ---: | ---: |
| +NumRotatableBonds | 0.855 | 0.829 |
| +HeavyAtomCount | 0.856 | 0.825 |
| +AromaticProportion | 0.859 | 0.815 |
| +FractionCSP3 | 0.861 | 0.811 |
| +RingCount | 0.865 | 0.798 |

**Key finding:** RingCount alone (6 features total) gets to within 0.002 R² / 0.004 RMSE
of the full 10-feature model (0.867/0.794) — it's carrying nearly all the benefit the five
new descriptors were collectively credited with. This directly contradicts the Stage 5
importance ranking, where HeavyAtomCount (0.018) outranked RingCount (0.007) by more than
double. Concrete evidence that feature importance measures credit *within a specific
model*, not standalone/unique contribution — RingCount's signal gets partly absorbed by
correlated features when all ten are present together, so it's under-credited there.

**Incremental feature-combination tests** — built up from the two strongest individual
contributors found above:

| Model | R² | RMSE |
| :--- | ---: | ---: |
| LogP only | 0.696 | 1.199 |
| LogP + RingCount | 0.781 | 1.016 |
| LogP + RingCount + MolWt | 0.856 | 0.826 |

**Key findings:**
- LogP alone (0.696) is *worse* than the 5-feature linear regression baseline (0.740) —
  despite carrying ~80% of feature importance in every Random Forest so far, importance
  dominance does not mean sufficiency. Part of this is also mechanical: with only one
  feature, Random Forest's per-split feature subsampling has nothing to subsample from,
  so all 100 trees just fit different thresholds on the same variable — the averaging
  benefit that makes a forest more than one tree is largely lost.
- Adding RingCount recovers +0.085 R² (0.696 → 0.781) — a much bigger jump than
  RingCount's tiny standalone importance score would suggest, more evidence of an
  interaction effect between LogP and RingCount rather than either carrying independent
  signal alone.
- LogP + RingCount + MolWt (3 features, R² = 0.856) slightly *beats* the original
  5-descriptor Lipinski baseline (0.851) and comes within 0.011 of the full 10-descriptor
  model (0.867) — using well under a third of the features. Suggests TPSA, NumHDonors,
  and NumHAcceptors (the three lowest-importance descriptors throughout) add very little
  beyond what these three already capture. A compact 3-descriptor model is nearly as good
  as the full 10-descriptor one for this dataset.

**Cross-validation** — run on the full 10-feature Random Forest, comparing default
(unshuffled) `cross_val_score(cv=5)` against an explicit shuffled `KFold`:

```python
from sklearn.model_selection import cross_val_score, KFold

cv_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Default: cv=5 as a bare integer does NOT shuffle -- takes contiguous blocks
cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring='r2')

# Explicit shuffled KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_shuffled = cross_val_score(cv_model, X, y, cv=kf, scoring='r2')
```

| | Per-fold R² | Mean | Std |
| :--- | :--- | ---: | ---: |
| Unshuffled (default) | 0.917, 0.899, 0.905, 0.879, 0.849 | 0.890 | 0.024 |
| Shuffled (`KFold(shuffle=True)`) | 0.865, 0.908, 0.891, 0.878, 0.900 | 0.888 | 0.015 |
| Shuffled, 3-feature (LogP + RingCount + MolWt) | 0.855, 0.892, 0.864, 0.858, 0.875 | 0.869 | 0.013 |

**Interpretation:**
- Mean R² (~0.888–0.890) is *higher* than the original single 80/20 split's 0.867 —
  that original split sat toward the low end of the fold range, so it modestly
  understated true model performance rather than overstating it.
- Gotcha caught: `cross_val_score(cv=5)` with a bare integer does not shuffle the data by
  default (unlike `train_test_split`, which does) — it slices the 1,128 rows into 5
  contiguous blocks in whatever order they're already in. Explicitly shuffling
  (`KFold(shuffle=True, random_state=42)`) tightened the std from 0.024 to 0.015,
  confirming some of the original fold-to-fold spread was an artifact of unshuffled,
  contiguous blocks rather than genuine model variability.
- Conclusion: mean R² ≈ 0.888, std ≈ 0.015 (shuffled) is the more trustworthy
  characterization of this model's performance than any single train/test split.
- The compact 3-feature model, cross-validated the same way (shuffled, same `KFold`
  splitter), scored mean R² = 0.869, std = 0.013 — a real but modest ~0.019 gap below
  the full 10-feature model's 0.888, and actually slightly *more* stable across folds
  (std 0.013 vs. 0.015). Confirms the compact-model finding from the single-split
  comparison holds up under more rigorous testing, not just as a fluke of one split.

## Concepts covered (Q&A, not yet acted on in code)

**How RDKit parses SMILES into a graph:** heavy atoms (as written in the SMILES) become
graph nodes; bonds become edges. Hydrogens implied by valence rules are *not* separate
nodes by default — each atom just carries an implicit-H count (`GetTotalNumHs()`).
`Chem.AddHs(mol)` rewrites the graph to add real H nodes/bonds, only needed for things
like 3D conformer generation — not for standard 2D descriptors like the ones used here.
A node, generally, is just "one thing" in a graph (CS term, not chemistry-specific) — an
atom object holding data (element, charge, aromaticity, implicit-H count, etc.), not a
literal drawn circle.

**Does traversal starting point matter?** For simple sum-based descriptors like MolWt, no
— addition is commutative/associative, so visiting atoms in any order gives the same
total. It matters more for other things: canonical SMILES generation needs a
deterministically chosen starting atom so the same molecule always serializes to the same
string; path-based descriptors (e.g. Wiener index) may traverse from every atom in turn.

**How a descriptor is computed, mechanically (using MolWt as the example):** walk every
atom node, look up its element's atomic weight from a reference table, add the mass of
its attached (mostly implicit) hydrogens, sum it all. Other descriptors follow the same
"walk the graph, sum contributions" pattern with different per-atom values/rules: LogP
(Crippen method) classifies atoms into ~68 types and sums pre-fitted contribution values;
TPSA sums pre-computed polar-surface-area values per N/O/S/P atom type; NumHDonors/
NumHAcceptors use SMARTS substructure pattern matching instead of summing.

**PEOE_VSA descriptors:** PEOE = Partial Equalization of Orbital Electronegativities, aka
Gasteiger charges — a fast, purely empirical (non-quantum-mechanical) way to estimate
atomic partial charges via iterative electronegativity equalization, distinct from DFT-
derived charges (Mulliken, ESP/RESP). `PEOE_VSA#` descriptors bin atoms by PEOE charge
range and sum each bin's van der Waals surface area contribution — a compact way to
describe how charge is distributed across a molecule's surface. Sibling families:
`SlogP_VSA#` (bins by LogP contribution), `SMR_VSA#` (bins by molar refractivity).

**How a decision tree picks a split, mechanically:** at each node, for every feature and
every candidate threshold (midpoints between consecutive sorted values in that node), the
tree computes the weighted-average MSE of the two resulting child groups and compares it
to the parent's MSE. Whichever (feature, threshold) pair gives the largest MSE reduction
becomes that node's actual split rule. Worked example with 4 toy compounds showed a
`LogP ≤ 2` split reducing MSE from 6.5 to 0.25 — illustrating mechanically why LogP wins
so many splits (and therefore dominates feature importance). Random Forest adds two more
sources of randomness beyond a single tree: each tree trains on a bootstrap sample of
rows, and each split only considers a random subset of features as candidates — this is
what decorrelates the trees and makes averaging them useful.

**On feature importance vs. R² (a correction made mid-session):** these are not directly
comparable/convertible quantities. Feature importance is computed purely from training
data (impurity reduction from splits, always summing to 1.0 by construction); R² is
computed purely from test-set predictions vs. actual values. Their rough numerical
similarity in Stage 5 (~5.5% importance vs. +1.6 pts R²) is not a mathematical
relationship — an ablation study would be the rigorous way to attribute R² gain to
specific features.

**Can a bad sample set make a regression meaningless?** Yes, several distinct ways: small
test-set size inflates run-to-run variance in metrics; label noise (measurement error in
the original experimental solubility values) puts a hard ceiling on achievable accuracy;
duplicate/near-duplicate compounds split across train/test can inflate R² via leakage
rather than real generalization; imbalanced target-range coverage can hide poor
performance at the extremes; and — most relevant professionally — a model's R²/RMSE only
describes performance *within the distribution it was trained on* ("applicability
domain" in QSAR terminology). ESOL is mostly drug-like small organics; a model trained on
it says nothing reliable about a structurally different space like semiconductor-grade
solvent candidates without new data from that space.

## Lessons learned
- **Same `random_state` does not guarantee an identical split across sessions/runs if the underlying data differs.** `train_test_split(..., random_state=42)` reproduces the same split only when splitting the same rows in the same order — if `df`'s row count or order changes between runs (e.g. a SMILES parse difference, re-loading the CSV), the same seed can produce a different split and a small metric shift (observed: baseline RMSE 0.838 vs. an earlier recollection of 0.841) without anything being "wrong."
- **Jupyter reproducibility gotcha:** got two different sets of feature importances from
  what looked like "the same cell." Root cause: `RandomForestRegressor(random_state=42)`
  is fully deterministic given identical training data — different output means
  `X_train`/`y_train` differed between runs, almost always from cells being re-run out of
  order or a downstream cell not re-run after an upstream one changed. Fix: **Kernel →
  Restart** then **Run All**, so every cell executes exactly once in dependency order,
  before trusting any result.
- **Saved notebook output ≠ live kernel state.** Reopening a notebook shows prior printed
  output, but variables are not restored — the kernel starts empty regardless. Verify
  with a quick `df` or `model` call (expect `NameError` on a fresh kernel); re-run cells
  from the top to actually repopulate state.
- **Closing down properly:** save + "Close and Halt" the notebook, then `Ctrl+C` in the
  terminal running Jupyter to shut down the server itself (closing the browser alone does
  not stop the server process or free the port).

## Reflection
This stage (ablation study, incremental feature combinations, and cross-validation) was a good hands-on exploration of model validation and stability across a dataset — moving past a single train/test split's R²/RMSE to actually test how trustworthy those numbers are, and where a model's predictive power really comes from versus what feature importance alone suggests.

## Next steps (not yet done)
- Eventually write this project up as a second portfolio piece, noting which parts were
  done unaided vs. with help
