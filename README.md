# ML Assignment 2: Bagging, SAMME Boosting and Bias-Variance on Olivetti Faces

Name: Deep Das  
Roll No.: m26ma004  
Seed: 26004 (numeric characters of the roll number, used with `numpy.random.default_rng`)

## What is in this repository

| File / folder | Purpose |
|---|---|
| `data_setup.ipynb` | Loads Olivetti Faces, makes the 6/2/2 split per identity, builds the masked test set, saves `dat_split.npz` |
| `dat_split.npz` | The saved split (train / val / clean test / masked test). Used by every notebook |
| `q1_solved.ipynb` | Q1: single tree grid, own bagging, OOB / disagreement / margin, clean vs masked |
| `q2_solved.ipynb` | Q2: toy SAMME round, SAMME on faces, boosting vs bagging, clean vs masked |
| `q3_solved.ipynb` | Q3: 30 resamples, measured bias and variance, depth study |
| `config.json` | All settings used (seed, grids, frozen choices, B, boosting depth, Q3 setup) |
| `logs/` | Final logs: printed output of every notebook cell (`<notebook>_log.txt`) |
| `figures/` | Figures used in the report, plus every figure the notebooks draw |
| `figure_scripts/` | The plotting code of each figure cell, copied from the notebooks |
| `tools/extract_outputs.py` | Rebuilds `logs/`, `figures/` and `figure_scripts/` from the executed notebooks |
| `report/` | CVPR two-column report (`main.tex`, `cvpr.sty`, `main.pdf`) |
| `requirements.txt` | Python packages |
| `run_all.sh` | Runs everything in order |

## Environment

Python 3.12 (developed on 3.12.3).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the Olivetti Faces dataset through scikit-learn, so it needs internet access.

## Exact reproduction commands

```bash
# 1. split and masked test set
jupyter nbconvert --to notebook --execute --inplace data_setup.ipynb

# 2. experiments (run in this order)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 q1_solved.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 q2_solved.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 q3_solved.ipynb

# 3. logs, figures, figure scripts
python tools/extract_outputs.py

# 4. report
cd report && pdflatex main.tex && pdflatex main.tex
```

Or simply: `bash run_all.sh`

Approximate run time: Q1 a few minutes, Q2 about 6 minutes, Q3 about 30 minutes or more, depending on the machine.

To pin the exact package versions used, run `pip freeze > requirements-lock.txt` in the same environment and commit that file.

## Notes on reproducibility

- The split, the bagging ensemble used from Q1(c) onwards, and the boosting trees are seeded.
- The Q1(b) bagging table and the 30 resamples in Q3 use the global NumPy generator without a seed, so rerunning them gives slightly different numbers (see "Deviations and limitations" in the report).
- Boosting in Q2(c), Q2(d) and Q3 uses depth-3 weak learners; the selected model in Q2(b) is depth 4. This is also listed in the report.
