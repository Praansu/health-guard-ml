# Data

- `heart.csv` — bundled 100-row, 4-feature sample of unknown provenance.
  Runs end to end in seconds. Good for a smoke test, not for conclusions.
- `cleveland.csv` — UCI Cleveland heart disease data, cleaned for this repo:
  303 rows → 297 after dropping 6 rows with missing `ca`/`thal` values,
  13 features, target binarized (0 = no disease, 1 = disease present).
  Source: https://archive.ics.uci.edu/dataset/45/heart+disease
  (file: `processed.cleveland.data`, fetched Oct 2026).

Regenerate `cleveland.csv` from the raw file if needed:

```python
import pandas as pd
cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
        "exang","oldpeak","slope","ca","thal","target"]
df = pd.read_csv("cleveland_raw.data", header=None, names=cols, na_values="?")
df["target"] = (df["target"] > 0).astype(int)
df.dropna().reset_index(drop=True).to_csv("cleveland.csv", index=False)
```
