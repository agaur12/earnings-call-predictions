# Earnings calls and next-quarter EPS surprises

CS 74 final project (Applied ML track) by Aditya, AJ and Eric.

**Question.** Can the words in a company's earnings call predict whether its next-quarter EPS will beat or miss analyst expectations?

**Answer.** Only a little. Our best model (naive Bayes on TF-IDF features) ranks calls better than chance and catches 29 of the 121 misses in the test set, mostly by recognizing which industry a company is in.

## Files

| File | What it is |
|---|---|
| `project.ipynb` | **Start here.** The full analysis with explanations, tables and figures |
| `helpers.py` | Functions shared by the notebook and the demo |
| `app.py` | Streamlit demo: paste a transcript, see the prediction and the words behind it |
| `results/` | Tables and figures saved by the notebook |

## How to run

```bash
pip install -r requirements.txt
```

Then run all cells of `project.ipynb` which takes about 6 minutes. The data downloads automatically. The first run also scores every transcript chunk with FinBERT, which takes about 2 hours 45 minutes and is saved to `data/` for later runs.

```bash
streamlit run app.py
```

## Data split

2,686 earnings calls (2 duplicates removed). Label 1 is a beat and label 0 is a miss.

| Set | Year | Calls | Misses | Used for |
|---|---|---|---|---|
| Training | 2019 | 919 | 188 | fitting every model |
| Validation | 2020 | 704 | 95 | hyperparameter tuning and model selection |
| Test | 2021 | 1,063 | 121 | final evaluation, done once |

## Test-set results

AP (average precision) measures how well a model ranks calls by how likely they are to miss; random guessing scores 0.114. Precision is the share of flagged calls that were misses, and recall is the share of misses caught.

| Model | AP | Macro-F1 | Precision | Recall |
|---|---|---|---|---|
| Always predict a beat | 0.114 | 0.470 | 0.000 | 0.000 |
| TF-IDF + logistic regression | 0.172 | 0.526 | 0.172 | 0.455 |
| TF-IDF + linear SVC | 0.183 | 0.553 | 0.195 | 0.298 |
| **TF-IDF + naive Bayes** (best on validation) | **0.197** | 0.547 | 0.188 | 0.240 |
| FinBERT sentence embeddings | 0.168 | 0.547 | 0.186 | 0.281 |

## Data and licenses

FinCall-Surprise (Shu et al., ACL 2026), github.com/Tizzzzy/FinCall-Surprise, commit `36bb82d`, Apache 2.0. FinBERT is ProsusAI/finbert (revision `0bc2ac1`), CC BY-NC-SA 4.0; its weights download from Hugging Face and are not included.
