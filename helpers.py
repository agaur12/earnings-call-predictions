"""Functions used by project.ipynb.

Loading the data, splitting transcripts into sections, text processing, and
building the TF-IDF models. All analysis lives in project.ipynb.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
RANDOM_SEED = 74

# The dataset is pinned to one commit so everyone gets the same files.
DATASET_REVISION = "36bb82d2298fe61de9d127c14ebda2c7be454136"
DATASET_URL = "https://raw.githubusercontent.com/Tizzzzy/FinCall-Surprise/%s/transcripts_%d.json"
TRAIN_YEAR, VALIDATION_YEAR, TEST_YEAR = 2019, 2020, 2021  # training, validation and test sets


# ---- Loading the data -----------------------------------------------------
def load_data() -> pd.DataFrame:
    """One row per call: year, call_id, text, label (1 = beat, 0 = miss) and word count."""
    frames = []
    for year in (TRAIN_YEAR, VALIDATION_YEAR, TEST_YEAR):
        path = DATA_DIR / "raw" / ("transcripts_%d.json" % year)
        if not path.exists():
            print("downloading", path.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(DATASET_URL % (DATASET_REVISION, year), timeout=300)
            response.raise_for_status()
            path.write_bytes(response.content)
        frame = pd.read_json(path, orient="index")
        frames.append(pd.DataFrame({
            "year": year,
            "call_id": frame.index.astype(str),
            "text": frame["input"].to_numpy(),
            "label": frame["label"].astype(int).to_numpy(),
        }))
    data = pd.concat(frames).sort_values(["year", "call_id"]).reset_index(drop=True)
    data["words"] = data["text"].str.split().str.len()

    # Two transcripts appear twice. Compare lowercased text with whitespace
    # collapsed, and keep the first call ID.
    normalized_text = data["text"].str.lower().str.split().str.join(" ")
    return data.loc[~normalized_text.duplicated()].reset_index(drop=True)


# ---- Transcript structure -------------------------------------------------
# Every speaker turn starts with a role tag. Tags are sometimes glued to the
# previous word ("...Giovanni.Executives:"), so there is no word boundary on the left.
ROLE_TAG = re.compile(r"(Executives|Analysts|Operator|Attendees|Shareholders):")


def speaker_turns(text: str) -> list[tuple[str, str]]:
    """Split a transcript into (role, text) speaker turns."""
    parts = ROLE_TAG.split(text)
    turns = [("Executives", parts[0])] + list(zip(parts[1::2], parts[2::2]))
    return [(role, turn.strip()) for role, turn in turns if turn.strip()]


def split_sections(text: str) -> tuple[list, list]:
    """Return (prepared remarks, Q&A) as lists of (role, text) turns.

    The Q&A starts at the first analyst turn. Operator turns are dropped because
    they are procedural ("our next question comes from...").
    """
    turns = [(role, turn) for role, turn in speaker_turns(text) if role != "Operator"]
    roles = [role for role, _ in turns]
    qa_start = next((i for i, role in enumerate(roles) if role in ("Analysts", "Attendees")), len(turns))
    return turns[:qa_start], turns[qa_start:]


def transcript_view(text: str, view: str) -> str:
    """The five parts of a transcript we compare as model input."""
    if view == "raw":  # as released, role tags and operator turns included
        return text
    prepared, qa = split_sections(text)
    chosen = {
        "clean": prepared + qa,  # role tags and operator turns removed
        "management": [turn for turn in prepared + qa if turn[0] == "Executives"],
        "prepared": prepared,
        "qa": qa,
    }[view]
    return " ".join(turn for _, turn in chosen)


# ---- Text processing and the name controls --------------------------------
NUMBER = re.compile(r"\b\d[\d,.%]*\b")
# Only catches names followed by a suffix such as "Inc." or "Corporation", so
# bare brand names ("MGM") survive.
COMPANY_NAME = re.compile(r"\b(?:[A-Z][\w&.-]*\s+){0,4}[A-Z][\w&.-]*(?:,)?\s+"
                          r"(?:Inc\.?|Corporation|Corp\.?|Limited|Ltd\.?|PLC|plc|LLC|Group|Holdings|Company)\b")
WEBSITE = re.compile(r"\b(?:www\.)?[\w-]+\.(?:com|net|org)\b", re.I)
# A capitalized word that does not start a sentence. Removing these drops most
# company, person, product and place names (and a few ordinary words such as "I").
MID_SENTENCE_CAPITAL = re.compile(r"(?<![.!?:] )(?<![.!?:])\b[A-Z][\w&'-]*")


def mask_numbers(text: str) -> str:
    return NUMBER.sub(" number ", text)


def mask_company_names(text: str) -> str:
    return WEBSITE.sub(" entity ", COMPANY_NAME.sub(" entity ", text))


def remove_capitalized_words(text: str) -> str:
    return MID_SENTENCE_CAPITAL.sub(" ", text)


def prepare_texts(texts, view, numbers):
    """Text processing before TF-IDF: pick a view of the transcript, then optionally mask numbers."""
    texts = [transcript_view(text, view) for text in texts]
    return [mask_numbers(text) for text in texts] if numbers == "masked" else texts


# ---- TF-IDF models --------------------------------------------------------
VIEWS = ["raw", "clean", "management", "prepared", "qa"]

# Each classifier and its hyperparameter grid. C is the inverse of the
# regularization strength lambda, so a larger C means weaker regularization.
# "balanced" class weights scale each class's errors in the cost by the inverse
# of its frequency.
CLASSIFIERS = {
    # L2-regularized logistic regression (sigmoid + cross-entropy cost)
    "logistic regression": (LogisticRegression(solver="liblinear", max_iter=2000, random_state=RANDOM_SEED),
                            {"C": [0.01, 0.1, 1, 10, 100, 1000], "class_weight": [None, "balanced"]}),
    # soft-margin support vector classifier with a linear kernel (hinge loss)
    "linear SVC": (LinearSVC(max_iter=20000, random_state=RANDOM_SEED),
                   {"C": [0.001, 0.01, 0.1, 1, 10], "class_weight": [None, "balanced"]}),
    # multinomial naive Bayes; alpha is the smoothing constant (alpha = 1 is Laplace smoothing)
    "naive Bayes": (MultinomialNB(), {"alpha": [0.001, 0.01, 0.1, 1.0]}),
}


def make_pipeline(setup, classifier):
    """Text processing -> TF-IDF feature vectors -> classifier.

    setup holds the text-processing choices: view, ngram (1 = unigrams, 2 = unigrams
    and bigrams), numbers ("kept" or "masked"), min_df and max_features. The
    vocabulary and IDF weights are learned from whatever text the pipeline is fit on.
    """
    return Pipeline([
        ("prepare", FunctionTransformer(prepare_texts, kw_args={"view": setup["view"], "numbers": setup["numbers"]})),
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, setup["ngram"]),
                                  min_df=setup["min_df"], max_df=0.95, max_features=setup["max_features"],
                                  sublinear_tf=True, dtype=np.float32)),
        ("classifier", classifier),
    ])


def setup_of(row):
    """The text-processing choices stored in one row of a results table."""
    return {"view": row["view"], "numbers": row["numbers"], "ngram": int(row["ngram"]), "min_df": int(row["min_df"]),
            "max_features": None if pd.isna(row["max_features"]) else int(row["max_features"])}


def fit_chosen(row, train):
    """Refit a chosen model on the training set. Fitting is deterministic, so this rebuilds the same model."""
    params = row["params"] if isinstance(row["params"], dict) else json.loads(row["params"])
    classifier = clone(CLASSIFIERS[row["classifier"]][0]).set_params(**params)
    return make_pipeline(setup_of(row), classifier).fit(train.text, train.label)


def positive_score(model, X):
    """Higher means more likely a beat: the predicted probability, or the SVC's margin."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def term_weights(classifier):
    """How much each vocabulary term pushes a prediction toward a beat (+) or a miss (-)."""
    if hasattr(classifier, "coef_"):
        return classifier.coef_[0]
    return classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]  # naive Bayes
