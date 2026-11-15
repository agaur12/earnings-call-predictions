"""Streamlit demo: paste a transcript, get a prediction and the words behind it.

Run: streamlit run app.py   (after project.ipynb has written results/)

The demo refits the chosen model on the training set (2019 calls), using the
hyperparameters and the decision threshold (tuned on the 2020 validation set)
that project.ipynb recorded in results/test_results.csv. Fitting is
deterministic, so this is the same model the report evaluates.
"""
import numpy as np
import pandas as pd
import streamlit as st

from helpers import RESULTS_DIR, TRAIN_YEAR, fit_chosen, load_data, positive_score, term_weights

st.set_page_config(page_title="Earnings Call Surprise", layout="wide")
st.title("Earnings call → next-quarter EPS surprise")
st.caption("CS 74 project (Aditya, AJ, Eric). Trained on the 2019 training set, hyperparameters tuned on the 2020 "
           "validation set, evaluated on the 2021 test set. Scores are not calibrated probabilities. This is a class "
           "project, not investment advice.")

if not (RESULTS_DIR / "test_results.csv").exists():
    st.error("No results yet. Run project.ipynb first.")
    st.stop()


@st.cache_resource(show_spinner="Fitting the model on the training set (about 20 seconds)...")
def load_model(name):
    row = pd.read_csv(RESULTS_DIR / "test_results.csv").set_index("method").loc["TF-IDF + " + name]
    data = load_data()
    return fit_chosen(row, data[data.year == TRAIN_YEAR]), row


try_tab, results_tab = st.tabs(["Try a transcript", "Results"])

with try_tab:
    name = st.selectbox("Model", ["naive Bayes", "logistic regression", "linear SVC"])
    pipeline, row = load_model(name)
    st.caption("Chosen on the validation set: %s transcript, %s, numbers %s, decision threshold %.3f."
               % (row["view"], "unigrams" if row["ngram"] == 1 else "unigrams and bigrams",
                  row["numbers"], row["threshold"]))
    text = st.text_area("Paste an earnings call transcript", height=250)
    if st.button("Predict", type="primary"):
        if len(text.split()) < 100:
            st.warning("Paste at least 100 words. Real calls in the dataset run about 9,000.")
        else:
            left, right = st.columns(2)
            score = positive_score(pipeline, [text])[0]
            beat = score >= row["threshold"]
            left.metric("Prediction", "Beat (positive surprise)" if beat else "Miss (negative surprise)")
            right.metric("Score (below the decision threshold means miss)", "%.3f" % score)

            x = pipeline[:-1].transform([text])  # text processing + TF-IDF
            contributions = x.multiply(term_weights(pipeline.named_steps["classifier"])).toarray()[0]
            biggest = np.argsort(-np.abs(contributions))[:15]
            biggest = biggest[contributions[biggest] != 0]
            terms = pipeline.named_steps["tfidf"].get_feature_names_out()[biggest]
            st.subheader("Terms that moved the score most")
            st.bar_chart(pd.DataFrame({"term": terms, "push toward beat (+) / miss (-)": contributions[biggest]})
                         .set_index("term"))

with results_tab:
    columns = ["method", "test_accuracy", "test_macro_f1", "test_balanced_accuracy", "test_ap_negative",
               "test_recall_negative"]
    st.dataframe(pd.read_csv(RESULTS_DIR / "test_results.csv")[columns].round(3), hide_index=True)
    st.caption("All metrics are on the 2021 test set. AP negative = average precision (area under the "
               "precision-recall curve) for misses; chance = 0.114. Recall negative = share of misses caught.")
    for figure in ("fig_test.png", "fig_text_processing.png", "fig_data.png"):
        if (RESULTS_DIR / figure).exists():
            st.image(str(RESULTS_DIR / figure))
