
# Sentiment Analysis of IMDB Movie Reviews

A compact NLP pipeline that predicts **positive** or **negative** sentiment on the 50 000-row IMDB dataset.  
The notebook covers cleaning raw reviews, building TF-IDF features, selecting informative terms, training fast linear classifiers, and visualising results.

---

## Quick start

# 1. clone
git clone https://github.com/SibiVasudevan/movie-reviews-sentiment-analysis.git
cd movie-reviews-sentiment-analysis

# 2. create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. install the minimal dependencies
pip install pandas numpy scikit-learn nltk matplotlib jupyterlab

# 4. launch Jupyter
jupyter lab SentimentAnalysisOfMovieReviews.ipynb


Everything runs in under two minutes on a free-tier Colab or a mid-range laptop.

---

## Pipeline overview

| Stage              | Key steps                                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------------------------- |
| **Data loading**   | Pull `movie_data.csv` from the GitHub raw link                                                                   |
| **Pre-processing** | Strip HTML, keep emoticons, lower-case, Porter stemming, remove English stop-words                               |
| **Feature sets**   | 1. Full TF-IDF<br>2. Top 4 000 terms by Mutual Information (Information Gain)<br>3. Top 1 000 terms by χ²        |
| **Classifiers**    | Logistic Regression (saga), LinearSVC, SGDClassifier (“hinge” loss)                                              |
| **Evaluation**     | Accuracy and micro-F1 on a fixed 50 % hold-out, confusion matrix, bar chart of model scores, feature-weight plot |
| **Typical scores** | IG + LogReg → **0.890 acc / 0.890 F1**                                                                           |

---

## Visualisations

The notebook produces:

1. Confusion matrix for the best model
2. Accuracy vs F1 bar chart across the three linear models
3. Bar chart of the twenty most discriminative words

All plots use plain Matplotlib.

---

## Folder structure

```
SentimentAnalysisOfMovieReviews.ipynb   # main, cleaned notebook
README.md                               # this file
```

Add datasets or auxiliary scripts under `data/` or `src/` if needed.

---

## Reproducibility notes

* All estimators use `random_state=7`.
* The train-test split is fixed (`shuffle=False`).
* Library versions current as of mid-2025; pin them in a `requirements.txt` later if exact repeats are required.

---

## Contributing

1. Fork the repo and create a feature branch.
2. Keep commits focused and run basic linting before pushing.
3. Open a pull request for review.

---

## License

MIT License. The IMDB dataset is released for academic use; see the original ACL 2011 paper.
