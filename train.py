import os
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import csv


DATA_PATH = os.environ.get("DATA_CSV", "data/sms_spam_sample.csv")
MODEL_DIR = pathlib.Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "spam_model.joblib"

print(f"[train] loading data from: {DATA_PATH}")
DATA_PATH = "data/sms_spam_full.csv"

print(f"[train] loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, encoding="latin-1")
assert {"label", "text"}.issubset(df.columns), "CSV must have 'label' and 'text' columns"

# Keep labels as strings ('ham'/'spam') so the classifier exposes class names directly
X = df["text"].astype(str)
y = df["label"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )),
    ("clf", LogisticRegression(max_iter=1000, n_jobs=None))
])

print("[train] fitting model...")
pipeline.fit(X_train, y_train)

print("[train] evaluating...")
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[train] accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

print(f"[train] saving model to: {MODEL_PATH}")
joblib.dump(pipeline, MODEL_PATH)
print("[train] done ✅")