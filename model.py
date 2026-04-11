# model.py

import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def train_model():
    resume_df = pd.read_csv("data/processed_resumes.csv", low_memory=False)
    resume_df = resume_df.dropna(subset=['clean_resume', 'Category'])

    # ✅ Remove categories with very few samples
    category_counts = resume_df['Category'].value_counts()
    valid_categories = category_counts[category_counts >= 10].index
    resume_df = resume_df[resume_df['Category'].isin(valid_categories)]

    X = resume_df['clean_resume'].apply(clean_text)
    y = resume_df['Category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ✅ Improved Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,       # increased from 5000
            ngram_range=(1, 3),       # increased from (1,2)
            sublinear_tf=True,        # better TF scaling
            min_df=2,                 # ignore very rare words
            max_df=0.95               # ignore very common words
        )),
        ('clf', RandomForestClassifier(
            n_estimators=300,         # increased from 100
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',  # ✅ handles imbalanced data
            random_state=42,
            n_jobs=-1                 # use all CPU cores — faster
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"✅ Accuracy: {round(accuracy * 100, 2)}%")
    print(report)

    with open("resume_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("✅ Model saved!")
    return pipeline, accuracy


def load_model():
    with open("resume_model.pkl", "rb") as f:
        return pickle.load(f)


def predict_category(text, pipeline):
    cleaned = clean_text(text)
    category = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]
    confidence = round(max(proba) * 100, 2)
    return category, confidence


def find_matching_jobs(predicted_category, top_n=3):
    job_df = pd.read_csv("data/processed_jobs.csv", low_memory=False)
    job_df = job_df.dropna(subset=['clean_job'])

    if 'Job Title' in job_df.columns:
        matched = job_df[
            job_df['Job Title'].str.lower().str.contains(
                predicted_category.lower(), na=False
            )
        ].head(top_n)
    else:
        matched = job_df.head(top_n)

    return matched[['clean_job']].reset_index(drop=True)


# ✅ NEW FUNCTION — Check Accuracy
def evaluate_model():

    print("\n" + "="*50)
    print("        MODEL EVALUATION REPORT")
    print("="*50)

    # Load data
    resume_df = pd.read_csv("data/processed_resumes.csv", low_memory=False)
    resume_df = resume_df.dropna(subset=['clean_resume', 'Category'])

    X = resume_df['clean_resume'].apply(clean_text)
    y = resume_df['Category']

    # Same split as training — random_state=42 ensures same test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load saved model
    pipeline = load_model()

    # Predict
    y_pred = pipeline.predict(X_test)

    # ---- Accuracy ----
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy  : {round(accuracy * 100, 2)}%")
    print(f"📊 Total Test Data : {len(y_test)} resumes")
    print(f"✔️  Correct Predictions : {sum(y_test == y_pred)}")
    print(f"❌ Wrong Predictions   : {sum(y_test != y_pred)}")

    # ---- Classification Report ----
    print("\n📋 Classification Report:")
    print("-"*50)
    print(classification_report(y_test, y_pred))

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=pipeline.classes_,
        yticklabels=pipeline.classes_
    )
    plt.title("Confusion Matrix — Resume Category Prediction")
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
    print("\n✅ Confusion Matrix saved as confusion_matrix.png")
    print("="*50)


if __name__ == "__main__":
    train_model()       # Step 1 — Train and save model
    evaluate_model()    # Step 2 — Check accuracy