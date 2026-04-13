# model.py

import pandas as pd
import re
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def train_model():
    resume_df = pd.read_csv("data/processed_resumes.csv", low_memory=True)
    resume_df = resume_df.dropna(subset=['clean_resume', 'Category'])

    # Remove categories with very few samples
    category_counts = resume_df['Category'].value_counts()
    valid_categories = category_counts[category_counts >= 10].index
    resume_df = resume_df[resume_df['Category'].isin(valid_categories)]

    X = resume_df['clean_resume'].apply(clean_text)
    y = resume_df['Category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ✅ LogisticRegression instead of RandomForest
    # — uses ~10x less RAM, trains faster, accuracy is comparable for text
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            solver='saga',       # best solver for large sparse TF-IDF
            C=1.0,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {round(accuracy * 100, 2)}%")
    print(classification_report(y_test, y_pred))

    with open("resume_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("✅ Model saved!")
    return pipeline, accuracy


def load_model():
    with open("resume_model.pkl", "rb") as f:
        return pickle.load(f)


def predict_category(text, pipeline):
    """Predict resume category and return (category, confidence%)."""
    try:
        cleaned = clean_text(text)
        category = pipeline.predict([cleaned])[0]
        proba = pipeline.predict_proba([cleaned])[0]
        confidence = round(max(proba) * 100, 2)
        return category, confidence
    except Exception as e:
        print(f"⚠️ predict_category error: {e}")
        return "Unknown", 0.0


def find_matching_jobs(predicted_category, top_n=3):
    """
    Find job descriptions matching predicted category.
    Uses chunked reading to avoid loading the entire CSV into RAM.
    """
    try:
        results = []
        chunk_size = 500  # read 500 rows at a time

        for chunk in pd.read_csv(
            "data/processed_jobs.csv",
            chunksize=chunk_size,
            low_memory=True,
            usecols=lambda c: c in ['clean_job', 'Job Title', 'title', 'Category']
        ):
            chunk = chunk.dropna(subset=['clean_job'])

            # Try to match by available title/category column
            if 'Job Title' in chunk.columns:
                matched = chunk[
                    chunk['Job Title'].str.lower().str.contains(
                        predicted_category.lower(), na=False
                    )
                ]
            elif 'title' in chunk.columns:
                matched = chunk[
                    chunk['title'].str.lower().str.contains(
                        predicted_category.lower(), na=False
                    )
                ]
            elif 'Category' in chunk.columns:
                matched = chunk[
                    chunk['Category'].str.lower().str.contains(
                        predicted_category.lower(), na=False
                    )
                ]
            else:
                matched = chunk

            results.append(matched[['clean_job']])

            # Stop early once we have enough rows
            combined = pd.concat(results, ignore_index=True)
            if len(combined) >= top_n:
                return combined.head(top_n)

        # If nothing matched, just return first top_n rows from file
        if not results or pd.concat(results, ignore_index=True).empty:
            fallback = []
            for chunk in pd.read_csv(
                "data/processed_jobs.csv",
                chunksize=chunk_size,
                low_memory=True,
                usecols=lambda c: c in ['clean_job']
            ):
                chunk = chunk.dropna(subset=['clean_job'])
                fallback.append(chunk[['clean_job']])
                if sum(len(f) for f in fallback) >= top_n:
                    break
            return pd.concat(fallback, ignore_index=True).head(top_n)

        return pd.concat(results, ignore_index=True).head(top_n)

    except Exception as e:
        print(f"⚠️ find_matching_jobs error: {e}")
        return pd.DataFrame(columns=['clean_job'])


def evaluate_model():
    """Evaluate saved model and print classification report."""
    print("\n" + "="*50)
    print("        MODEL EVALUATION REPORT")
    print("="*50)

    resume_df = pd.read_csv("data/processed_resumes.csv", low_memory=True)
    resume_df = resume_df.dropna(subset=['clean_resume', 'Category'])

    X = resume_df['clean_resume'].apply(clean_text)
    y = resume_df['Category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = load_model()
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy  : {round(accuracy * 100, 2)}%")
    print(f"📊 Total Test Data : {len(y_test)} resumes")
    print(f"✔️  Correct        : {sum(y_test == y_pred)}")
    print(f"❌ Wrong           : {sum(y_test != y_pred)}")
    print("\n📋 Classification Report:")
    print("-"*50)
    print(classification_report(y_test, y_pred))
    print("="*50)


if __name__ == "__main__":
    train_model()
    evaluate_model()
