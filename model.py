# model.py

import pandas as pd
import re
import pickle
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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

    category_counts = resume_df['Category'].value_counts()
    valid_categories = category_counts[category_counts >= 10].index
    resume_df = resume_df[resume_df['Category'].isin(valid_categories)]

    X = resume_df['clean_resume'].apply(clean_text)
    y = resume_df['Category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95
        )),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=40,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
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


def predict_category(resume_text, job_text, pipeline):
    """
    Predicts resume category AND computes how well the job description
    matches that category — giving a meaningful, variable confidence score.

    Returns: (predicted_category, resume_confidence, category_match_score)
    - resume_confidence : how sure the model is about the resume category (fixed per resume)
    - category_match_score : how well the JOB matches the predicted category (varies with job)
    """
    try:
        resume_cleaned = clean_text(resume_text)
        job_cleaned = clean_text(job_text)

        # Predict category from resume
        resume_category = pipeline.predict([resume_cleaned])[0]
        resume_proba = pipeline.predict_proba([resume_cleaned])[0]
        resume_confidence = round(max(resume_proba) * 100, 2)

        # ✅ KEY FIX: Also predict what category the JOB belongs to
        job_proba = pipeline.predict_proba([job_cleaned])[0]
        classes = pipeline.classes_

        # Find probability the job assigns to the same category as resume
        if resume_category in classes:
            idx = list(classes).index(resume_category)
            job_category_prob = job_proba[idx]
        else:
            job_category_prob = 0.0

        # ✅ Scale job category match to a meaningful 0-100 range
        # multiply by number of classes to normalize (since prob is split across N classes)
        n_classes = len(classes)
        # If job perfectly matches resume category: job_category_prob = 1/n * dominance
        # Scale so that average match = ~50, perfect match = ~90+
        category_match_score = round(
            min(job_category_prob * n_classes * 30, 95.0), 2
        )

        return resume_category, resume_confidence, category_match_score

    except Exception as e:
        print(f"⚠️ predict_category error: {e}")
        return "Unknown", 0.0, 0.0


def find_matching_jobs(predicted_category, top_n=3):
    """Find job descriptions matching the predicted category."""
    try:
        try:
            job_df = pd.read_csv(
                "data/processed_jobs.csv",
                low_memory=True,
                usecols=lambda c: c.strip() in [
                    'clean_job', 'Job Title', 'title', 'Category', 'category'
                ]
            )
        except Exception:
            job_df = pd.read_csv("data/processed_jobs.csv", low_memory=True)

        job_df = job_df.dropna(subset=['clean_job'])
        job_df.columns = [c.strip().lower() for c in job_df.columns]

        category_lower = predicted_category.lower()
        matched = pd.DataFrame()

        for col in ['job title', 'title', 'category']:
            if col in job_df.columns:
                mask = job_df[col].astype(str).str.lower().str.contains(
                    category_lower, na=False
                )
                matched = job_df[mask][['clean_job']]
                if not matched.empty:
                    break

        if matched.empty:
            keyword = category_lower.split()[0] if category_lower else ""
            if keyword and len(keyword) > 3:
                mask = job_df['clean_job'].str.lower().str.contains(keyword, na=False)
                matched = job_df[mask][['clean_job']]

        if matched.empty:
            matched = job_df[['clean_job']].sample(
                n=min(top_n * 5, len(job_df)), random_state=None
            )

        matched = matched.sample(frac=1, random_state=None).reset_index(drop=True)
        return matched.head(top_n)

    except Exception as e:
        print(f"⚠️ find_matching_jobs error: {e}")
        return pd.DataFrame(columns=['clean_job'])


def evaluate_model():
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
