# skill_extraction.py

import pandas as pd
import re
import os

def load_skills_from_dataset():
    try:
        # ✅ Works on both local and Streamlit Cloud
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "data", "processed_skills.csv")

        skills_df = pd.read_csv(csv_path)
        all_skills = set()

        for text in skills_df['Skills'].dropna():
            text = text.lower()

            multi_word_skills = [
                "machine learning", "deep learning", "natural language processing",
                "data analysis", "data science", "data engineering",
                "computer vision", "feature engineering", "model deployment",
                "transfer learning", "reinforcement learning",
                "supervised learning", "unsupervised learning",
                "natural language", "random forest", "decision tree",
                "neural network", "support vector", "big data",
                "data visualization", "power bi", "google cloud",
                "rest api", "ci/cd", "social media", "public relations",
                "content marketing", "human resource", "project management",
                "problem solving", "scikit-learn", "hugging face",
                "node js", "react js", "next js"
            ]

            for skill in multi_word_skills:
                if skill in text:
                    all_skills.add(skill)

            single_word_skills = re.findall(
                r'\b(python|java|javascript|typescript|php|swift|'
                r'kotlin|scala|rust|golang|r|'
                r'tensorflow|keras|pytorch|pandas|numpy|'
                r'matplotlib|seaborn|plotly|scipy|nltk|spacy|opencv|'
                r'xgboost|lightgbm|catboost|'
                r'mysql|postgresql|mongodb|sqlite|cassandra|redis|'
                r'elasticsearch|oracle|'
                r'aws|azure|gcp|docker|kubernetes|git|github|'
                r'gitlab|jenkins|linux|'
                r'html|css|react|angular|django|flask|fastapi|'
                r'tableau|excel|looker|'
                r'communication|leadership|management|teamwork|'
                r'agile|scrum|'
                r'accounting|finance|budgeting|recruitment|'
                r'hr|training|payroll|'
                r'marketing|sales|advertising|branding|seo|'
                r'cybersecurity|networking|blockchain|iot|'
                r'hadoop|spark|airflow|etl|statistics|mathematics)\b',
                text
            )
            all_skills.update(single_word_skills)

        print(f"✅ Loaded {len(all_skills)} skills from dataset")
        return list(all_skills)

    except Exception as e:
        print(f"⚠️ Could not load skills dataset: {e}")
        # ✅ Fallback hardcoded list if CSV not found
        return [
            "python", "java", "sql", "machine learning", "deep learning",
            "data analysis", "communication", "leadership", "marketing",
            "excel", "nlp", "html", "css", "javascript", "management",
            "accounting", "finance", "recruitment", "human resource",
            "hr", "training", "budgeting", "advertising", "social media",
            "tensorflow", "keras", "pytorch", "scikit-learn", "pandas",
            "numpy", "tableau", "power bi", "docker", "git"
        ]


# ✅ Load once when file is imported
skills_list = load_skills_from_dataset()


def extract_skills(text):
    if not text:
        return []
    text = text.lower()
    return [skill for skill in skills_list if skill in text]
