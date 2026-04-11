# skill_extraction.py

import pandas as pd
import re

# ✅ Load skills FROM the actual dataset
def load_skills_from_dataset():
    try:
        skills_df = pd.read_csv("data/processed_skills.csv")

        all_skills = set()

        # ✅ Loop through every row in the Skills column
        for text in skills_df['Skills'].dropna():
            text = text.lower()

            # Extract known technical skill patterns
            # Multi-word skills first
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

            # Single word skills using regex
            single_word_skills = re.findall(
                r'\b(python|java|sql|javascript|typescript|php|swift|'
                r'kotlin|scala|rust|golang|r|c\+\+|'
                r'tensorflow|keras|pytorch|sklearn|pandas|numpy|'
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
        # Fallback list if CSV not found
        return [
            "python", "java", "sql", "machine learning",
            "data analysis", "communication", "leadership"
        ]


# ✅ Load skills from dataset when file is imported
skills_list = load_skills_from_dataset()


# ✅ Extract skills from any text using dataset skills
def extract_skills(text):
    if not text:
        return []
    text = text.lower()
    return [skill for skill in skills_list if skill in text]