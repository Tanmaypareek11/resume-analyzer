# skill_extraction.py

import pandas as pd
import re
import os


# ── Comprehensive built-in skill list (always available as fallback) ──
FALLBACK_SKILLS = [
    # Programming languages
    "python", "java", "javascript", "typescript", "php", "swift", "kotlin",
    "scala", "rust", "golang", "r", "c", "c++", "c#", "ruby", "perl",
    # ML / AI
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "feature engineering", "model deployment",
    "transfer learning", "reinforcement learning", "supervised learning",
    "unsupervised learning", "random forest", "decision tree",
    "neural network", "support vector", "data science", "data analysis",
    "data engineering", "data visualization", "big data",
    # ML libraries
    "tensorflow", "keras", "pytorch", "pandas", "numpy", "matplotlib",
    "seaborn", "plotly", "scipy", "nltk", "spacy", "opencv",
    "xgboost", "lightgbm", "catboost", "scikit-learn", "hugging face",
    # Databases
    "mysql", "postgresql", "mongodb", "sqlite", "cassandra", "redis",
    "elasticsearch", "oracle",
    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "git", "github", "gitlab", "jenkins", "linux", "ci/cd", "airflow", "etl",
    # Web
    "html", "css", "react", "react js", "angular", "django", "flask",
    "fastapi", "node js", "next js", "rest api",
    # BI / Analytics
    "tableau", "power bi", "excel", "looker",
    # Data tools
    "hadoop", "spark", "statistics", "mathematics",
    # Soft skills
    "communication", "leadership", "management", "teamwork",
    "problem solving", "project management",
    # Agile
    "agile", "scrum",
    # Domain
    "accounting", "finance", "budgeting", "recruitment", "human resource",
    "hr", "training", "payroll", "marketing", "sales", "advertising",
    "branding", "seo", "social media", "public relations",
    "content marketing", "cybersecurity", "networking",
    "blockchain", "iot",
]

MULTI_WORD_SKILLS = [
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
    "node js", "react js", "next js",
]

SINGLE_WORD_PATTERN = re.compile(
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
    r'hadoop|spark|airflow|etl|statistics|mathematics)\b'
)


def load_skills_from_dataset():
    """Load skills from the processed_skills.csv if available, else use fallback."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "data", "processed_skills.csv")

        if not os.path.exists(csv_path):
            print("ℹ️ processed_skills.csv not found — using built-in skill list.")
            return list(FALLBACK_SKILLS)

        skills_df = pd.read_csv(csv_path)
        all_skills = set()

        # Support multiple possible column names
        text_col = None
        for col in ["Skills", "Resume", "skill", "skills"]:
            if col in skills_df.columns:
                text_col = col
                break

        if text_col is None:
            print("⚠️ No recognized skills column in CSV — using built-in skill list.")
            return list(FALLBACK_SKILLS)

        for text in skills_df[text_col].dropna():
            text = text.lower()

            for skill in MULTI_WORD_SKILLS:
                if skill in text:
                    all_skills.add(skill)

            matches = SINGLE_WORD_PATTERN.findall(text)
            all_skills.update(matches)

        if not all_skills:
            return list(FALLBACK_SKILLS)

        print(f"✅ Loaded {len(all_skills)} skills from dataset")
        return list(all_skills)

    except Exception as e:
        print(f"⚠️ Could not load skills dataset: {e} — using built-in skill list.")
        return list(FALLBACK_SKILLS)


# ── Load once when file is imported ──
skills_list = load_skills_from_dataset()


def extract_skills(text):
    """Extract known skills from text."""
    if not text or not isinstance(text, str):
        return []
    text = text.lower()
    return [skill for skill in skills_list if skill in text]
