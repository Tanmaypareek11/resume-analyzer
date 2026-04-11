# src/preprocessing.py

import pandas as pd
import re # regular expression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# -------------------------------
# Download required NLTK data
# -------------------------------
nltk.download('stopwords')
nltk.download('wordnet')  # For lemmatization

# -------------------------------
# Load datasets
# -------------------------------
resume_df = pd.read_csv("data/resume_dataset.csv")
job_df = pd.read_csv("data/job_description_dataset.csv")
skills_df = pd.read_csv("data/skills_dataset.csv")

# -------------------------------
# Remove duplicates & handle missing values
# -------------------------------
resume_df = resume_df.drop_duplicates().fillna('')
job_df = job_df.drop_duplicates().fillna('')
skills_df = skills_df.drop_duplicates().fillna('')

# -------------------------------
# Ensure correct column types
# -------------------------------
resume_df['Resume_str'] = resume_df['Resume_str'].astype(str)

# Fix job_df column names (strip spaces)
job_df.columns = [col.strip() for col in job_df.columns]
job_df['Job Description'] = job_df['Job Description'].astype(str)

# Use the 'resume' column in skills_df as skills list
skills_df['Resume'] = skills_df['Resume'].astype(str)
skills_list = skills_df['Resume'].str.lower().tolist()

# -------------------------------
# Stopwords & Lemmatizer
# -------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -------------------------------
# Text cleaning and tokenization
# -------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # remove emails
    text = re.sub(r'[^a-zA-Z ]', ' ', text)  # remove special chars & numbers

    # Regex tokenizer (no punkt needed)
    tokens = re.findall(r'\b[a-zA-Z]+\b', text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

# -------------------------------
# Clean text columns
# -------------------------------
resume_df['clean_resume'] = resume_df['Resume_str'].apply(clean_text)
job_df['clean_job'] = job_df['Job Description'].apply(clean_text)

# -------------------------------
# Tokenization (optional, for later matching)
# -------------------------------
resume_df['tokens'] = resume_df['clean_resume'].str.split()
job_df['tokens'] = job_df['clean_job'].str.split()

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
resume_vectors = vectorizer.fit_transform(resume_df['clean_resume'])
job_vectors = vectorizer.transform(job_df['clean_job'])

# -------------------------------
# Skill Extraction
# -------------------------------
def extract_skills(text):
    found_skills = []
    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)
    return found_skills

resume_df['extracted_skills'] = resume_df['clean_resume'].apply(extract_skills)
job_df['required_skills'] = job_df['clean_job'].apply(extract_skills)

# -------------------------------
# Save processed datasets
# -------------------------------
resume_df.to_csv("data/processed_resumes.csv", index=False)
job_df.to_csv("data/processed_jobs.csv", index=False)
skills_df.to_csv("data/processed_skills.csv", index=False)

print("✅ Preprocessing complete. CSVs saved in data/ folder.")