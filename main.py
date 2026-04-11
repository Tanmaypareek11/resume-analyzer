import pandas as pd
from skill_extraction import extract_skills, skills_list
from matcher import skill_match, missing_skills

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# BERT
from sentence_transformers import SentenceTransformer


# -------------------------------
# Load data
# -------------------------------
resume_df = pd.read_csv("data/processed_resumes.csv", low_memory=False)
job_df = pd.read_csv("data/processed_jobs.csv", low_memory=False)

# -------------------------------
# Handle missing values
# -------------------------------
resume_df['clean_resume'] = resume_df['clean_resume'].fillna("")
job_df['clean_job'] = job_df['clean_job'].fillna("")

# -------------------------------
# Extract skills
# -------------------------------
resume_df['skills'] = resume_df['clean_resume'].apply(extract_skills)
job_df['skills'] = job_df['clean_job'].apply(extract_skills)

# -------------------------------
# Take one resume & one job
# -------------------------------
resume_skills = resume_df['skills'][0]
job_skills = job_df['skills'][0]

resume_text = resume_df['clean_resume'][0]
job_text = job_df['clean_job'][0]

# -------------------------------
# Skill Matching
# -------------------------------
score = skill_match(resume_skills, job_skills)
missing = missing_skills(resume_skills, job_skills)

print("\n===== SKILL MATCHING =====")
print("Resume Skills:", resume_skills)
print("Job Skills:", job_skills)
print("Match Score:", score, "%")
print("Missing Skills:", missing)


# -------------------------------
# TF-IDF Similarity
# -------------------------------
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])

tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
tfidf_score = round(tfidf_similarity[0][0] * 100, 2)

print("\n===== TF-IDF SIMILARITY =====")
print("TF-IDF Similarity:", tfidf_score, "%")


# -------------------------------
# BERT Similarity
# -------------------------------
print("\nLoading BERT model... (first time may take time)")

model = SentenceTransformer('all-MiniLM-L6-v2')

resume_embedding = model.encode(resume_text)
job_embedding = model.encode(job_text)

bert_similarity = cosine_similarity([resume_embedding], [job_embedding])
bert_score = round(bert_similarity[0][0] * 100, 2)

print("\n===== BERT SIMILARITY =====")
print("BERT Similarity:", bert_score, "%")


# -------------------------------
# Final Score
# -------------------------------
final_score = (0.3 * tfidf_score) + (0.3 * score) + (0.4 * bert_score)

print("\n===== FINAL RESULT =====")
print("Final Resume Score:", round(final_score, 2), "%")