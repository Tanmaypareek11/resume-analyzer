# 🧠 ResumeIQ — AI Job Match System

An AI-powered resume analyzer that matches your resume against any job description and gives you an instant score with skill gap analysis.

---

## 🚀 Live Demo

> Deploy on [Streamlit Cloud](https://streamlit.io/cloud) — free tier supported.
[> https://resume-analyzer-jxhdzxuunj3f4fec3bcdgd.streamlit.app/
---

## ✨ Features

- 📄 **PDF Resume Parsing** — extracts text directly from your resume
- 🎯 **Skill Match Analysis** — identifies matched and missing skills
- 🤖 **ML Job Category Prediction** — classifies your resume using a trained RandomForest model
- 📊 **Multi-Signal Scoring** — combines 4 independent signals for a reliable final score
- 💼 **Similar Job Suggestions** — shows relevant job descriptions from the dataset
- ⚡ **Optimized for Streamlit Cloud** — runs within the free 1GB RAM limit

---

## 📐 How Scoring Works

| Signal | Weight | Description |
|---|---|---|
| 🎯 Skill Match | 30% | % of job-required skills found in resume |
| 🔍 Keyword Overlap | 25% | % of job keywords present in resume |
| 🧬 Semantic Similarity | 20% | TF-IDF word + character n-gram similarity |
| 📈 ML Confidence | 15% | Model confidence in resume category prediction |
| 📝 TF-IDF Score | 10% | Direct text cosine similarity |

---

## 🗂️ Project Structure

```
├── app.py                  # Streamlit UI and main app logic
├── model.py                # ML model training, prediction, job matching
├── skill_extraction.py     # Skill detection from text
├── matcher.py              # Skill match and missing skill logic
├── load_data.py            # Downloads datasets from Google Drive
├── preprocessing.py        # One-time data cleaning (run locally)
├── main.py                 # Standalone CLI test script
├── requirements.txt        # Python dependencies
└── data/
    ├── processed_resumes.csv   # Downloaded from Google Drive at runtime
    ├── processed_jobs.csv      # Downloaded from Google Drive at runtime
    └── processed_skills.csv    # Committed directly to repo (small file)
```

---

## ⚙️ Setup & Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/your-username/resumeiq.git
cd resumeiq
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Google Drive File IDs in `load_data.py`**
```python
RESUMES_FILE_ID = "your_resumes_file_id"
JOBS_FILE_ID    = "your_jobs_file_id"
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Make sure `data/processed_skills.csv` is committed
3. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select your repo
4. Set `app.py` as the main file
5. Deploy — datasets auto-download from Google Drive on first run

> ⚠️ Delete `resume_model.pkl` before deploying so the model retrains fresh on the server.

---

## 🧠 ML Model

- **Algorithm:** Random Forest Classifier
- **Features:** TF-IDF (5000 features, unigrams + bigrams)
- **Training data:** `processed_resumes.csv` with labeled job categories
- **Auto-trains** on first run if `resume_model.pkl` is not found
- **Saved** as `resume_model.pkl` after training

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| ML Model | Scikit-learn (RandomForest + TF-IDF) |
| PDF Parsing | pdfplumber |
| Data | pandas, numpy |
| Similarity | Scikit-learn cosine similarity |
| Dataset Download | gdown |

---

## 📁 Datasets

| File | Source |
|---|---|
| `processed_resumes.csv` | Google Drive (auto-downloaded) |
| `processed_jobs.csv` | Google Drive (auto-downloaded) |
| `processed_skills.csv` | Committed to repo |

---

## 🙌 Built With

Python · Streamlit · Scikit-learn · pdfplumber

---

