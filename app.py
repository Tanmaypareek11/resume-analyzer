# app.py

import streamlit as st
import re
import pdfplumber
import pickle
import os
import nltk

# -------------------------------------------------------
# STEP 1 — Download NLTK data (needed on server)
# -------------------------------------------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# -------------------------------------------------------
# STEP 2 — Download datasets from Google Drive FIRST
#           before any other code runs
# -------------------------------------------------------
from load_data import download_datasets

with st.spinner("⏳ Loading datasets... please wait"):
    download_datasets()
#-------------------------------------------------------

# ✅ Step 3 — Safety check inside get_ml_model()
if not os.path.exists("processed_resumes.csv"):
    download_datasets()

# -------------------------------------------------------
# STEP 4 — Now import everything else
# -------------------------------------------------------
from skill_extraction import extract_skills
from matcher import skill_match, missing_skills
from model import train_model, load_model, predict_category, find_matching_jobs

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="ResumeIQ — AI Job Match System",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------------------------------
# CUSTOM CSS — Professional Dark Theme
# -------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0A0F;
    color: #E8E8F0;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 0%, #1a1040 0%, #0A0A0F 50%),
                radial-gradient(ellipse at 80% 100%, #0d2040 0%, #0A0A0F 50%);
    background-blend-mode: screen;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 2rem 3rem 4rem 3rem;
    max-width: 1100px;
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6C47FF20, #00D4FF20);
    border: 1px solid #6C47FF50;
    border-radius: 100px;
    padding: 6px 18px;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #A78BFA;
    margin-bottom: 1.2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 1rem 0;
    background: linear-gradient(135deg, #FFFFFF 0%, #A78BFA 50%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #8888AA;
    font-weight: 300;
    margin-bottom: 2.5rem;
    line-height: 1.7;
}

.glass-card {
    background: linear-gradient(135deg, #ffffff08, #ffffff04);
    border: 1px solid #ffffff12;
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
}

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #6C47FF;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #E8E8F0;
    margin-bottom: 1rem;
}

[data-testid="stFileUploader"] {
    background: #ffffff06 !important;
    border: 2px dashed #6C47FF50 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #6C47FF !important;
    background: #6C47FF0A !important;
}

[data-testid="stFileUploader"] label { color: #A78BFA !important; }

[data-testid="stTextArea"] textarea {
    background: #ffffff06 !important;
    border: 1px solid #ffffff15 !important;
    border-radius: 12px !important;
    color: #E8E8F0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border 0.3s ease;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: #6C47FF !important;
    box-shadow: 0 0 0 2px #6C47FF20 !important;
}

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6C47FF, #4F46E5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px #6C47FF40 !important;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px #6C47FF60 !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff08, #ffffff04) !important;
    border: 1px solid #ffffff12 !important;
    border-radius: 14px !important;
    padding: 1.2rem 1.5rem !important;
}

[data-testid="stMetricLabel"] {
    color: #8888AA !important;
    font-size: 0.8rem !important;
    font-family: 'DM Sans', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #E8E8F0 !important;
}

.score-bar-wrap { margin: 0.6rem 0 1.2rem 0; }

.score-bar-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
    font-size: 0.85rem;
    color: #8888AA;
    font-family: 'DM Sans', sans-serif;
}

.score-bar-label span:last-child { color: #E8E8F0; font-weight: 500; }

.score-bar-bg {
    background: #ffffff10;
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
}

.score-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1s ease;
}

.skill-tag {
    display: inline-block;
    background: #6C47FF20;
    border: 1px solid #6C47FF40;
    color: #A78BFA;
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.8rem;
    margin: 3px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}

.skill-tag.missing { background: #FF4D4D15; border-color: #FF4D4D40; color: #FF8080; }
.skill-tag.job { background: #00D4FF15; border-color: #00D4FF40; color: #67E8F9; }

.result-excellent {
    background: linear-gradient(135deg, #00875A15, #00875A08);
    border: 1px solid #00875A50;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    color: #34D399;
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    text-align: center;
}

.result-good {
    background: linear-gradient(135deg, #92400E15, #92400E08);
    border: 1px solid #D9770650;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    color: #FCD34D;
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    text-align: center;
}

.result-low {
    background: linear-gradient(135deg, #7F1D1D15, #7F1D1D08);
    border: 1px solid #EF444450;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    color: #FCA5A5;
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    text-align: center;
}

.custom-divider { border: none; border-top: 1px solid #ffffff10; margin: 2rem 0; }

[data-testid="stExpander"] {
    background: #ffffff06 !important;
    border: 1px solid #ffffff10 !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary {
    color: #A78BFA !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSpinner"] { color: #6C47FF !important; }
[data-testid="stAlert"] { border-radius: 12px !important; font-family: 'DM Sans', sans-serif !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0A0A0F; }
::-webkit-scrollbar-thumb { background: #6C47FF50; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #6C47FF; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def score_bar(label, value, color):
    st.markdown(f"""
    <div class="score-bar-wrap">
        <div class="score-bar-label">
            <span>{label}</span>
            <span>{value}%</span>
        </div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{value}%; background:{color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def skill_tags(skills, tag_type=""):
    if not skills:
        return "<span style='color:#555566;font-size:0.85rem;'>None found</span>"
    return "".join([f'<span class="skill-tag {tag_type}">{s}</span>' for s in skills])


# -------------------------------------------------------
# CACHED MODELS
# -------------------------------------------------------
@st.cache_resource
def get_ml_model():
    # ✅ Double check datasets exist before training
    if not os.path.exists("data/processed_resumes.csv"):
        download_datasets()

    if os.path.exists("resume_model.pkl"):
        return load_model()
    else:
        st.info("⏳ Training ML model for the first time — please wait 2-3 minutes...")
        pipeline, _ = train_model()
        return pipeline


@st.cache_resource
def get_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


# -------------------------------------------------------
# HERO SECTION
# -------------------------------------------------------
st.markdown('<div class="hero-badge">✦ AI Powered</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Resume Analyzer &<br/>Job Match System</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Upload your resume and paste any job description.<br/>Get an instant AI-powered match score with skill gap analysis.</p>', unsafe_allow_html=True)

# -------------------------------------------------------
# INPUT SECTION
# -------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-label">Step 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Your Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your PDF resume here",
        type=["pdf"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        st.markdown(f"""
        <div style="margin-top:0.8rem; padding:0.7rem 1rem;
             background:#6C47FF15; border:1px solid #6C47FF40;
             border-radius:10px; font-size:0.85rem; color:#A78BFA;">
            ✓ &nbsp; {uploaded_file.name}
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-label">Step 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Paste Job Description</div>', unsafe_allow_html=True)
    job_text = st.text_area(
        "Job description",
        height=160,
        placeholder="Paste the job description here...",
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("🔍 Analyze My Resume", use_container_width=True)

# -------------------------------------------------------
# ANALYSIS
# -------------------------------------------------------
if analyze_btn:
    if uploaded_file and job_text:

        with st.spinner("Analyzing your resume with AI..."):

            # Read PDF
            resume_text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        resume_text += text

            resume_clean = clean_text(resume_text)
            job_clean = clean_text(job_text)

            # ML Model
            ml_model = get_ml_model()
            predicted_category, confidence = predict_category(resume_clean, ml_model)

            # Skills
            resume_skills = extract_skills(resume_clean)
            job_skills = extract_skills(job_clean)
            score = skill_match(resume_skills, job_skills)
            missing = missing_skills(resume_skills, job_skills)

            # TF-IDF
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])
            tfidf_score = round(
                cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100, 2
            )

            # BERT
            bert_model = get_bert_model()
            r_emb = bert_model.encode(resume_clean)
            j_emb = bert_model.encode(job_clean)
            bert_score = round(
                cosine_similarity([r_emb], [j_emb])[0][0] * 100, 2
            )

            # Normalize & Final Score
            tfidf_normalized = min(tfidf_score * 5, 100)
            final_score = (
                (0.15 * tfidf_normalized) +
                (0.30 * score) +
                (0.40 * bert_score) +
                (0.15 * confidence)
            )
            final_score = round(min(final_score, 100), 2)

        # ── RESULTS ──────────────────────────────────────────
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Analysis Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Here is your report</div>', unsafe_allow_html=True)

        # Top Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("🔥 Final Score", f"{final_score}%")
        with m2:
            st.metric("🤖 Job Category", predicted_category)
        with m3:
            st.metric("📈 ML Confidence", f"{confidence}%")
        with m4:
            st.metric("🎯 Skill Match", f"{score}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # Result Banner
        if final_score >= 75:
            st.markdown('<div class="result-excellent">🎉 Excellent Match — You are strongly suited for this role. Apply now!</div>', unsafe_allow_html=True)
        elif final_score >= 55:
            st.markdown('<div class="result-good">✅ Good Match — You are suitable with a few minor improvements needed.</div>', unsafe_allow_html=True)
        elif final_score >= 35:
            st.markdown('<div class="result-good">⚠️ Average Match — Work on the missing skills before applying.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-low">❌ Low Match — Your profile needs significant improvement for this role.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Score Breakdown + Skills Side by Side
        col_scores, col_skills = st.columns([1, 1], gap="large")

        with col_scores:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Score Breakdown</div>', unsafe_allow_html=True)
            score_bar("BERT Semantic Similarity", bert_score, "linear-gradient(90deg, #6C47FF, #A78BFA)")
            score_bar("Skill Match Score", score, "linear-gradient(90deg, #0EA5E9, #38BDF8)")
            score_bar("TF-IDF Text Similarity", tfidf_score, "linear-gradient(90deg, #10B981, #34D399)")
            score_bar("ML Model Confidence", confidence, "linear-gradient(90deg, #F59E0B, #FCD34D)")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_skills:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Skills Found in Resume</div>', unsafe_allow_html=True)
            st.markdown(skill_tags(resume_skills), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Skills Required by Job</div>', unsafe_allow_html=True)
            st.markdown(skill_tags(job_skills, "job"), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Missing Skills to Add</div>', unsafe_allow_html=True)
            st.markdown(
                skill_tags(missing, "missing") if missing
                else '<span style="color:#34D399; font-size:0.85rem;">✓ No missing skills!</span>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Similar Jobs from Dataset
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">From the Dataset</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💼 Similar Job Descriptions</div>', unsafe_allow_html=True)

        matched_jobs = find_matching_jobs(predicted_category, top_n=3)
        if not matched_jobs.empty:
            for i, row in matched_jobs.iterrows():
                with st.expander(f"Similar Job {i + 1} — {predicted_category}"):
                    st.markdown(
                        f'<p style="color:#AAAACC; font-size:0.88rem; line-height:1.8;">{row["clean_job"][:600]}...</p>',
                        unsafe_allow_html=True
                    )
        else:
            st.markdown('<p style="color:#555566;">No similar jobs found in dataset.</p>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#FF4D4D10; border:1px solid #FF4D4D40;
             border-radius:12px; padding:1rem 1.5rem; color:#FCA5A5;
             font-family:'DM Sans',sans-serif; text-align:center; margin-top:1rem;">
            ⚠️ &nbsp; Please upload your resume PDF and paste a job description.
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.markdown("""
<div style="text-align:center; margin-top:4rem; padding-top:2rem;
     border-top:1px solid #ffffff08; color:#333344;
     font-size:0.78rem; font-family:'DM Sans',sans-serif;">
    ResumeIQ &nbsp;·&nbsp; AI-Powered Resume Analyzer &nbsp;·&nbsp; Built with Python & Streamlit
</div>
""", unsafe_allow_html=True)
