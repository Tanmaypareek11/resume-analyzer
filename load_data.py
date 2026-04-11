# load_data.py

import os
import gdown

# ── Paste your File IDs here ──
RESUMES_FILE_ID = "1OMYvC6AZl7GUJ2jUNDRUICujf4eKk527"
JOBS_FILE_ID    = "1ZuC71FDC6VB5xh34Dvm-UqwSOAOhCJw0"

def download_datasets():

    os.makedirs("data", exist_ok=True)

    # Download processed_resumes.csv
    if not os.path.exists("data/processed_resumes.csv"):
        print("⏳ Downloading resumes dataset from Google Drive...")
        gdown.download(
            f"https://drive.google.com/uc?id={RESUMES_FILE_ID}",
            "data/processed_resumes.csv",
            quiet=False
        )
        print("✅ Resumes dataset downloaded!")
    else:
        print("✅ Resumes dataset already exists!")

    # Download processed_jobs.csv
    if not os.path.exists("data/processed_jobs.csv"):
        print("⏳ Downloading jobs dataset from Google Drive...")
        gdown.download(
            f"https://drive.google.com/uc?id={JOBS_FILE_ID}",
            "data/processed_jobs.csv",
            quiet=False
        )
        print("✅ Jobs dataset downloaded!")
    else:
        print("✅ Jobs dataset already exists!")

    print("✅ All datasets ready!")
