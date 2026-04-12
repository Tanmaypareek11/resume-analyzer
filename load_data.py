# load_data.py

import os
import gdown

# ── Paste your File IDs here ──
RESUMES_FILE_ID = "1OMYvC6AZl7GUJ2jUNDRUICujf4eKk527"
JOBS_FILE_ID    = "1ZuC71FDC6VB5xh34Dvm-UqwSOAOhCJw0"

# ── Add your processed_skills.csv Google Drive File ID here ──
# If you don't have one hosted, set to None and the fallback skill list will be used
SKILLS_FILE_ID  = None   # e.g. "1AbcXYZ..."


def download_datasets():

    os.makedirs("data", exist_ok=True)

    # Download processed_resumes.csv
    if not os.path.exists("data/processed_resumes.csv"):
        print("⏳ Downloading resumes dataset from Google Drive...")
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={RESUMES_FILE_ID}",
                "data/processed_resumes.csv",
                quiet=False
            )
            print("✅ Resumes dataset downloaded!")
        except Exception as e:
            print(f"❌ Failed to download resumes dataset: {e}")
    else:
        print("✅ Resumes dataset already exists!")

    # Download processed_jobs.csv
    if not os.path.exists("data/processed_jobs.csv"):
        print("⏳ Downloading jobs dataset from Google Drive...")
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={JOBS_FILE_ID}",
                "data/processed_jobs.csv",
                quiet=False
            )
            print("✅ Jobs dataset downloaded!")
        except Exception as e:
            print(f"❌ Failed to download jobs dataset: {e}")
    else:
        print("✅ Jobs dataset already exists!")

    # Download processed_skills.csv (optional)
    if SKILLS_FILE_ID and not os.path.exists("data/processed_skills.csv"):
        print("⏳ Downloading skills dataset from Google Drive...")
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={SKILLS_FILE_ID}",
                "data/processed_skills.csv",
                quiet=False
            )
            print("✅ Skills dataset downloaded!")
        except Exception as e:
            print(f"⚠️ Could not download skills dataset (fallback list will be used): {e}")
    elif not SKILLS_FILE_ID:
        print("ℹ️ No skills file ID set — using built-in skill list.")

    print("✅ All datasets ready!")
