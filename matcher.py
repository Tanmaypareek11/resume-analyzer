# matcher.py

def skill_match(resume_skills, job_skills):
    """Return percentage of job skills found in resume."""
    resume_set = set(resume_skills) if resume_skills else set()
    job_set = set(job_skills) if job_skills else set()

    if len(job_set) == 0:
        return 0

    matched = resume_set.intersection(job_set)
    score = (len(matched) / len(job_set)) * 100
    return round(score, 2)


def missing_skills(resume_skills, job_skills):
    """Return skills required by job but missing from resume."""
    resume_set = set(resume_skills) if resume_skills else set()
    job_set = set(job_skills) if job_skills else set()
    return list(job_set - resume_set)
