def skill_match(resume_skills, job_skills):

    resume_set = set(resume_skills)
    job_set = set(job_skills)

    if len(job_set) == 0:
        return 0

    matched = resume_set.intersection(job_set)

    score = (len(matched) / len(job_set)) * 100

    return round(score, 2)


def missing_skills(resume_skills, job_skills):

    resume_set = set(resume_skills)
    job_set = set(job_skills)

    missing = job_set - resume_set

    return list(missing)

