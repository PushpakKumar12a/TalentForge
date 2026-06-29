from .config import MAX_TEXT_BM25, MAX_TEXT_RERANK

def join_parts(parts, limit=None):
    filtered = []
    for part in parts:
        if part:
            filtered.append(part)
    text = " ".join(filtered)
    if limit:
        return text[:limit]
    return text


def get_skill_names(candidate):
    names = []
    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        if name:
            names.append(name)
    return " ".join(names)


def build_text_short(candidate):
    profile = candidate.get("profile", {})

    parts = [
        profile.get("headline", ""),
        profile.get("summary", "")[:200],
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
        get_skill_names(candidate),
    ]

    for job in candidate.get("career_history", [])[:2]:
        title = job.get("title", "")
        description = job.get("description", "")[:80]
        parts.append(f"{title} {description}")

    return join_parts(parts, limit=MAX_TEXT_BM25)


def build_text_full(candidate):
    profile = candidate.get("profile", {})

    skill_list = []
    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        if name:
            skill_list.append(name)

    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        f"{profile.get('current_title', '')} at {profile.get('current_company', '')}",
        f"{profile.get('years_of_experience', 0)} years experience",
        f"Skills: {', '.join(skill_list)}",
    ]

    for job in candidate.get("career_history", [])[:3]:
        title = job.get("title", "")
        company = job.get("company", "")
        description = job.get("description", "")[:150]
        parts.append(f"{title} at {company}: {description}")

    for edu in candidate.get("education", [])[:2]:
        degree = edu.get("degree", "")
        field = edu.get("field_of_study", "")
        institution = edu.get("institution", "")
        parts.append(f"{degree} {field} {institution}")

    return join_parts(parts, limit=MAX_TEXT_RERANK)
