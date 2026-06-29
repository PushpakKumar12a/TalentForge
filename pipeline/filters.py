from datetime import datetime
from .config import TODAY, INACTIVE_CUTOFF_DAYS, MIN_RESPONSE_RATE


def parse_date(date_string):
    if not date_string:
        return None
    try:
        return datetime.strptime(date_string, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def days_since(date_value):
    if date_value:
        return (TODAY - date_value).days
    return 9999


def check_zero_exp_advanced(skills, career, years):
    if years != 0:
        return False
    count = 0
    for skill in skills:
        if skill.get("proficiency") in ("expert", "advanced"):
            count += 1
    return count >= 3

def check_zero_duration_experts(skills, career, years):
    count = 0
    for skill in skills:
        if skill.get("proficiency") == "expert" and skill.get("duration_months", 0) == 0:
            count += 1
    return count >= 3

def check_too_many_experts(skills, career, years):
    count = 0
    for skill in skills:
        if skill.get("proficiency") == "expert":
            count += 1
    return count >= 8 and years < 5

def check_career_overflow(skills, career, years):
    if years <= 0:
        return False
    total_months = 0
    for job in career:
        total_months += job.get("duration_months", 0)
    return total_months > years * 24

def check_role_overflow(skills, career, years):
    if years <= 0:
        return False
    max_allowed = years * 12 + 18
    for job in career:
        if job.get("duration_months", 0) > max_allowed:
            return True
    return False

def check_broad_expertise(skills, career, years):
    count = 0
    for skill in skills:
        if skill.get("proficiency") == "expert":
            count += 1
    return count >= 10

def check_zero_endorsements(skills, career, years):
    count = 0
    for skill in skills:
        if skill.get("proficiency") == "expert" and skill.get("endorsements", 0) == 0:
            count += 1
    return count >= 5

def check_ghost_responder(candidate):
    signals = candidate.get("redrob_signals", {})
    return signals.get("recruiter_response_rate", 0) > 0 and signals.get("profile_views_received_30d", 0) == 0 and signals.get("search_appearance_30d", 0) == 0

def check_skill_duration_overflow(candidate):
    skills = candidate.get("skills", [])
    years_exp = candidate.get("profile", {}).get("years_of_experience", 0)
    max_skill_dur = max([s.get("duration_months", 0) for s in skills] + [0])
    return max_skill_dur > years_exp * 12 + 60

HONEYPOT_RULES = [
    (check_zero_exp_advanced, 3),
    (check_zero_duration_experts, 3),
    (check_too_many_experts, 2),
    (check_career_overflow, 2),
    (check_role_overflow, 2),
    (check_broad_expertise, 2),
    (check_zero_endorsements, 2),
]


def is_honeypot(candidate):
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    years = candidate.get("profile", {}).get("years_of_experience", 0)
    total_score = 0
    for check_fn, weight in HONEYPOT_RULES:
        try:
            if check_fn(skills, career, years):
                total_score += weight
        except TypeError:
            if check_fn(candidate):
                total_score += weight

    if check_ghost_responder(candidate):
        total_score += 2
    if check_skill_duration_overflow(candidate):
        total_score += 2

    return total_score >= 2


def passes_prefilter(candidate):
    if is_honeypot(candidate):
        return False
    return True
