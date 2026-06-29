import numpy as np

from .config import (
    CONSULTING_FIRMS, LOC_PRI, LOC_SEC,
    TITLE_TIERS, TITLE_DEFAULT_MULT,
)
from .filters import parse_date, days_since

# Threshold tables as Numpy arrays
TABLES = {
    "exp":        (np.array([3, 4, 5, 9, 12, 15]),      np.array([0.78, 0.92, 1.00, 1.08, 1.00, 0.92, 0.78])),
    "notice":     (np.array([30, 60, 90]),              np.array([1.04, 0.97, 0.90, 0.82])),
    "rr":         (np.array([0.06, 0.20, 0.40, 0.60]),  np.array([0.10, 0.92, 0.97, 1.02, 1.05])),
    "recency":    (np.array([14, 45, 90, 179]),         np.array([1.05, 1.02, 0.98, 0.93, 0.10])),
    "github":     (np.array([20, 50]),                  np.array([1.00, 1.02, 1.05])),
    "profile":    (np.array([70, 85]),                  np.array([0.95, 1.00, 1.03])),
    "avg_resp":   (np.array([2, 12, 24, 48]),           np.array([1.05, 1.00, 0.95, 0.90, 0.85])),
    "views":      (np.array([10, 50, 100]),             np.array([0.95, 1.00, 1.02, 1.05])),
    "apps":       (np.array([5, 20, 50]),               np.array([0.98, 1.00, 1.02, 1.00])),
    "assess":     (np.array([60, 80, 95]),              np.array([0.95, 1.00, 1.03, 1.06])),
    "conn":       (np.array([50, 200, 500]),            np.array([0.95, 1.00, 1.02, 1.05])),
    "endors":     (np.array([5, 20, 50]),               np.array([0.98, 1.00, 1.02, 1.05])),
    "search":     (np.array([5, 20, 50]),               np.array([0.95, 1.00, 1.02, 1.04])),
    "saved":      (np.array([1, 5, 10]),                np.array([0.98, 1.00, 1.03, 1.06])),
    "int_rate":   (np.array([0.5, 0.8, 0.95]),          np.array([0.85, 0.95, 1.00, 1.05])),
    "offer_rate": (np.array([0.3, 0.7, 0.9]),           np.array([0.90, 0.98, 1.02, 1.05])),
}

# Numpy searchsorted threshold lookup.
def step_lookup(value, table_key):
    breakpoints, values = TABLES[table_key]
    index = np.searchsorted(breakpoints, value, side="right")
    return float(values[index])

# Returns multiplier based on job title using keywords.  
def title_multiplier(title_lower):
    for tier_name, (keywords, multiplier) in TITLE_TIERS.items():
        for keyword in keywords:
            if keyword in title_lower:
                return multiplier
    return TITLE_DEFAULT_MULT

# Returns multiplier based on location.
def location_multiplier(location, country, willing_to_relocate):
    for loc in LOC_PRI:
        if loc in location:
            return 1.08
    for loc in LOC_SEC:
        if loc in location:
            return 1.04
    if "india" in country:
        if willing_to_relocate:
            return 1.00
        return 0.92
    if willing_to_relocate:
        return 0.88
    return 0.78

# Returns multiplier based on whether candidate has worked in consulting firms.
def consulting_multiplier(career_history):
    if not career_history:
        return 1.0
    for job in career_history:
        company_name = job.get("company", "").lower()
        is_consulting = False
        for firm in CONSULTING_FIRMS:
            if firm in company_name:
                is_consulting = True
                break
        if not is_consulting:
            return 1.0
    return 0.65

# Returns multiplier based on verification signals.
def verification_multiplier(signals):
    verified_count = 0
    if signals.get("verified_email", False):
        verified_count += 1
    if signals.get("verified_phone", False):
        verified_count += 1
    if signals.get("linkedin_connected", False):
        verified_count += 1
    lookup = [0.95, 1.0, 1.0, 1.03]
    return lookup[verified_count]

# Returns multiplier based on signup date.
def loyalty_multiplier(signup_date_str):
    if not signup_date_str: return 1.0
    days = days_since(parse_date(signup_date_str))
    if days > 1000: return 1.05
    if days > 365: return 1.02
    return 1.0

# Returns multiplier based on expected salary range.
def salary_multiplier(salary_dict):
    if not salary_dict: return 1.0
    s_min = salary_dict.get("min", 0)
    if s_min > 50: return 0.85
    if s_min > 35: return 0.95
    return 1.0

# Returns multiplier based on work mode.
def work_mode_multiplier(mode):
    mode = (mode or "").lower()
    if mode in ["onsite", "hybrid", "flexible"]: return 1.05
    if mode == "remote": return 0.95
    return 1.0

# Returns multiplier based on candidate behavior.
def behavioral_multiplier(candidate):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career_history = candidate.get("career_history", [])

    # 1. last_active_date
    last_active_date = signals.get("last_active_date")
    recency_days = days_since(parse_date(last_active_date))

    # 2. open_to_work_flag
    open_to_work_mult = 1.04 if signals.get("open_to_work_flag") else 0.96

    # 3. offer_acceptance_rate
    offer_rate_val = signals.get("offer_acceptance_rate", -1)
    offer_mult = 1.0 if offer_rate_val < 0 else step_lookup(offer_rate_val, "offer_rate")

    # 4. skill_assessment_scores
    assess_dict = signals.get("skill_assessment_scores", {})
    if assess_dict:
        avg_score = sum(assess_dict.values()) / len(assess_dict)
    else:
        avg_score = 70
    assess_mult = step_lookup(avg_score, "assess")

    factors = np.array([
        # Base multipliers
        step_lookup(profile.get("years_of_experience", 0), "exp"),
        title_multiplier(profile.get("current_title", "").lower()),
        location_multiplier(
            profile.get("location", "").lower(),
            profile.get("country", "").lower(),
            signals.get("willing_to_relocate", False),
        ),
        consulting_multiplier(career_history),
        
        # Redrob signals
        step_lookup(signals.get("notice_period_days", 30), "notice"),
        step_lookup(signals.get("recruiter_response_rate", 0.5), "rr"),
        step_lookup(recency_days, "recency"),
        open_to_work_mult,
        step_lookup(signals.get("github_activity_score", -1), "github"),
        step_lookup(signals.get("profile_completeness_score", 50), "profile"),
        verification_multiplier(signals),
        
        loyalty_multiplier(signals.get("signup_date")),
        step_lookup(signals.get("profile_views_received_30d", 0), "views"),
        step_lookup(signals.get("applications_submitted_30d", 0), "apps"),
        step_lookup(signals.get("avg_response_time_hours", 24), "avg_resp"),
        assess_mult,
        step_lookup(signals.get("connection_count", 0), "conn"),
        step_lookup(signals.get("endorsements_received", 0), "endors"),
        salary_multiplier(signals.get("expected_salary_range_inr_lpa")),
        work_mode_multiplier(signals.get("preferred_work_mode")),
        step_lookup(signals.get("search_appearance_30d", 0), "search"),
        step_lookup(signals.get("saved_by_recruiters_30d", 0), "saved"),
        step_lookup(signals.get("interview_completion_rate", 1.0), "int_rate"),
        offer_mult,
    ], dtype=np.float16)

    return float(np.prod(factors))


def rank_based_score(rank, total=100):
    ratio = (rank - 1) / (total - 1)
    decay = ratio ** 0.55
    score = 1.0 - 0.65 * decay
    return round(score, 6)
