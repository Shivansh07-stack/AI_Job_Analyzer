# Prediction models — exposure scoring, demand forecast, YoY growth

from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from config import AI_EXPOSURE_SCORES


def get_ai_exposure_score(job_title: str) -> dict:
    """Return an AI exposure score (0-100) for the given job title.

    Tries exact match first, then fuzzy matching, then keyword heuristic.
    """
    job_lower = job_title.lower().strip()

    # exact match
    if job_lower in AI_EXPOSURE_SCORES:
        return _format(job_title, AI_EXPOSURE_SCORES[job_lower], "exact match")

    # fuzzy fallback
    best_key, best_ratio = "", 0.0
    for key in AI_EXPOSURE_SCORES:
        ratio = SequenceMatcher(None, job_lower, key).ratio()
        if ratio > best_ratio:
            best_ratio, best_key = ratio, key

    if best_ratio >= 0.55:
        score = AI_EXPOSURE_SCORES[best_key]
        return _format(job_title, score, f"fuzzy match → '{best_key}'")

    # keyword heuristic
    score = _keyword_score(job_lower)
    return _format(job_title, score, "keyword heuristic")


def _keyword_score(title):
    # check from most-specific to least-specific
    very_low = ["mlops", "ai security", "robotics", "deep learning", "llm"]
    low      = ["machine learning", "ml ", "ai ", "engineer", "scientist",
                "architect", "devops", "security", "researcher", "cloud"]
    medium   = ["analyst", "coordinator", "specialist", "designer",
                "writer", "reporter", "manager", "consultant"]
    high     = ["entry", "clerk", "assistant", "agent", "operator",
                "bookkeeper", "accountant", "telemarket"]

    for kw in very_low:
        if kw in title:
            return 13
    for kw in low:
        if kw in title:
            return 28
    for kw in medium:
        if kw in title:
            return 52
    for kw in high:
        if kw in title:
            return 80
    return 45   # default: moderate


def _format(job_title, score, method):
    if score >= 70:
        level = "High Risk"
        colour = "red"
        advice = "High automation risk — worth looking into AI-adjacent roles."
    elif score >= 40:
        level = "Medium Risk"
        colour = "orange"
        advice = "Some automation exposure. Picking up AI-related skills would help."
    else:
        level = "Low Risk"
        colour = "green"
        advice = "Low risk for now. Keep building depth in your area."

    return {
        "job_title": job_title,
        "score": score,
        "risk_level": level,
        "colour": colour,
        "advice": advice,
        "method": method,
    }


# --- demand prediction ---

def predict_skill_demand(
    timeseries_df: pd.DataFrame,
    skill: str,
    forecast_years: list[int] | None = None,
) -> pd.DataFrame | None:
    """Historical + forecasted demand for a skill. Returns None if not enough data."""
    if forecast_years is None:
        forecast_years = [2026, 2027, 2028]

    skill_df = (
        timeseries_df[timeseries_df["skill"].str.lower() == skill.lower()]
        .sort_values("year")
    )

    if len(skill_df) < 2:
        return None

    years = skill_df["year"].to_numpy(dtype=float)
    pct   = skill_df["pct_respondents"].to_numpy(dtype=float)

    coeffs = np.polyfit(years, pct, 1)

    historical = pd.DataFrame(
        {
            "year": years.astype(int),
            "pct_respondents": pct.round(2),
            "row_type": "historical",
        }
    )

    forecasted = pd.DataFrame(
        {
            "year": forecast_years,
            "pct_respondents": [
                round(float(np.clip(np.polyval(coeffs, yr), 0, 100)), 2)
                for yr in forecast_years
            ],
            "row_type": "forecast",
        }
    )

    return pd.concat([historical, forecasted], ignore_index=True)


def compute_yoy_growth(timeseries_df: pd.DataFrame) -> pd.DataFrame:
    """Average year-over-year % growth for every skill."""
    records = []

    for skill, grp in timeseries_df.groupby("skill"):
        grp = grp.sort_values("year")
        if len(grp) < 2:
            continue

        pct     = grp["pct_respondents"].to_numpy(dtype=float)
        yoy     = np.diff(pct) / np.where(pct[:-1] > 0, pct[:-1], np.nan) * 100
        avg_yoy = float(np.nanmean(yoy))
        latest  = float(pct[-1])

        records.append(
            {
                "skill": skill,
                "avg_yoy_growth_pct": round(avg_yoy, 2),
                "latest_pct": round(latest, 2),
                "data_points": len(grp),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("avg_yoy_growth_pct", ascending=False)
        .reset_index(drop=True)
    )
