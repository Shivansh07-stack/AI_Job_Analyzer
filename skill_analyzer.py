# Core analytics — growth trends, survival scores, forecasts, recommendations

from pathlib import Path

import numpy as np
import pandas as pd

from config import AI_RESISTANT_SKILLS

DATA_DIR = Path("data")


def load_timeseries() -> pd.DataFrame:
    """Load the skill time-series CSV built by data_collector.py."""
    path = DATA_DIR / "skill_demand_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(
            "data/skill_demand_timeseries.csv not found — run data_collector.py first"
        )
    return pd.read_csv(path)


def compute_skill_growth(df: pd.DataFrame, min_years: int = 3) -> pd.DataFrame:
    """Fit a linear trend on pct_respondents vs year for each skill.

    Only includes skills that appear in at least `min_years` surveys.
    Returns slope, r², earliest/latest %, total growth, etc.
    """
    records = []

    for skill, grp in df.groupby("skill"):
        grp = grp.sort_values("year")

        if len(grp) < min_years:
            continue

        years = grp["year"].to_numpy(dtype=float)
        pct   = grp["pct_respondents"].to_numpy(dtype=float)

        coeffs = np.polyfit(years, pct, 1)
        slope = float(coeffs[0])
        pct_hat = np.polyval(coeffs, years)
        ss_res = float(np.sum((pct - pct_hat) ** 2))
        ss_tot = float(np.sum((pct - pct.mean()) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        earliest_pct = float(pct[0])
        latest_pct = float(pct[-1])

        if earliest_pct > 0:
            total_growth = (latest_pct - earliest_pct) / earliest_pct * 100
        else:
            total_growth = 0.0

        records.append(
            {
                "skill": skill,
                "slope": round(slope, 4),
                "r_squared": round(r_squared, 3),
                "latest_pct": round(latest_pct, 2),
                "earliest_pct": round(earliest_pct, 2),
                "total_growth_pct": round(total_growth, 2),
                "years_tracked": len(grp),
                "first_year": int(years[0]),
                "last_year": int(years[-1]),
            }
        )

    return pd.DataFrame(records).sort_values("slope", ascending=False).reset_index(drop=True)


def get_emerging_skills(growth_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    return growth_df[growth_df["slope"] > 0].head(top_n).reset_index(drop=True)


def get_declining_skills(growth_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    return (
        growth_df[growth_df["slope"] < 0]
        .tail(top_n)
        .sort_values("slope")
        .reset_index(drop=True)
    )


def compute_survival_score(growth_df: pd.DataFrame) -> pd.DataFrame:
    """Composite score (0-100): demand growth (40) + AI resistance (40) + current demand (20).

    Skills without a resistance entry default to 0.50.
    """
    df = growth_df.copy()

    # demand growth component (0-40)
    s_min, s_max = df["slope"].min(), df["slope"].max()
    if s_max != s_min:
        df["growth_score"] = ((df["slope"] - s_min) / (s_max - s_min)) * 40
    else:
        df["growth_score"] = 20.0

    # AI resistance component (0-40)
    df["ai_resistance_raw"] = df["skill"].map(
        lambda s: AI_RESISTANT_SKILLS.get(s, 0.50)
    )
    df["resistance_score"] = df["ai_resistance_raw"] * 40

    # current demand component (0-20)
    p_min, p_max = df["latest_pct"].min(), df["latest_pct"].max()
    if p_max != p_min:
        df["demand_score"] = ((df["latest_pct"] - p_min) / (p_max - p_min)) * 20
    else:
        df["demand_score"] = 10.0

    df["survival_score"] = (
        df["growth_score"] + df["resistance_score"] + df["demand_score"]
    ).round(1)

    return df.sort_values("survival_score", ascending=False).reset_index(drop=True)


def forecast_demand(df, skill, target_year=2028):
    """Extrapolate skill demand to target_year using a linear fit."""
    skill_df = df[df["skill"] == skill].sort_values("year")

    if len(skill_df) < 2:
        return None

    years = skill_df["year"].to_numpy(dtype=float)
    pct = skill_df["pct_respondents"].to_numpy(dtype=float)

    coeffs = np.polyfit(years, pct, 1)
    predicted = float(np.clip(np.polyval(coeffs, target_year), 0, 100))
    current = float(pct[-1])

    def label(p):
        if p >= 40:  return "Very High"
        if p >= 20:  return "High"
        if p >= 10:  return "Moderate"
        if p >= 3:   return "Low"
        return "Very Low"

    return {
        "skill": skill,
        "current_pct": round(current, 2),
        "current_demand": label(current),
        "predicted_pct": round(predicted, 2),
        "predicted_demand": label(predicted),
        "target_year": target_year,
        "trend": "Growing" if predicted > current else "Declining",
        "years": years.astype(int).tolist(),
        "historical_pct": [round(float(v), 2) for v in pct],
    }


def recommend_skills(current_skills, growth_df, survival_df, top_n=5):
    """Suggest next skills to learn based on what the user already knows."""
    current_lower = {s.lower() for s in current_skills}

    candidates = survival_df[
        ~survival_df["skill"].str.lower().isin(current_lower)
    ].head(50)

    recs = []
    for _, row in candidates.iterrows():
        recs.append(
            {
                "skill": row["skill"],
                "survival_score": row["survival_score"],
                "growth_trend": "Growing" if row["slope"] > 0 else "Declining",
                "slope": row["slope"],
                "current_demand_pct": row["latest_pct"],
                "reason": _reason(row),
            }
        )
        if len(recs) == top_n:
            break

    return recs


def _reason(row):
    if row["slope"] > 1.0 and row["survival_score"] > 70:
        return "fast growth + AI-resistant"
    if row["slope"] > 0.5:
        return "solid growth trend"
    if row["survival_score"] > 80:
        return "high survival score"
    if row["latest_pct"] > 30:
        return "high demand right now"
    if row["ai_resistance_raw"] > 0.75:
        return "AI-resistant"
    return "good overall score"
