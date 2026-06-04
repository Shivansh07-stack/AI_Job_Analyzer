# Downloads SO survey CSVs and builds the skill demand time-series

import os
import logging
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

from config import SURVEY_URLS, SKILL_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def download_survey(year: int, force: bool = False) -> pd.DataFrame | None:
    """Download and cache the SO Developer Survey CSV for a given year."""
    cache_path = DATA_DIR / f"survey_{year}.csv"

    if cache_path.exists() and not force:
        logger.info(f"[{year}] Loading from cache: {cache_path}")
        return pd.read_csv(cache_path, low_memory=False)

    url = SURVEY_URLS.get(year)
    if url is None:
        logger.error(f"[{year}] No URL configured.")
        return None

    logger.info(f"[{year}] Downloading from {url} ...")
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        with open(cache_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)

        df = pd.read_csv(cache_path, low_memory=False)
        logger.info(f"[{year}] done — {len(df):,} rows x {len(df.columns)} columns")
        return df

    except requests.RequestException as exc:
        logger.error(f"[{year}] Download failed: {exc}")
        if cache_path.exists():
            cache_path.unlink()
        return None


def extract_skills(df, year):
    """Parse skill columns for a given year, return Counter of {skill: count}."""
    counts = Counter()
    columns = SKILL_COLUMNS.get(year, [])

    for col in columns:
        if col not in df.columns:
            logger.debug(f"[{year}] Column '{col}' not found, skipping")
            continue

        for cell in df[col].dropna():
            skills = [s.strip() for s in str(cell).split(";") if s.strip()]
            counts.update(skills)

    return counts


def build_timeseries(years=None, force_download=False):
    """Build the master skill-demand time-series CSV from SO survey data."""
    if years is None:
        years = sorted(SURVEY_URLS.keys())

    output_path = DATA_DIR / "skill_demand_timeseries.csv"

    all_rows = []

    for year in years:
        df = download_survey(year, force=force_download)
        if df is None:
            continue

        total = len(df)
        counts = extract_skills(df, year)

        if not counts:
            logger.warning(f"[{year}] No skills extracted — check SKILL_COLUMNS config.")
            continue

        for skill, freq in counts.items():
            all_rows.append(
                {
                    "skill": skill,
                    "year": year,
                    "frequency": freq,
                    "pct_respondents": round(freq / total * 100, 4),
                    "total_respondents": total,
                    "source": "stackoverflow_survey",
                }
            )

    if not all_rows:
        raise RuntimeError("no survey data was collected")

    timeseries = pd.DataFrame(all_rows)
    timeseries.to_csv(output_path, index=False)

    logger.info(f"Saved {output_path}")
    logger.info(f"  {len(timeseries):,} records, {timeseries['skill'].nunique():,} skills")
    logger.info(f"  years: {sorted(timeseries['year'].unique())}")

    return timeseries


if __name__ == "__main__":
    print("Starting data collection...")
    df = build_timeseries()
    print(f"Done. {df.shape[0]} rows, {df.shape[1]} cols")
    print(df.head(15).to_string(index=False))
