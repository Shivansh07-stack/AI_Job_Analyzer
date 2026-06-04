# AI Job Market Disruption Analyzer

Ever wondered if your skills will still matter in 2 years? Or whether the job you're training for is slowly being eaten by AI?

That's exactly what this project tries to answer.

It pulls 7 years of real data from the Stack Overflow Developer Survey (2019–2025) — where over 49,000 developers worldwide report what they use, what they're learning, and what they're moving away from — and turns that into a live, interactive dashboard that tells you which skills are rising, which are dying, and how vulnerable any job is to AI automation.

---

## What Problem Does This Solve?

Students spend years learning skills that go out of demand. Professionals upskill in the wrong direction. No one has a clear, data-backed answer to the question:

> *"What should I learn today to stay relevant tomorrow?"*

This project builds that answer — not from guesswork, but from actual developer survey data tracked year by year.

---

## Getting Started

You only need three commands to go from zero to a running dashboard.

**Step 1 — Install the dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Download the survey data** *(runs once, takes ~2 minutes)*
```bash
python data_collector.py
```
This downloads 7 years of Stack Overflow survey CSVs and saves them locally so you never have to download them again. Skipping this step is fine too — the dashboard will handle it automatically on first launch.

**Step 3 — Launch the dashboard**
```bash
streamlit run dashboard.py
```
Then open your browser at `http://localhost:8501` and you're in.

---

## How the Project is Organized

```
ai_job_analyzer/
│
├── config.py            → The control room. All URLs, column names, and
│                          AI exposure scores live here. Edit this to
│                          add new job roles or tweak score values.
│
├── data_collector.py    → Downloads the SO survey CSVs and combines
│                          them into one clean time-series file.
│
├── skill_analyzer.py    → The brain. Computes growth trends, survival
│                          scores, forecasts, and recommendations.
│
├── models.py            → Handles the AI Disruption scoring and
│                          demand prediction logic.
│
├── dashboard.py         → The Streamlit dashboard you actually see.
│                          Six tabs, fully interactive.
│
├── requirements.txt     → Python packages needed to run the project.
│
└── data/                → Auto-created folder. Stores downloaded CSVs
                           and the master time-series file. You don't
                           need to touch this.
```

---

## What's Inside the Dashboard

### Emerging Skills
Shows the top N fastest-growing skills based on how consistently their
demand has risen across surveys. Not just "popular right now" — but
genuinely trending upward year over year.

### Declining Skills
The flip side. Skills that developers are quietly abandoning. Useful
if you want to know what *not* to invest time in.

### AI Disruption Index
Type any job title and get an automation risk score from 0 to 100.
Under 40 is relatively safe. Above 70 is a serious warning sign.
The score uses fuzzy matching so you don't need to type perfectly —
"software eng" works just as well as "software engineer".

### Career Recommender
Tell the app what skills you already have and it recommends what to
learn next — ranked by a survival score that factors in growth trend,
current market demand, and AI resistance.

### Skill Survival Score
Every skill gets a score out of 100, calculated from three things:
how fast it's growing, how resistant it is to AI replacement, and
how much demand it currently has. Think of it as a long-term bet
score for each skill.

### Future Forecast
Pick any skill and see its projected demand up to 2030, shown as a
chart with a clear historical line and a dashed forecast line. A
batch forecast table covers the top 15 skills at once.

---

## Where the Data Comes From

Everything is built on the **Stack Overflow Annual Developer Survey** —
one of the largest and most respected developer surveys in the world.

- Website: https://survey.stackoverflow.co
- GitHub mirror: https://github.com/StackExchange/Survey
- Years covered: 2019, 2020, 2021, 2022, 2023, 2024, 2025
- Roughly 49,000 respondents per year from 177 countries

Each response tells us which languages, databases, frameworks, and
tools that developer worked with that year. By tracking those numbers
across 7 years, we can see real trends instead of just snapshots.

One small gotcha — Stack Overflow renamed their columns in 2021.
`LanguageWorkedWith` became `LanguageHaveWorkedWith`, for example.
The project handles this automatically so you don't have to worry about it.

---

## How the Models Work

**Skill Growth Predictor**
Fits a straight line through each skill's yearly demand percentages.
The slope of that line tells you how fast a skill is growing or shrinking.
Simple, but surprisingly effective over a 7-year window.

**Skill Survival Score**
```
Score = Demand Growth (up to 40 pts)
      + AI Resistance (up to 40 pts)
      + Current Demand (up to 20 pts)
      ─────────────────────────────
      Total out of 100
```

**AI Disruption Index**
When you type a job title, it first tries an exact match, then a fuzzy
match (so typos and partial titles still work), and finally falls back
to keyword-based scoring if nothing else fits. The base scores come from
Frey & Osborne's landmark Oxford research on automation probability by
occupation.

**Future Demand Forecast**
Extends the growth line into the future using the same linear model.
The forecast is clamped between 0% and 100% so you never get nonsensical
predictions.

---

## Research Questions This Project Addresses

1. Which technical skills have the highest resistance to AI automation?
2. Does AI affect entry-level jobs more than senior roles?
3. Can job market disruptions be predicted 12 months in advance?

These make solid thesis statements if you're writing a report alongside this.

---

## Ideas for Taking It Further

The project is built to be extended. A few directions worth exploring:

- **Google Trends data** — use the `pytrends` library to layer in search
  interest alongside survey data for a richer signal.
- **GitHub API** — track repository growth for frameworks and languages
  to cross-validate what developers are actually building.
- **Better forecasting** — swap the linear model for Facebook Prophet or
  ARIMA if you want to handle seasonality and non-linear trends.
- **NLP skill extraction** — add spaCy and SkillNer to parse raw job
  descriptions from Naukri or Internshala and feed real posting data in.
- **Salary signals** — the SO surveys include salary data. Adding salary
  growth to the survival score would make it significantly more powerful.

---

## Dependencies

| Package | What it's used for |
|---|---|
| `streamlit` | Building the interactive dashboard |
| `plotly` | All the charts and the gauge meter |
| `pandas` | Loading, cleaning, and reshaping the survey data |
| `numpy` | Linear regression and numerical operations |
| `requests` | Downloading the survey CSV files |
| `scikit-learn` | Available for future ML model additions |

---

## Common Questions

**Do I need an API key for anything?**
No. The Stack Overflow survey data is completely free and publicly available.

**How long does the first download take?**
Usually 1–3 minutes depending on your connection. After that, everything
runs from local cache and is instant.

**Can I add my own job roles to the AI Disruption Index?**
Yes — just open `config.py` and add entries to the `AI_EXPOSURE_SCORES`
dictionary. The format is `"job title": score` where score is 0–100.

**Why does the data folder appear after I run the project?**
It's created automatically by the code. You don't need to create it yourself.
