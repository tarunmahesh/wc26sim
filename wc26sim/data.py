import io
import urllib.request
import numpy as np
import pandas as pd

DATA_URL   = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
DECAY_RATE = 0.0018
SEED       = 42

# Elo ratings for all 48 qualified teams — sourced from eloratings.net (May 2026)
ELO = {
    "France": 2003, "Spain": 1966, "England": 1977,
    "Portugal": 1959, "Netherlands": 1952, "Belgium": 1944,
    "Germany": 1942, "Croatia": 1922, "Switzerland": 1817,
    "Austria": 1798, "Sweden": 1790, "Norway": 1775,
    "Scotland": 1740, "Bosnia and Herzegovina": 1720,
    "Czechia": 1760, "Turkey": 1786,
    "Brazil": 1988, "Argentina": 1977, "Colombia": 1826,
    "Uruguay": 1914, "Ecuador": 1751, "Paraguay": 1695,
    "United States": 1876, "Mexico": 1833, "Canada": 1692,
    "Panama": 1671, "Haiti": 1580, "Curaçao": 1560,
    "Morocco": 1870, "Senegal": 1843, "Ivory Coast": 1701,
    "Egypt": 1695, "Ghana": 1712, "Tunisia": 1703,
    "Algeria": 1698, "South Africa": 1665, "Cape Verde": 1640,
    "DR Congo": 1650,
    "Japan": 1856, "South Korea": 1762, "Iran": 1742,
    "Saudi Arabia": 1731, "Australia": 1773, "Qatar": 1652,
    "Iraq": 1658, "Jordan": 1620, "Uzbekistan": 1600,
    "New Zealand": 1560,
}

CONF = {
    "Mexico": "CONCACAF", "South Africa": "CAF", "South Korea": "AFC",
    "Czechia": "UEFA", "Canada": "CONCACAF", "Bosnia and Herzegovina": "UEFA",
    "Qatar": "AFC", "Switzerland": "UEFA", "Brazil": "CONMEBOL",
    "Morocco": "CAF", "Haiti": "CONCACAF", "Scotland": "UEFA",
    "United States": "CONCACAF", "Paraguay": "CONMEBOL", "Australia": "AFC",
    "Turkey": "UEFA", "Germany": "UEFA", "Curaçao": "CONCACAF",
    "Ivory Coast": "CAF", "Ecuador": "CONMEBOL", "Netherlands": "UEFA",
    "Japan": "AFC", "Sweden": "UEFA", "Tunisia": "CAF",
    "Belgium": "UEFA", "Egypt": "CAF", "Iran": "AFC",
    "New Zealand": "OFC", "Spain": "UEFA", "Cape Verde": "CAF",
    "Saudi Arabia": "AFC", "Uruguay": "CONMEBOL", "France": "UEFA",
    "Senegal": "CAF", "Iraq": "AFC", "Norway": "UEFA",
    "Argentina": "CONMEBOL", "Algeria": "CAF", "Austria": "UEFA",
    "Jordan": "AFC", "Portugal": "UEFA", "DR Congo": "CAF",
    "Uzbekistan": "AFC", "Colombia": "CONMEBOL", "England": "UEFA",
    "Croatia": "UEFA", "Ghana": "CAF", "Panama": "CONCACAF",
}

GROUPS = {
    "A": ["Mexico",        "South Africa",          "South Korea",   "Czechia"],
    "B": ["Canada",        "Bosnia and Herzegovina", "Qatar",         "Switzerland"],
    "C": ["Brazil",        "Morocco",               "Haiti",         "Scotland"],
    "D": ["United States", "Paraguay",              "Australia",     "Turkey"],
    "E": ["Germany",       "Curaçao",               "Ivory Coast",   "Ecuador"],
    "F": ["Netherlands",   "Japan",                 "Sweden",        "Tunisia"],
    "G": ["Belgium",       "Egypt",                 "Iran",          "New Zealand"],
    "H": ["Spain",         "Cape Verde",            "Saudi Arabia",  "Uruguay"],
    "I": ["France",        "Senegal",               "Iraq",          "Norway"],
    "J": ["Argentina",     "Algeria",               "Austria",       "Jordan"],
    "K": ["Portugal",      "DR Congo",              "Uzbekistan",    "Colombia"],
    "L": ["England",       "Croatia",               "Ghana",         "Panama"],
}

TEAMS = list(CONF.keys())
assert len(TEAMS) == 48

# R32 slot pairings for the 2026 bracket
R32_PAIRINGS = [
    ("A1","C2"), ("C1","A2"),
    ("B1","D2"), ("D1","B2"),
    ("E1","G2"), ("G1","E2"),
    ("F1","H2"), ("H1","F2"),
    ("I1","K2"), ("K1","I2"),
    ("J1","L2"), ("L1","J2"),
]


def get_elo(team):
    return ELO.get(team, 1650.0)


def load_results(url=DATA_URL, min_year=2010):
    print("Downloading match data...", end=" ", flush=True)
    with urllib.request.urlopen(url) as resp:
        raw = resp.read().decode("utf-8")
    print("done.")

    df = pd.read_csv(io.StringIO(raw), parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[df["date"].dt.year >= min_year].copy()
    df = df.rename(columns={
        "home_team": "team_a", "away_team": "team_b",
        "home_score": "goals_a", "away_score": "goals_b",
    })

    latest = df["date"].max()
    df["days_ago"] = (latest - df["date"]).dt.days
    df["weight"]   = np.exp(-DECAY_RATE * df["days_ago"])

    is_wc      = df["tournament"] == "FIFA World Cup"
    is_wc_year = df["date"].dt.year.isin([2018, 2022])
    is_euro24  = (df["tournament"] == "UEFA Euro") & (df["date"].dt.year == 2024)
    holdout    = (is_wc & is_wc_year) | is_euro24

    return df[~holdout].copy(), df[is_wc & is_wc_year].copy(), df[is_euro24].copy()
