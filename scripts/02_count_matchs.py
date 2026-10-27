"""Exploration: how many matches and teams per candidate competition?

A season can be listed without being complete (e.g. only one team's matches),
so we count matches and teams before choosing the training data.
"""

import warnings

import pandas as pd
from statsbombpy import sb

# Expected warning when using open data without credentials
warnings.filterwarnings("ignore", message="credentials were not supplied")

# (readable name, competition_id, season_id): ids taken from script 01
candidates = [

	("Premier League 2015/16", 2, 27),
	("La Liga 2015/16", 11, 27),
	("Serie A 2015/16", 12, 27),
	("Bundesliga 2015/16", 9, 27),
	("Ligue 1 2015/16", 7, 27),
	("La Liga 2020/21", 11, 90),
	("Ligue 1 2022/23", 7, 235),
	("Bundesliga 2023/24", 9, 281),
	("Coupe du monde 2018", 43, 3),
	("Coupe du monde 2022", 43, 106),
	("Euro 2020", 55, 43),
	("Euro 2024", 55, 282),
]

rows = []

for name, competition_id, season_id in candidates:
	matches = sb.matches(competition_id=competition_id, season_id=season_id)
	teams = set(matches["home_team"]) | set(matches["away_team"])
	rows.append({"competition": name, "matchs": len(matches), "equipes" : len(teams)})
	print(f"{name} : ok")

print()
print(pd.DataFrame(rows).to_string(index=False))
