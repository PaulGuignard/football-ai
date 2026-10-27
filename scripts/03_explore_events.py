"""Exploration: what does a match look like as a sequence of events?"""

import warnings

from statsbombpy import sb

warnings.filterwarnings("ignore", message="credentials were not supplied")

# Try with the first Premier League 2015/2016 match in the list
matches = sb.matches(competition_id=2, season_id=27)
match = matches.iloc[0]
print(f"{match['home_team']} {match['home_score']} - {match['away_score']}"
      f"{match['away_team']} ({match['match_date']})")

events = sb.events(match_id=match["match_id"])
print("Shape (rows, columns):", events.shape)
print()

print("Events types:")
print(events["type"].value_counts().to_string())
print()

# "index" is the order of the event within the match
cols = ["index", "period", "minute", "second", "team", "player", "type", "location"]
print(events.sort_values("index")[cols].head(20).to_string(index=False))