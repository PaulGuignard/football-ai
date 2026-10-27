"""Exploration: which competitions are available in StatsBomb open data?"""

from statsbombpy import sb

# Download the list of free competition/seasons as a pandas DataFrame
competitions = sb.competitions()

print("Dimensions (ligne, colonnes) :", competitions.shape)
print("Colonnes :", competitions.columns.tolist())
print()

# Only show the columns needed to choose a competition
cols = ["competition_id", "season_id", "competition_name", "season_name", "competition_gender"]

print(competitions[cols].to_string(index= False))
