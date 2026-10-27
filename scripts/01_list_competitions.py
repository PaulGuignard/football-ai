"""Exploration : quelles compétitions sont disponibles en open data?"""

from statsbombpy import sb

# Récupère la liste des compétitions/saisons gratuites (un DataFrame pandas)
competitions = sb.competitions()

print("Dimensions (ligne, colonnes) :", competitions.shape)
print("Colonnes :", competitions.columns.tolist())
print()

# On n'affiche que les colonnes utiles pour choisir
cols = ["competition_id", "season_id", "competition_name", "season_name", "competition_gender"]

print(competitions[cols].to_string(index= False))
