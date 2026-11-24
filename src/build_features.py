import pandas as pd

df = pd.read_csv("data/processed/player_games_clean.csv", parse_dates=["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

# Fix rest days: first game of a season = fully rested, cap at 7
first_of_season = df["SEASON"] != df.groupby("PLAYER_ID")["SEASON"].shift()
df.loc[first_of_season, "REST_DAYS"] = 7
df["REST_DAYS"] = df["REST_DAYS"].fillna(7).clip(upper=7)

# Recent form: averages over the last 3, 5 and 10 games (shift(1) = past games only)
g = df.groupby("PLAYER_ID")
for col in ["PTS", "MIN", "FGA", "FTA", "FG3A"]:
    for w in [3, 5, 10]:
        df[f"{col}_avg{w}"] = g[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

# Season-to-date average and games played this season
gs = df.groupby(["PLAYER_ID", "SEASON"])
df["PTS_season_avg"] = gs["PTS"].transform(lambda s: s.shift(1).expanding().mean())
df["GAMES_PLAYED"] = gs.cumcount()

# Opponent defence: points the opponent allowed over their last 10 games
t = pd.read_csv("data/raw/team_games.csv", parse_dates=["GAME_DATE"])
other = t[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].rename(
    columns={"TEAM_ABBREVIATION": "OTHER_TEAM", "PTS": "PTS_ALLOWED"})
t = t.merge(other, on="GAME_ID")
t = t[t["TEAM_ABBREVIATION"] != t["OTHER_TEAM"]].sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
t["OPP_PTS_ALLOWED_avg10"] = t.groupby("TEAM_ABBREVIATION")["PTS_ALLOWED"].transform(
    lambda s: s.shift(1).rolling(10, min_periods=1).mean())
opp = t[["GAME_ID", "TEAM_ABBREVIATION", "OPP_PTS_ALLOWED_avg10"]].rename(
    columns={"TEAM_ABBREVIATION": "OPPONENT"})
df = df.merge(opp, on=["GAME_ID", "OPPONENT"], how="left")

# Drop each player's very first game (no history to learn from)
df = df.dropna(subset=["PTS_avg3"])

df.to_csv("data/processed/features.csv", index=False)
print(f"Saved {len(df):,} rows with {df.shape[1]} columns")
print("\nHow strongly each feature tracks points (1 = perfectly):")
print(df.corr(numeric_only=True)["PTS"].sort_values(ascending=False).round(2).head(12))