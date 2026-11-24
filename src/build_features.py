import pandas as pd

STATS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "PRA", "PR", "PA", "RA"]
USAGE = ["MIN", "FGA", "FTA", "FG3A"]
WINDOWS = [3, 5, 10]

df = pd.read_csv("data/processed/player_games_clean.csv", parse_dates=["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

# Combo stats
df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
df["PR"] = df["PTS"] + df["REB"]
df["PA"] = df["PTS"] + df["AST"]
df["RA"] = df["REB"] + df["AST"]

# Rest days: first game of a season = fully rested, cap at 7
first_of_season = df["SEASON"] != df.groupby("PLAYER_ID")["SEASON"].shift()
df.loc[first_of_season, "REST_DAYS"] = 7
df["REST_DAYS"] = df["REST_DAYS"].fillna(7).clip(upper=7)

# Rolling and season averages, using PAST games only (shift(1))
new = {}
player = df["PLAYER_ID"]
season_keys = [df["PLAYER_ID"], df["SEASON"]]
for col in STATS + USAGE:
    past = df.groupby("PLAYER_ID")[col].shift(1)
    for w in WINDOWS:
        new[f"{col}_avg{w}"] = past.groupby(player).transform(
            lambda s: s.rolling(w, min_periods=1).mean())
    past_in_season = df.groupby(["PLAYER_ID", "SEASON"])[col].shift(1)
    new[f"{col}_season_avg"] = past_in_season.groupby(season_keys).transform(
        lambda s: s.expanding().mean())
df = pd.concat([df, pd.DataFrame(new)], axis=1)
df["GAMES_PLAYED"] = df.groupby(["PLAYER_ID", "SEASON"]).cumcount()

# Opponent defence: points they allowed over their last 10 games
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

# Drop each player's very first game (no history)
df = df.dropna(subset=["PTS_avg3"])

df.to_csv("data/processed/features.csv", index=False)
print(f"Saved {len(df):,} rows with {df.shape[1]} columns")
print("Stats ready to model:", STATS)