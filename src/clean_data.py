import pandas as pd

df = pd.read_csv("data/raw/player_games.csv")

# Dates and ordering
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

# Home/away and opponent, from MATCHUP like "BOS vs. CHA" or "BOS @ NYK"
df["HOME"] = df["MATCHUP"].str.contains("vs.").astype(int)
df["OPPONENT"] = df["MATCHUP"].str.split(" ").str[-1]

# Rest days since the player's previous game (1 = back-to-back)
df["REST_DAYS"] = df.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days
df["BACK_TO_BACK"] = (df["REST_DAYS"] == 1).astype(int)

# Drop garbage-time cameos (under 5 minutes)
before = len(df)
df = df[df["MIN"] >= 5]
print(f"Dropped {before - len(df):,} games under 5 minutes")

keep = ["SEASON", "GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME",
        "TEAM_ABBREVIATION", "OPPONENT", "HOME", "REST_DAYS", "BACK_TO_BACK",
        "MIN", "FGA", "FG3A", "FG3M", "FTA", "REB", "AST", "STL", "BLK", "TOV", "PTS"]
df = df[keep]

df.to_csv("data/processed/player_games_clean.csv", index=False)
print(f"Saved {len(df):,} rows, {df['PLAYER_ID'].nunique():,} players")
print(df[["PTS", "MIN", "REST_DAYS"]].describe().round(1))