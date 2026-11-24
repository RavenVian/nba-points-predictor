import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

def fetch(kind):
    # kind "P" = player game logs, "T" = team game logs
    frames = []
    for season in SEASONS:
        print(f"Downloading {season} ({kind})...")
        log = leaguegamelog.LeagueGameLog(
            season=season,
            player_or_team_abbreviation=kind,
            season_type_all_star="Regular Season",
            timeout=60,
        )
        df = log.get_data_frames()[0]
        df["SEASON"] = season
        frames.append(df)
        time.sleep(1.5)  # be polite to the NBA's servers
    return pd.concat(frames, ignore_index=True)

players = fetch("P")
players.to_csv("data/raw/player_games.csv", index=False)
print(f"Saved {len(players):,} player-games")

teams = fetch("T")
teams.to_csv("data/raw/team_games.csv", index=False)
print(f"Saved {len(teams):,} team-games")