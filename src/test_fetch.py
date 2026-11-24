from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

player = players.find_players_by_full_name("Jayson Tatum")[0]
log = playergamelog.PlayerGameLog(player_id=player["id"], season="2024-25")
df = log.get_data_frames()[0]

print(df[["GAME_DATE", "MATCHUP", "MIN", "PTS"]].head(10))
print(f"\n{len(df)} games found")