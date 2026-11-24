import argparse
import joblib
import numpy as np
import pandas as pd


def load_history():
    df = pd.read_csv("data/processed/player_games_clean.csv", parse_dates=["GAME_DATE"])
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    df["PR"] = df["PTS"] + df["REB"]
    df["PA"] = df["PTS"] + df["AST"]
    df["RA"] = df["REB"] + df["AST"]
    return df.sort_values(["PLAYER_ID", "GAME_DATE"])


def opponent_defence(opp):
    t = pd.read_csv("data/raw/team_games.csv", parse_dates=["GAME_DATE"])
    other = t[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].rename(
        columns={"TEAM_ABBREVIATION": "OTHER_TEAM", "PTS": "PTS_ALLOWED"})
    t = t.merge(other, on="GAME_ID")
    t = t[(t["TEAM_ABBREVIATION"] == opp) & (t["OTHER_TEAM"] != opp)].sort_values("GAME_DATE")
    if t.empty:
        raise ValueError(f"Unknown team '{opp}'. Use a code like BOS, LAL, MIA.")
    return t["PTS_ALLOWED"].tail(10).mean()


def find_player(hist, name):
    p = hist[hist["PLAYER_NAME"].str.lower() == name.lower()]
    if p.empty:
        p = hist[hist["PLAYER_NAME"].str.contains(name, case=False)]
    if p.empty:
        raise ValueError(f"No player matching '{name}'")
    if p["PLAYER_ID"].nunique() > 1:
        raise ValueError(f"Several players match: {sorted(p['PLAYER_NAME'].unique())}")
    return p


def build_row(p, opponent, home, rest_days, features):
    this_season = p[p["SEASON"] == p["SEASON"].iloc[-1]]
    row = {}
    for f in features:
        if f == "OPP_PTS_ALLOWED_avg10":
            row[f] = opponent_defence(opponent)
        elif f.endswith("_season_avg"):
            row[f] = this_season[f.replace("_season_avg", "")].mean()
        elif "_avg" in f:
            col, w = f.rsplit("_avg", 1)
            row[f] = p[col].tail(int(w)).mean()
    row["GAMES_PLAYED"] = len(this_season)
    row["HOME"] = int(home)
    row["REST_DAYS"] = min(rest_days, 7)
    row["BACK_TO_BACK"] = int(rest_days == 1)
    return pd.DataFrame([row])[features]


def predict(player, stat, line, opponent, home, rest_days=2):
    bundle = joblib.load(f"models/{stat}.pkl")
    p = find_player(load_history(), player)
    X = build_row(p, opponent.upper(), home, rest_days, bundle["features"])
    pred = float(bundle["model"].predict(X)[0])

    # How wrong was the model in the 1,500 test games with the most similar predictions?
    tp, ta = bundle["test_pred"], bundle["test_actual"]
    nearest = np.argsort(np.abs(tp - pred))[:1500]
    outcomes = np.clip(pred + (ta[nearest] - tp[nearest]), 0, None)

    return {
        "player": p["PLAYER_NAME"].iloc[-1],
        "last_game": p["GAME_DATE"].iloc[-1].date(),
        "expected": pred,
        "prob_over": (outcomes > line).mean(),
        "low": np.percentile(outcomes, 10),
        "high": np.percentile(outcomes, 90),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("player")
    ap.add_argument("stat", help="PTS REB AST FG3M STL BLK TOV PRA PR PA RA")
    ap.add_argument("line", type=float)
    ap.add_argument("opponent", help="team code, e.g. MIA")
    ap.add_argument("--home", action="store_true")
    ap.add_argument("--rest", type=int, default=2)
    a = ap.parse_args()

    r = predict(a.player, a.stat.upper(), a.line, a.opponent, a.home, a.rest)
    print(f"\n{r['player']} vs {a.opponent.upper()} ({'home' if a.home else 'away'}), {a.stat.upper()}")
    print(f"Expected:        {r['expected']:.1f}")
    print(f"P(over {a.line}):  {r['prob_over']:.0%}")
    print(f"80% range:       {r['low']:.0f} to {r['high']:.0f}")
    print(f"(based on games up to {r['last_game']})")