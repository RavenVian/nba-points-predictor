import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

STATS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "PRA", "PR", "PA", "RA"]

df = pd.read_csv("data/processed/features.csv", parse_dates=["GAME_DATE"])
FEATURES = [c for c in df.columns if "_avg" in c] + [
    "GAMES_PLAYED", "HOME", "REST_DAYS", "BACK_TO_BACK"]

train = df[df["SEASON"] != "2025-26"]
test = df[df["SEASON"] == "2025-26"]
params = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=50, verbose=-1)

results = []
for stat in STATS:
    print(f"Training {stat}...")
    # Honest test: train on the past, predict the unseen season
    m = lgb.LGBMRegressor(**params).fit(train[FEATURES], train[stat])
    test_pred = m.predict(test[FEATURES])

    # Final model: retrain on everything
    final = lgb.LGBMRegressor(**params).fit(df[FEATURES], df[stat])

    joblib.dump({
        "model": final,
        "features": FEATURES,
        "test_pred": test_pred,              # what the model predicted
        "test_actual": test[stat].values,    # what really happened
    }, f"models/{stat}.pkl")

    results.append({
        "stat": stat,
        "baseline_MAE": round(mean_absolute_error(test[stat], test[f"{stat}_avg10"]), 2),
        "model_MAE": round(mean_absolute_error(test[stat], test_pred), 2),
    })

print("\n", pd.DataFrame(results).to_string(index=False))