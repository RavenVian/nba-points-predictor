import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

df = pd.read_csv("data/processed/features.csv", parse_dates=["GAME_DATE"])

# Only information known BEFORE the game
FEATURES = [c for c in df.columns if "_avg" in c] + [
    "GAMES_PLAYED", "HOME", "REST_DAYS", "BACK_TO_BACK"]
print("Features used:", FEATURES)

df["PTS_season_avg"] = df["PTS_season_avg"].fillna(df["PTS_avg10"])
df["OPP_PTS_ALLOWED_avg10"] = df["OPP_PTS_ALLOWED_avg10"].fillna(df["OPP_PTS_ALLOWED_avg10"].mean())

# Split by time: train on the past, test on the latest season
train = df[df["SEASON"] != "2025-26"]
test = df[df["SEASON"] == "2025-26"]
print(f"Train: {len(train):,} games | Test: {len(test):,} games\n")

# 1. Baseline
base_mae = mean_absolute_error(test["PTS"], test["PTS_avg10"])
print(f"Baseline (last-10 avg)  MAE: {base_mae:.2f} points")

# 2. Linear regression
lr = LinearRegression().fit(train[FEATURES], train["PTS"])
lr_mae = mean_absolute_error(test["PTS"], lr.predict(test[FEATURES]))
print(f"Linear regression       MAE: {lr_mae:.2f} points")

# 3. LightGBM
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, verbose=-1)
model.fit(train[FEATURES], train["PTS"])
lgb_mae = mean_absolute_error(test["PTS"], model.predict(test[FEATURES]))
print(f"LightGBM                MAE: {lgb_mae:.2f} points")

print(f"\nLightGBM beats the baseline by {base_mae - lgb_mae:.2f} points per game")

imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop 8 features LightGBM relied on:")
print(imp.head(8))