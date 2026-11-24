import joblib
import numpy as np
import pandas as pd

STATS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "PRA", "PR", "PA", "RA"]
rng = np.random.default_rng(0)
all_rows = []

for stat in STATS:
    b = joblib.load(f"models/{stat}.pkl")
    tp, ta = b["test_pred"], b["test_actual"]

    # Half the test games supply the misses, the other half checks the probabilities
    idx = rng.permutation(len(tp))
    cal, ev = idx[: len(tp) // 2], idx[len(tp) // 2:]
    order = np.argsort(tp[cal])
    cp, ca = tp[cal][order], ta[cal][order]

    for i in ev:
        pred = tp[i]
        line = max(0.5, np.floor(pred) + 0.5 + rng.integers(-2, 2))  # realistic lines near the prediction
        j = np.searchsorted(cp, pred)
        lo, hi = max(0, j - 750), min(len(cp), j + 750)
        outcomes = np.clip(pred + (ca[lo:hi] - cp[lo:hi]), 0, None)
        all_rows.append({"stat": stat, "prob": (outcomes > line).mean(), "hit": ta[i] > line})

df = pd.DataFrame(all_rows)
df["bucket"] = pd.cut(df["prob"], [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)

table = df.groupby("bucket", observed=True).agg(
    games=("hit", "size"),
    said=("prob", "mean"),
    happened=("hit", "mean"),
).round(3)
print("When the model said X%, how often did it happen?\n")
print(table)

brier = df.groupby("stat").apply(lambda d: ((d["prob"] - d["hit"]) ** 2).mean()).round(3)
print("\nBrier score by stat (lower is better, 0.25 = coin flip):")
print(brier)