import streamlit as st
import matplotlib.pyplot as plt
from src.predict import predict, load_history

st.set_page_config(page_title="NBA Stat Predictor", page_icon="🏀")
st.title("🏀 NBA Player Stat Predictor")
st.caption("Predicts a player's next-game stat and the chance of going over a line.")

STATS = {"Points": "PTS", "Rebounds": "REB", "Assists": "AST", "3-Pointers Made": "FG3M",
         "Steals": "STL", "Blocks": "BLK", "Turnovers": "TOV", "Pts+Reb+Ast": "PRA",
         "Pts+Reb": "PR", "Pts+Ast": "PA", "Reb+Ast": "RA"}

@st.cache_data
def get_history():
    return load_history()

hist = get_history()
latest = hist[hist["SEASON"] == hist["SEASON"].max()]
players = sorted(latest["PLAYER_NAME"].unique())
teams = sorted(hist["OPPONENT"].unique())

col1, col2 = st.columns(2)
player = col1.selectbox("Player", players, index=players.index("Jayson Tatum") if "Jayson Tatum" in players else 0)
stat_label = col2.selectbox("Stat", list(STATS))
opponent = col1.selectbox("Opponent", teams)
line = col2.number_input("Line (over)", min_value=0.0, value=5.5, step=1.0)
home = col1.radio("Venue", ["Home", "Away"], horizontal=True) == "Home"
rest = col2.slider("Days of rest", 1, 7, 2)

if st.button("Predict", type="primary"):
    stat = STATS[stat_label]
    r = predict(player, stat, line, opponent, home, rest)

    a, b, c = st.columns(3)
    a.metric("Expected", f"{r['expected']:.1f}")
    b.metric(f"Chance over {line}", f"{r['prob_over']:.0%}")
    c.metric("80% range", f"{r['low']:.0f} – {r['high']:.0f}")
    st.caption(f"Based on games up to {r['last_game']}.")

    recent = hist[hist["PLAYER_NAME"] == player].tail(15)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(recent)), recent[stat], color="#4C78A8")
    ax.axhline(line, color="#E45756", linestyle="--", label=f"Line {line}")
    ax.set_xticks(range(len(recent)))
    ax.set_xticklabels(recent["GAME_DATE"].dt.strftime("%d %b"), rotation=45, fontsize=7)
    ax.set_title(f"{player}: {stat_label}, last {len(recent)} games")
    ax.legend()
    st.pyplot(fig)