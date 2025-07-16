import streamlit as st
import pandas as pd

# Load prediction data
player_df = pd.read_csv("hitters_2024.csv")

# Features used in model
features = ['1_WAR', 'G', 'PA', 'HR', 'R', 'RBI', 'SB', 'AVG', 'OBP', 'SLG', 'OPS',
            'avg_exit_velocity', 'avg_launch_angle', 'hard_hit_pct']
features = [f for f in features if f in player_df.columns]

# Impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(player_df[features]), columns=features)

# Load models
import joblib
clf_3 = joblib.load("model_decline_3.joblib")
clf_5 = joblib.load("model_decline_5.joblib")

# Predict
player_df['pred_decline_3'] = clf_3.predict(X)
player_df['pred_decline_5'] = clf_5.predict(X)

# --- Streamlit UI ---
st.title("MLB Player Delinquency Dashboard")

player_name = st.selectbox("Select Player", player_df['Name'].values)
decision = st.selectbox("Decision Context", ['Currently Own', 'Free Agent', 'Trade Candidate'])
contract_years = st.number_input("Contract Years", min_value=1, max_value=10, step=1)

player = player_df[player_df['Name'] == player_name].iloc[0]
decline_3 = player['pred_decline_3']
decline_5 = player['pred_decline_5']

# --- Recommendation Logic ---
recommendation = "No recommendation"

if decision == "Currently Own" and contract_years <= 3:
    if decline_3:
        recommendation = "Trade away/DFA/release"
    elif decline_5:
        recommendation = "Retain but do not extend"
    else:
        recommendation = "Extend"

elif decision == "Free Agent" and contract_years <= 3:
    if decline_3:
        recommendation = "Recommend signing to a 1-year contract or with team options"
    elif decline_5:
        recommendation = "2-3 year deal"
    else:
        recommendation = f"Sign FA {player_name} for {contract_years} years approved"

elif decision == "Trade Candidate":
    if contract_years <= 1 and decline_3:
        recommendation = "Trade recommended"
    elif 2 <= contract_years <= 3 and decline_3:
        recommendation = "Potential risk"
    elif contract_years > 3 and decline_3:
        recommendation = f"DO NOT TRADE FOR {player_name}"
    elif contract_years <= 3 and not decline_3:
        if decline_5:
            recommendation = f"Trade for {player_name} but do not extend"
        else:
            recommendation = f"Trade for {player_name} and extend"

st.markdown(f"### Recommendation: {recommendation}")
