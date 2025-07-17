import streamlit as st
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from fpdf import FPDF
from io import BytesIO

# --- Load Data & Models ---
player_df = pd.read_csv("hitters_2024.csv")
clf_3, features_3 = joblib.load("model_decline_3.joblib")
clf_5, features_5 = joblib.load("model_decline_5.joblib")

# --- Impute ---
imputer = SimpleImputer(strategy="mean")
X_3 = pd.DataFrame(imputer.fit_transform(player_df[features_3]), columns=features_3)
X_5 = pd.DataFrame(imputer.fit_transform(player_df[features_5]), columns=features_5)

# --- Predict Probabilities ---
player_df['prob_decline_3'] = clf_3.predict_proba(X_3)[:, 1]
player_df['prob_decline_5'] = clf_5.predict_proba(X_5)[:, 1]

# --- Recommendation Logic (your full logic here) ---
def generate_recommendation(decision, avg_WAR_career, p3, p5, years):
    def between(x, low, high): return low <= x < high

    if decision == "Free Agent":
        if avg_WAR_career >= 5:
            if p3 > 0.7:
                return 'Consider signing player to a 3-4 year contract with higher AAV'
            elif p5 > 0.7:
                return 'Consider signing player to a 5-6 year contract with higher AAV'
            else:
                return 'Consider signing player for 8 years or through age 35 season'
        elif between(avg_WAR_career, 4, 5):
            if p3 > 0.6:
                return 'Consider signing player to a 2-3 year contract with higher AAV'
            elif p5 > 0.6:
                return 'Consider signing player to a 4-5 year contract with higher AAV'
            else:
                return 'Consider signing player for 6 years or through age 32 season'
        elif between(avg_WAR_career, 3, 4):
            if p3 > 0.5:
                return 'Consider signing player to a 2 year contract with club options'
            elif p5 > 0.5:
                return 'Consider signing player to a 3 year contract with club options'
            else:
                return 'Consider signing player for 5 years with multiple club or player option years'
        elif between(avg_WAR_career, 2, 3):
            if p3 > 0.5:
                return 'Consider signing player to a 1 year contract with higher AAV or 2 year deal with a club option'
            elif p5 > 0.5:
                return 'Consider signing player to a 5-6 year contract with higher AAV'
            else:
                return 'Consider signing player for 8 years or through age 35 season'
        elif between(avg_WAR_career, 1, 2):
            if p3 > 0.5:
                return 'Consider signing a different player'
            elif p5 > 0.5:
                return 'Consider signing player to a 1 year deal with a club option for second'
            else:
                return 'Consider signing player to a team friendly deal with minimal AAV and club/player options'
        else:
            return 'Consider signing a different player'

    elif decision == "Trade Candidate":
        if avg_WAR_career >= 5:
            if years <= 3 and p3 < 0.7:
                return 'Consider trade, club option years preferred'
            elif years > 3 and p3 > 0.7:
                return 'Do not take on player without salary relief'
            elif years <= 5 and p3 < 0.7 and p5 < 0.7:
                return 'Make trade for player'
            elif years > 5 and p3 < 0.7 and p5 > 0.7:
                return 'Consider making trade with option years or salary relief added'
            elif p3 < 0.7 and p5 < 0.7:
                return 'Make trade, player retains value for a long time'
            else:
                return 'Consider trading for a different player'
        elif between(avg_WAR_career, 4, 5) or between(avg_WAR_career, 3, 4) or between(avg_WAR_career, 2, 3):
            if years < 3 and p3 < 0.5:
                return 'Consider trade, club option years preferred'
            elif years >= 3 and p3 > 0.5:
                return 'Do not take on player without salary relief'
            elif years < 5 and p3 < 0.6 and p5 < 0.6:
                return 'Make trade for player'
            elif years >= 5 and p3 < 0.6 and p5 > 0.6:
                return 'Consider making trade with option years or salary relief added'
            elif p3 < 0.6 and p5 < 0.6:
                return 'Make trade, player retains value for a long time'
            else:
                return 'Consider trading for a different player'
        elif between(avg_WAR_career, 1, 2) or avg_WAR_career < 1:
            if years < 3 and p3 < 0.5:
                return 'Player can be valuable for 1-2 seasons. consider trade'
            else:
                return 'Do not trade for player'

    elif decision == "Currently Own":
        if avg_WAR_career >= 5:
            if years == 1 and p3 > 0.7:
                return 'Trade player at deadline'
            elif years == 1 and p3 < 0.7 and p5 > 0.7:
                return 'Consider short extension if unable trade at deadline'
            elif years == 1 and p3 < 0.7 and p5 < 0.7:
                return 'Consider extending player, attempt to resign in offseason'
            elif years <= 3 and p3 < 0.7:
                return 'Retain player'
            elif years > 3 and p3 > 0.7:
                return 'Retain but consider trades for salary relief'
            elif years <= 5 and p3 < 0.7 and p5 < 0.7:
                return 'Retain player'
            elif years > 5 and p3 < 0.7 and p5 > 0.7:
                return 'Retain, but consider trades in future, do not extend'
            elif p3 < 0.7 and p5 < 0.7:
                return 'Franchise player, consider a long term extension'
        elif between(avg_WAR_career, 4, 5) or between(avg_WAR_career, 3, 4) or between(avg_WAR_career, 2, 3):
            if years == 1:
                return 'Trade player at deadline'
            elif years < 3 and p3 < 0.5:
                return 'Retain player'
            elif years >= 3 and p3 > 0.5:
                return 'Do not take on player without salary relief'
            elif years < 5 and p3 < 0.6 and p5 < 0.6:
                return 'Make trade for player'
            elif years >= 5 and p3 < 0.6 and p5 > 0.6:
                return 'Consider making trade with options or relief'
            elif p3 < 0.6 and p5 < 0.6:
                return 'Make trade, player retains value'
        elif between(avg_WAR_career, 1, 2) or avg_WAR_career < 1:
            if years == 1:
                return 'Trade player at deadline'
            elif years < 3 and p3 < 0.5:
                return 'Player can be valuable for 1-2 seasons. consider trade'
            else:
                return 'Do not trade for player'

    return "No recommendation"


# --- PDF Export Function ---
def generate_pdf_report(player_name, decision, contract_years, avg_WAR_career, prob_3, prob_5, recommendation, similar_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="MLB Player Contract Recommendation", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Player: {player_name}", ln=True)
    pdf.cell(200, 10, txt=f"Decision Context: {decision}", ln=True)
    pdf.cell(200, 10, txt=f"Contract Years: {contract_years}", ln=True)
    pdf.cell(200, 10, txt=f"Career Avg WAR: {avg_WAR_career:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"3-Year Similar WAR Probability: {prob_3:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"5-Year Similar WAR Probability: {prob_5:.2f}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Recommendation: {recommendation}")
    pdf.ln(10)

    # Color bar visualization for probabilities
    def add_prob_bar(label, prob, y_pos):
        if prob < 0.3:
        color = (255, 0, 0)     # Red
        elif prob > 0.7:
        color = (0, 200, 0)     # Green
        else:
        color = (160, 160, 160) # Gray
        bar_length = prob * 100  # Scale to 100 px width

        pdf.set_xy(10, y_pos)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(40, 10, f"{label}: {prob:.2f}")
        pdf.set_fill_color(*color)
        pdf.set_xy(60, y_pos)
        pdf.cell(bar_length, 10, "", 0, 0, '', True)

# Add bars for both predictions
        y_start = pdf.get_y() + 2
        add_prob_bar("3-Year Decline", prob_3, y_start)
        pdf.ln(12)
        add_prob_bar("5-Year Decline", prob_5, pdf.get_y() + 2)
        pdf.ln(15)


    if not similar_df.empty:
        pdf.cell(200, 10, txt="Similar Players with Alternative Recommendations:", ln=True)
        pdf.ln(5)
        for _, row in similar_df.iterrows():
           text = (
                f"{row['Name']} | avg_WAR_career: {row['avg_WAR_career']:.2f} | "
                f"3-Year Similar WAR Probability: {row['prob_decline_3']:.2f} | "
                f"5-Year Similar WAR Probability: {row['prob_decline_5']:.2f} | "
                f"Recommendation: {row['Rec']}"
        )
        pdf.multi_cell(0, 10, txt=text)
        pdf.ln(1)
    else:
        pdf.cell(200, 10, txt="No similar players with alternative recommendations found.", ln=True)

    # Output to memory as bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return BytesIO(pdf_bytes)


# --- Streamlit UI ---
st.title("MLB Player Delinquency Dashboard")
player_name = st.selectbox("Select Player", player_df['Name'].values)
decision = st.selectbox("Decision Context", ['Currently Own', 'Free Agent', 'Trade Candidate'])
contract_years = st.number_input("Contract Years", min_value=1, max_value=10, step=1)

# --- Lookup & Recommend ---
player = player_df[player_df['Name'] == player_name].iloc[0]
prob_3 = player['prob_decline_3']
prob_5 = player['prob_decline_5']
avg_WAR_career = player.get('avg_WAR_career', 0)
recommendation = generate_recommendation(decision, avg_WAR_career, prob_3, prob_5, contract_years)

# --- Similar Players (±1 WAR) with Different Rec ---
similar_players = player_df[
    (player_df['avg_WAR_career'] >= avg_WAR_career - 1) &
    (player_df['avg_WAR_career'] <= avg_WAR_career + 1) &
    (player_df['Name'] != player_name)
].copy()
similar_players['Rec'] = similar_players.apply(
    lambda row: generate_recommendation(
        decision, row['avg_WAR_career'], row['prob_decline_3'], row['prob_decline_5'], contract_years
    ), axis=1
)
matching_recs = similar_players[similar_players['Rec'] != recommendation]

# --- Display Player Report ---
st.subheader("Player Report")
st.write(f"**Decision Context:** {decision}")
st.write(f"**3-Year Similar WAR Probability:** {prob_3:.2f}")
st.write(f"**5-Year Similar WAR Probability:** {prob_5:.2f}")
st.write(f"**Career Avg WAR:** {avg_WAR_career:.2f}")
st.markdown(f"### **Recommendation:** {recommendation}")

import plotly.graph_objects as go

def color_for_prob(p):
    if p < 0.3:
        return "red"
    elif p > 0.7:
        return "green"
    else:
        return "gray"

bar_chart = go.Figure(data=[
    go.Bar(
        x=["3-Year Decline", "5-Year Decline"],
        y=[prob_3, prob_5],
        marker_color=[color_for_prob(prob_3), color_for_prob(prob_5)],
        text=[f"{prob_3:.2f}", f"{prob_5:.2f}"],
        textposition="outside"
    )
])
bar_chart.update_layout(
    yaxis=dict(range=[0, 1]),
    title="Decline Probabilities (Color Coded)",
    showlegend=False
)
st.plotly_chart(bar_chart)

# --- Similar Player Suggestions ---
if not matching_recs.empty:
    st.subheader("Alternative Options – Similar Players")
    similar_display = matching_recs[['Name', 'avg_WAR_career', 'prob_decline_3', 'prob_decline_5', 'Rec']].rename(
    columns={
        'avg_WAR_career': 'Avg WAR',
        'prob_decline_3': '3Y Prob',
        'prob_decline_5': '5Y Prob',
        'Rec': 'Recommendation'
    }
)
    st.dataframe(similar_display)

else:
    st.info("No similar players with alternative recommendations found.")

# --- PDF Export Button ---
pdf_bytes = generate_pdf_report(player_name, decision, contract_years, avg_WAR_career, prob_3, prob_5, recommendation, matching_recs)
st.download_button(
    label="📄 Download Player Report as PDF",
    data=pdf_bytes,
    file_name=f"{player_name.replace(' ', '_')}_recommendation.pdf",
    mime="application/pdf"
)
