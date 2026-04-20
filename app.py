import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Houston NBS Flood Explorer", layout="wide")

events = pd.read_csv("events.csv")
watersheds = pd.read_csv("watershed.csv")
gauges = pd.read_csv("gauge.csv")

st.title("Houston Nature-Based Solutions Flood Explorer")

event_name = st.sidebar.selectbox("Choose an event", events["name"].tolist())
event_row = events.loc[events["name"] == event_name].iloc[0]
watershed_row = watersheds.loc[watersheds["watershed_id"] == event_row["watershed_id"]].iloc[0]
gauge_row = gauges.loc[gauges["gauge_id"] == event_row["gauge_id"]].iloc[0]

solution_family = st.sidebar.selectbox("Solution family", ["Green Infrastructure", "Storage-Based", "Hybrid"])
implementation_pct = st.sidebar.slider("Implementation (%)", 0, 50, 10)

rainfall_mm = float(event_row["rainfall_mm"])
duration_hr = float(event_row["duration_hr"])
impervious_pct = float(watershed_row["impervious_pct"])

t = np.linspace(0, duration_hr * 1.5, 400)
peak_base = rainfall_mm * (0.8 + impervious_pct / 100.0)
center = duration_hr * 0.55
width = max(duration_hr / 5.0, 2.0)

Q_base = peak_base * np.exp(-((t - center) ** 2) / (2 * width ** 2))
recession = np.exp(-0.03 * np.maximum(t - center, 0))
Q_base = Q_base * recession

impl = implementation_pct / 50.0

if solution_family == "Green Infrastructure":
    runoff_red, peak_red, lag_hr = 0.3, 0.5, 0.5
elif solution_family == "Storage-Based":
    runoff_red, peak_red, lag_hr = 0.2, 0.6, 1.0
else:
    runoff_red, peak_red, lag_hr = 0.4, 0.7, 1.2

runoff_red *= impl
peak_red *= impl
lag_hr *= impl

Q_mod = Q_base * (1 - runoff_red) * (1 - peak_red)
t_mod = t + lag_hr

fig, ax = plt.subplots()
ax.plot(t, Q_base, '--', label="Baseline")
ax.plot(t_mod, Q_mod, label="With NBS")
ax.legend()
st.pyplot(fig)