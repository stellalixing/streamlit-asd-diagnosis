# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:16:51 2024

@author: 14915
"""

import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import normalize

# ===============================
# Load the model and scaler
# ===============================
best_model = joblib.load("ASD_TD_best_model.pkl")
scaler = joblib.load("ASD_TD_scaler.pkl")

# ===============================
# Feature names
# ===============================
feature_names = [
    "GMV ANG.l",
    "GMV TPOsup.r",
    "GMV TPOmid.r",
    "z-FC aSTG.l to PP.r",
    "z-FC aSTG.l to PP.l",
    "z-FC aSTG.l to FO.r",
    "z-FC aSTG.l to pSTG.l",
    "z-FC aSTG.l to Ver45",
    "z-FC aSTG.r to pSTG.l",
    "z-FC aSTG.r to FO.r",
    "z-FC aSTG.r to TP.r",
    "z-FC HG.l to Pallidum.l",
    "z-FC HG.l to aITG.l",
    "z-FC pSTG.r to PP.l",
    "z-FC pSTG.r to pSTG.l",
    "z-FC pSTG.r to iLOC.r",
    "z-FC TP.r to TP.l",
    "z-FC PC to pITG.l",
    "z-FC PC to pITG.r",
    "z-FC PC to sLOC.r",
    "z-FC PC to aITG.r",
    "z-FC toMTG.r to Forb.l",
    "z-FC PT.l to IC.r",
    "z-FC PT.l to SCC.r",
    "z-FC Ver3 to Caudate.l",
    "MD SCP.r",
    "MD CgC.r",
    "AD SCR.l",
    "AD CgC.l",
    "AD CgC.r"
]

# ===============================
# UI
# ===============================
st.title("ASD Diagnostic Prediction Model")
st.write("Enter feature data for prediction")

# ===============================
# Input features
# ===============================
features = []
for name in feature_names:
    value = st.number_input(name, format="%.12f")
    features.append(value)

new_sample = np.array([features])

# ===============================
# Prediction
# ===============================
if st.button("Predict"):
    new_sample_scaled = scaler.transform(new_sample)
    new_sample_normalized = normalize(new_sample_scaled, norm="l2")

    prediction = best_model.predict(new_sample_normalized)
    prediction_proba = best_model.predict_proba(new_sample_normalized)

    if prediction[0] == 0:
        st.write("Prediction: TD")
    else:
        st.write("Prediction: ASD")

    st.write(f"Probability of TD: {prediction_proba[0][0]:.4f}")
    st.write(f"Probability of ASD: {prediction_proba[0][1]:.4f}")

