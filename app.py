import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load('Oduti_best_model.pkl')

# -----------------------------
# TITLE
# -----------------------------
st.title("🩺 Lung Cancer Survival Prediction")
st.markdown("Provide patient details to predict survival outcome.")

st.markdown("---")

# -----------------------------
# INPUT SECTION (FIXED UI)
# -----------------------------
st.subheader("📝 Enter Patient Information")

age = st.number_input("Age", min_value=1, max_value=120, value=50)

gender = st.selectbox("Gender", ["Male", "Female"])

smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

tumor_size = st.number_input("Tumor Size", min_value=0.0, max_value=10.0, value=2.5)

fatigue = st.selectbox("Fatigue", ["No", "Yes"])

weight_loss = st.selectbox("Weight Loss", ["No", "Yes"])

# -----------------------------
# ENCODING (VERY IMPORTANT)
# -----------------------------
gender = 1 if gender == "Male" else 0

smoking = {
    "Never": 0,
    "Former": 1,
    "Current": 2
}[smoking]

fatigue = 1 if fatigue == "Yes" else 0
weight_loss = 1 if weight_loss == "Yes" else 0

# -----------------------------
# MATCH TRAINING FEATURES (FIX)
# -----------------------------
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Smoking_Status': [smoking],
    'Tumor_Size': [tumor_size],
    'Fatigue': [fatigue],
    'Weight_Loss': [weight_loss]
})

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
st.markdown("---")

if st.button("🔍 Predict"):

    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success("✅ Patient Likely to Survive")
        else:
            st.error("⚠️ Patient Less Likely to Survive")

        confidence = max(proba)
        st.write(f"Confidence: {confidence:.2f}")

        st.progress(float(confidence))

        # Chart
        fig, ax = plt.subplots()
        ax.bar(['Not Survived', 'Survived'], proba, color=['red', 'green'])
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")

        st.pyplot(fig)

    except Exception as e:
        st.error("⚠️ Prediction failed. Check model features match training data.")
        st.write(e)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("© 2026 Andrew Oduti | MSc Data Science Project")
