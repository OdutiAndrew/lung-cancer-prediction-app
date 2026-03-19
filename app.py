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
st.title("🩺 Lung Cancer Survival Prediction System")
st.markdown("Provide patient details below to generate predictions.")

st.markdown("---")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("📝 Patient Information")

age = st.number_input("Age", 1, 120, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
fatigue = st.selectbox("Fatigue", ["No", "Yes"])
weight_loss = st.selectbox("Weight Loss", ["No", "Yes"])

# -----------------------------
# ENCODING
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
# CREATE INPUT DATA (FIXED)
# -----------------------------
input_data = pd.DataFrame(columns=model.feature_names_in_)

# Fill all features with default value
input_data.loc[0] = 0

# Overwrite important ones
if 'Age' in input_data.columns:
    input_data['Age'] = age

if 'Gender' in input_data.columns:
    input_data['Gender'] = gender

if 'Smoking_Status' in input_data.columns:
    input_data['Smoking_Status'] = smoking

if 'Fatigue' in input_data.columns:
    input_data['Fatigue'] = fatigue

if 'Weight_Loss' in input_data.columns:
    input_data['Weight_Loss'] = weight_loss

# -----------------------------
# PREDICTION
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
        st.write(f"Confidence Level: {confidence:.2f}")

        st.progress(float(confidence))

        # -----------------------------
        # VISUALIZATION
        # -----------------------------
        st.subheader("📈 Prediction Probability")

        fig, ax = plt.subplots()
        ax.bar(['Not Survived', 'Survived'], proba, color=['red', 'green'])
        ax.set_ylabel("Probability")
        ax.set_title("Model Confidence")

        st.pyplot(fig)

    except Exception as e:
        st.error("⚠️ Prediction failed. Please check model compatibility.")
        st.write(e)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("© 2026 Andrew Oduti | MSc Data Science Project")
