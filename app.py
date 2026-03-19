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
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load('Oduti_best_model.pkl')

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Analytics", "About"])

# -----------------------------
# HOME
# -----------------------------
if page == "Home":
    st.title("🩺 Lung Cancer Survival Prediction System")

    st.markdown("""
    ### 🎯 Objective
    Predict patient survival using machine learning.

    ### 🚀 Features
    - Real-time prediction
    - Interactive dashboard
    - Model analytics & visualization

    ### 👨‍⚕️ Use Case
    Supports healthcare decision-making.
    """)

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "Prediction":
    st.title("📊 Patient Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 1, 120, 50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

    with col2:
        tumor_size = st.slider("Tumor Size", 0.0, 10.0, 2.5)
        fatigue = st.selectbox("Fatigue", [0, 1])
        weight_loss = st.selectbox("Weight Loss", [0, 1])

    # Encoding
    gender = 1 if gender == "Male" else 0
    smoking = {"Never": 0, "Former": 1, "Current": 2}[smoking]

    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Smoking_Status': [smoking],
        'Tumor_Size': [tumor_size],
        'Fatigue': [fatigue],
        'Weight_Loss': [weight_loss]
    })

    st.subheader("📌 Input Summary")
    st.dataframe(input_data)

    if st.button("🔍 Predict"):
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        st.subheader("📈 Prediction Result")

        if prediction == 1:
            st.success("✅ Survived")
        else:
            st.error("⚠️ Not Survived")

        st.write(f"Confidence: {max(proba):.2f}")
        st.progress(float(max(proba)))

        # 📊 Probability Chart
        st.subheader("📊 Prediction Probability")

        fig, ax = plt.subplots()
        ax.bar(['Not Survived', 'Survived'], proba, color=['red', 'green'])
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

# -----------------------------
# ANALYTICS PAGE (🔥 NEW)
# -----------------------------
elif page == "Analytics":
    st.title("📊 Model Insights & Analytics")

    st.subheader("Feature Importance")

    # Works for Logistic Regression
    try:
        importance = model.coef_[0]
        features = ['Age', 'Gender', 'Smoking_Status', 'Tumor_Size', 'Fatigue', 'Weight_Loss']

        fig, ax = plt.subplots()
        ax.barh(features, importance)
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    except:
        st.warning("Feature importance not available for this model.")

    # Dummy accuracy comparison (you can replace with real values)
    st.subheader("Model Comparison")

    models = ['Logistic Regression', 'Random Forest', 'Decision Tree', 'KNN']
    accuracy = [0.75, 0.71, 0.69, 0.69]

    fig2, ax2 = plt.subplots()
    ax2.bar(models, accuracy)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Model Performance Comparison")
    st.pyplot(fig2)

# -----------------------------
# ABOUT
# -----------------------------
elif page == "About":
    st.title("ℹ️ About")

    st.markdown("""
    ### 📚 Project
    Data Mining, Modelling and Analytics

    ### ⚙️ Model
    Logistic Regression for survival prediction

    ### ⚠️ Disclaimer
    This is an academic project and should not replace medical advice.

    ### 👨‍💻 Developer
    Andrew Oduti
    """)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("© 2026 Andrew Oduti | MSc Data Science")
