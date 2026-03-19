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
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Prediction", "📊 Analytics", "ℹ️ About"])

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "🏠 Home":
    st.title("🩺 Lung Cancer Survival Prediction System")

    st.markdown("""
    ### 🎯 Project Overview
    This system uses **Machine Learning** to predict lung cancer patient survival.

    ### 🚀 Key Features
    - Real-time predictions  
    - Interactive user interface  
    - Visual analytics dashboard  

    ### 🌍 Impact
    Supports healthcare professionals in making informed decisions.
    """)

    st.info("👉 Use the sidebar to navigate through the application.")

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "🔍 Prediction":

    st.title("🔍 Patient Survival Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 1, 120, 50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

    with col2:
        fatigue = st.selectbox("Fatigue", ["No", "Yes"])
        weight_loss = st.selectbox("Weight Loss", ["No", "Yes"])

    # Encoding
    gender = 1 if gender == "Male" else 0
    smoking = {"Never": 0, "Former": 1, "Current": 2}[smoking]
    fatigue = 1 if fatigue == "Yes" else 0
    weight_loss = 1 if weight_loss == "Yes" else 0

    # Create full feature input
    input_data = pd.DataFrame(columns=model.feature_names_in_)
    input_data.loc[0] = 0

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

    st.markdown("---")

    if st.button("🚀 Run Prediction"):

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Result")
            if prediction == 1:
                st.success("✅ Likely to Survive")
            else:
                st.error("⚠️ Less Likely to Survive")

            confidence = max(proba)
            st.metric("Confidence", f"{confidence:.2f}")

        with col2:
            st.subheader("📈 Probability")
            fig, ax = plt.subplots()
            ax.bar(['Not Survived', 'Survived'], proba, color=['#ff4b4b', '#4CAF50'])
            st.pyplot(fig)

# -----------------------------
# ANALYTICS PAGE
# -----------------------------
elif page == "📊 Analytics":

    st.title("📊 Model Analytics Dashboard")

    st.subheader("Feature Importance")

    try:
        importance = model.coef_[0]
        features = model.feature_names_in_

        fig, ax = plt.subplots()
        ax.barh(features, importance, color='skyblue')
        ax.set_title("Feature Importance")

        st.pyplot(fig)

    except:
        st.warning("Feature importance not available for this model.")

    st.subheader("Model Performance (Example)")

    models = ['Logistic Regression', 'Random Forest', 'Decision Tree']
    scores = [0.75, 0.72, 0.69]

    fig2, ax2 = plt.subplots()
    ax2.bar(models, scores, color='orange')
    ax2.set_ylabel("Accuracy")

    st.pyplot(fig2)

# -----------------------------
# ABOUT PAGE
# -----------------------------
elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""
    ### 📚 Course
    Data Mining, Modelling and Analytics  

    ### 🎯 Objective
    Predict lung cancer survival using machine learning models.

    ### ⚙️ Technologies Used
    - Python  
    - Scikit-learn  
    - Streamlit  

    ### ⚠️ Disclaimer
    This application is for academic purposes only and should not replace professional medical advice.

    ### 👨‍💻 Developer
    Andrew Oduti
    """)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("© 2026 Andrew Oduti | MSc Data Science")
