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
# STYLE (UI IMPROVEMENT)
# -----------------------------
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        height: 3em;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load('Oduti_best_model.pkl')

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Prediction", "📊 Analytics", "ℹ️ About"])

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "🏠 Home":
    st.title("🩺 Lung Cancer Survival Prediction System")

    st.markdown("""
    ### 🎯 Overview
    This application predicts lung cancer patient survival using machine learning.

    ### 🚀 Features
    - Real-time prediction  
    - Interactive dashboard  
    - Visual analytics  

    ### 🌍 Impact
    Supports healthcare decision-making.
    """)

    st.info("👉 Use the sidebar to navigate")

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "🔍 Prediction":

    st.title("🔍 Patient Survival Prediction")
    st.markdown("### 📝 Enter Patient Details")

    # INPUTS
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=None, placeholder="Enter age")

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col3:
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

    col4, col5 = st.columns(2)

    with col4:
        fatigue = st.selectbox("Fatigue", ["No", "Yes"])

    with col5:
        weight_loss = st.selectbox("Weight Loss", ["No", "Yes"])

    st.markdown("---")

    # ENCODING
    gender = 1 if gender == "Male" else 0
    smoking = {"Never": 0, "Former": 1, "Current": 2}[smoking]
    fatigue = 1 if fatigue == "Yes" else 0
    weight_loss = 1 if weight_loss == "Yes" else 0

    # CREATE FULL INPUT (MATCH MODEL)
    input_data = pd.DataFrame(columns=model.feature_names_in_)
    input_data.loc[0] = 0

    if 'Age' in input_data.columns:
        input_data['Age'] = age if age else 0
    if 'Gender' in input_data.columns:
        input_data['Gender'] = gender
    if 'Smoking_Status' in input_data.columns:
        input_data['Smoking_Status'] = smoking
    if 'Fatigue' in input_data.columns:
        input_data['Fatigue'] = fatigue
    if 'Weight_Loss' in input_data.columns:
        input_data['Weight_Loss'] = weight_loss

    # BUTTON
    predict_btn = st.button("🚀 Run Prediction")

    if predict_btn:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        st.markdown("## 📊 Results Dashboard")

        col1, col2 = st.columns([1, 2])

        with col1:
            if prediction == 1:
                st.success("✅ Likely to Survive")
            else:
                st.error("⚠️ Less Likely to Survive")

            confidence = max(proba)
            st.metric("Confidence Level", f"{confidence:.2f}")

        with col2:
            fig, ax = plt.subplots()
            ax.bar(['Not Survived', 'Survived'], proba, color=['#FF4B4B', '#00C49A'])
            ax.set_ylabel("Probability")
            ax.set_title("Prediction Confidence")
            st.pyplot(fig)

# -----------------------------
# ANALYTICS PAGE
# -----------------------------
elif page == "📊 Analytics":

    st.title("📊 Model Analytics")

    try:
        importance = model.coef_[0]
        features = model.feature_names_in_

        fig, ax = plt.subplots()
        ax.barh(features, importance, color='skyblue')
        ax.set_title("Feature Importance")

        st.pyplot(fig)

    except:
        st.warning("Feature importance not available")

    st.subheader("Model Comparison")

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

    st.title("ℹ️ About")

    st.markdown("""
    ### 📚 Course
    Data Mining, Modelling and Analytics  

    ### 🎯 Objective
    Predict lung cancer survival using machine learning  

    ### ⚙️ Tools
    - Python  
    - Scikit-learn  
    - Streamlit  

    ### ⚠️ Disclaimer
    Academic use only  

    ### 👨‍💻 Developer
    Andrew Oduti  
    """)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("© 2026 Andrew Oduti | MSc Data Science")
