elif page == "🔍 Prediction":

    st.title("🔍 Patient Survival Prediction")

    st.markdown("### 📝 Enter Patient Details")

    # Better layout using containers
    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=None, placeholder="Enter age")

        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])

        with col3:
            smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

    with st.container():
        col4, col5 = st.columns(2)

        with col4:
            fatigue = st.selectbox("Fatigue", ["No", "Yes"])

        with col5:
            weight_loss = st.selectbox("Weight Loss", ["No", "Yes"])

    st.markdown("---")

    # Encoding
    gender = 1 if gender == "Male" else 0
    smoking = {"Never": 0, "Former": 1, "Current": 2}[smoking]
    fatigue = 1 if fatigue == "Yes" else 0
    weight_loss = 1 if weight_loss == "Yes" else 0

    # Create input data
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

    # Centered button
    st.markdown("###")
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
