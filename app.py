import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("best_model_v2.pkl")


st.set_page_config(page_title="💼 Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction App")
st.markdown("Predict whether an employee earns **>50K** or **≤50K** based on their details.")

# -- Input Form
st.header("📥 Single Prediction")
with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 90, 30)
        fnlwgt = st.number_input("Final Weight (fnlwgt)", 10000, 1000000, 250000)
        education_num = st.slider("Education Number", 1, 16, 10)
        capital_gain = st.number_input("Capital Gain", 0, 100000, 0)

    with col2:
        workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc',
                                               'Federal-gov', 'Local-gov', 'State-gov'])
        marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married',
                                                         'Separated', 'Widowed'])
        capital_loss = st.number_input("Capital Loss", 0, 5000, 0)

    with col3:
        occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                                                 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                                                 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                                                 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Others'])
        relationship = st.selectbox("Relationship", ['Husband', 'Wife', 'Own-child', 'Not-in-family', 'Unmarried'])
        race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
        gender = st.selectbox("Gender", ['Male', 'Female'])
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)
        native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'India'])

    submitted = st.form_submit_button("🔮 Predict")

# -- Predict
if submitted:
    input_dict = {
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'education-num': [education_num],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [gender],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    }

    input_df = pd.DataFrame(input_dict)

    try:
        # Use the loaded pipeline to preprocess and predict
        prediction = model.predict(input_df)[0]
        result = ">50K" if prediction == '>50K' else "≤50K" # Assuming your model predicts '>50K' and '<=50K' as strings
        st.success(f"🎯 Predicted Salary Class: **{result}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")