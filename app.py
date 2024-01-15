# app.py
import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf

def predict_churn(sample_record):
    loaded_model = tf.keras.models.load_model("churn_predictor")
    with open('min_max_scaler.pkl', 'rb') as file:
        loaded_min_max_scaler = pickle.load(file)
    with open('one_hot_encoder.pkl', 'rb') as file:
        loaded_one_hot_encoder = pickle.load(file)
    sample_record.reset_index(drop=True,inplace=True)

    # preprocessing
    sample_record['TotalCharges'] = pd.to_numeric(sample_record['TotalCharges'])
    sample_record['MonthlyCharges'] = pd.to_numeric(sample_record['MonthlyCharges'])
    sample_record['tenure'] = pd.to_numeric(sample_record['tenure'])

    # Encoding
    cols_to_encode_ = ['Partner',
    'SeniorCitizen',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'PaperlessBilling']
    for col in cols_to_encode_:
        sample_record[col] = sample_record[col].replace({'Yes':1,'No':0})

    sample_record['gender'] = sample_record['gender'].replace({'Male':1,"Female":0})
    one_hot_cols_to_encode = ['InternetService','Contract','PaymentMethod']
    one_hot_arr = loaded_one_hot_encoder.transform(sample_record[one_hot_cols_to_encode])
    one_hot_cols_df = pd.DataFrame(one_hot_arr.toarray(),columns = loaded_one_hot_encoder.get_feature_names_out(one_hot_cols_to_encode))
    sample_record = pd.concat([sample_record,one_hot_cols_df],axis=1)
    sample_record.drop(columns=one_hot_cols_to_encode,inplace=True)

    #Scaling
    cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']
    sample_record[cols_to_scale] = loaded_min_max_scaler.transform(sample_record[cols_to_scale])

    #prediction
    prediction = loaded_model.predict(sample_record)
    prediction = prediction>0.5
    return prediction[0][0]

st.title(" Telecom Customer Churn Prediction!")

st.write("Please fill the necessary fields for churn prediction!")

col1,col2 = st.columns(2)

with col1:
# Create a dropdown with a list of values
    gender = st.selectbox("Gender:", ["Female", "Male"], index=None)

    # Create a dropdown with a list of values
    senior_citizen = st.selectbox("SeniorCitizen:", ["Yes","No"], index=None)

    # Create a dropdown with a list of values
    partner = st.selectbox("Partner:", ["Yes","No"], index=None)

    # Create a dropdown with a list of values
    dependents = st.selectbox("Dependents:", ["Yes","No"], index=None)

    # Create a dropdown with a list of values
    tenure = st.number_input("tenure:", min_value=0)

    # Create a dropdown with a list of values
    phoneservice = st.selectbox("PhoneService:", ["Yes","No"], index=None)
    # Create a dropdown with a list of values
    multiplelines = st.selectbox("MultipleLines:", ["Yes","No"], index=None)

    # Create a dropdown with a list of values
    internetservice = st.selectbox("InternetService:", ["DSL","Fiber optic","No"], index=None)

    # Create a dropdown with a list of values
    onlinesecurity = st.selectbox("OnlineSecurity:", ["Yes","No"], index=None)

    # Create a dropdown with a list of values
    onlinebackup = st.selectbox("OnlineBackup:", ["Yes","No"], index=None)

with col2:
    # Create a dropdown with a list of values
    deviceprotecttion = st.selectbox("DeviceProtection:", ["Yes","No"], index=None)

    # Create a dropdown with a list of values
    techsupport = st.selectbox("TechSupport:", ["Yes","No"], index=None)

    # Create a dropdown with a list of values
    streamingtv = st.selectbox("StreamingTV:", ["Yes","No"], index=None)

    # Create a dropdown with a list of values
    streamingmovies = st.selectbox("StreamingMovies:", ["Yes","No"], index=None)

    # Create a dropdown with a list of values
    contract = st.selectbox("Contract:", ["Month-to-month","One year",'Two year'], index=None)

    # Create a dropdown with a list of values
    paperlessbilling = st.selectbox("PaperlessBilling:", ["Yes","No"], index=None)

    # Create a dropdown with a list of values
    paymentmethod = st.selectbox("PaymentMethod:", ['Electronic check','Mailed check','Bank transfer (automatic)'
                                                    ,'Credit card (automatic)'], index=None)

    # Create a dropdown with a list of values
    monthlycharges = st.number_input("MonthlyCharges:", min_value=0.00)

    # Create a dropdown with a list of values
    totalcharges = st.number_input("TotalCharges:", min_value=0.00)

# Perform different actions based on the selected value

if st.button("Predict"):
    columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
    df = pd.DataFrame([[gender,senior_citizen,partner,dependents,tenure,phoneservice
                       ,multiplelines,internetservice,onlinesecurity,onlinebackup,
                       deviceprotecttion,techsupport,streamingtv,streamingmovies,
                       contract,paperlessbilling,paymentmethod,monthlycharges,totalcharges]]
                      ,columns=columns)
    st.write("Recieved Input:")
    st.dataframe(df,hide_index=True)
    result = predict_churn(df)
    if result:
        st.write("The Customer will Churn")
    else:
        st.write("The Customer will not Churn")