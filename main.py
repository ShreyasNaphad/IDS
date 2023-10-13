import streamlit as st
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from collections import Counter


# Load the trained logistic regression model
log_reg = joblib.load('idm.pkl')
#scaler = joblib.load('scaler.pkl')

st.title("Intrusion Detection System")
st.set_option('deprecation.showPyplotGlobalUse', False)
# File upload
st.write("Upload a CSV file containing network traffic data:")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    st.success("File uploaded successfully ✅")
    user_data = pd.read_csv(uploaded_file)
    user_data = user_data.drop(['outcome', 'level'], axis=1)

    predictions = log_reg.predict(user_data)
    prediction_labels = ["Normal" if pred == 0 else "Intrusions Detected" for pred in predictions]
    prediction_counts = Counter(prediction_labels)

    # Display the counts
    st.write("Prediction Result:")
    for label, count in prediction_counts.items():
        st.write(f"{label}: {count}")

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    ax = sns.barplot(x=list(prediction_counts.keys()), y=list(prediction_counts.values()))
    plt.title("Distribution of Normal and Intrusion")
    plt.xlabel("Result")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot()
