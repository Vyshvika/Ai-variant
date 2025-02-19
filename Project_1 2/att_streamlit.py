import pickle
import streamlit as st
import numpy as np
import pandas as pd

def load_model():
    model = pickle.load(open('Attorney_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

def preprocess_input(data, scaler):
    # Convert categorical features
    data['Policy_Type_Third-Party'] = 1 if data['Policy_Type'] == 'Third-Party' else 0
    data['Accident_Severity'] = {'Minor': 0, 'Moderate': 1, 'Severe': 2}[data['Accident_Severity']]
    data['Driving_Record'] = {'Clean': 0, 'Minor Offenses': 1, 'Major Offenses': 2}[data['Driving_Record']]
    
    # Standardizing numerical columns
    numerical_cols = ['CLMAGE', 'LOSS', 'Claim_Amount_Requested', 'Settlement_Amount']
    standardized_values = scaler.transform(np.array([[data[col] for col in numerical_cols]]))
    
    # Apply standardized values
    for i, col in enumerate(numerical_cols):
        data[col] = standardized_values[0][i]
    
    # Create Claim_Settlement_Total after standardization
    if data['Settlement_Amount'] > 0:
        data['Claim_Settlement_Total'] = data['Claim_Amount_Requested'] / data['Settlement_Amount']
    else:
        data['Claim_Settlement_Total'] = 0
    
    # Selecting relevant columns
    input_data = pd.DataFrame([data], columns=['CLMSEX', 'CLMINSUR', 'SEATBELT', 'CLMAGE', 'LOSS',
                                                'Accident_Severity', 'Claim_Approval_Status', 'Driving_Record',
                                                'Policy_Type_Third-Party', 'Claim_Settlement_Total'])
    
    return input_data

def main():
    st.title('Attorney Involvement Prediction')
    model, scaler = load_model()
    
    # Input fields
    CLMSEX = st.selectbox('Claimant Sex (1 = Male, 0 = Female)', [0, 1])
    CLMINSUR = st.selectbox('Claim Insurance (1 = yes, 0 = no)', [0, 1])
    SEATBELT = st.selectbox('Seatbelt Used (1 = yes, 0 = no)', [0, 1])
    CLMAGE = st.number_input('Claimant Age', min_value=0, max_value=100, value=30)
    LOSS = st.number_input('Loss Amount', min_value=0.0, value=1000.0)
    Accident_Severity = st.selectbox('Accident Severity', ['Minor', 'Moderate', 'Severe'])
    Claim_Approval_Status = st.selectbox('Claim Approval Status (1 = yes, 0 = no)', [0, 1])
    Driving_Record = st.selectbox('Driving Record', ['Clean', 'Minor Offenses', 'Major Offenses'])
    Policy_Type = st.selectbox('Policy Type', ['Third-Party', 'Comprehensive'])
    Settlement_Amount = st.number_input('Settlement Amount', min_value=0.0, value=4000.0)
    Claim_Amount_Requested = st.number_input('Claim Amount Requested', min_value=0.0, value=5000.0)
    
    # Prediction button
    if st.button('Predict Attorney Involvement'):
        input_data = preprocess_input({
            'CLMSEX': CLMSEX,
            'CLMINSUR': CLMINSUR,
            'SEATBELT': SEATBELT,
            'CLMAGE': CLMAGE,
            'LOSS': LOSS,
            'Accident_Severity': Accident_Severity,
            'Claim_Amount_Requested': Claim_Amount_Requested,
            'Settlement_Amount': Settlement_Amount,
            'Claim_Approval_Status': Claim_Approval_Status,
            'Driving_Record': Driving_Record,
            'Policy_Type': Policy_Type
        }, scaler)
        
        prediction = model.predict(input_data)[0]
        st.write(f'Predicted Attorney Involvement: {"Yes" if prediction == 1 else "No"}')

if __name__ == '__main__':
    main()
