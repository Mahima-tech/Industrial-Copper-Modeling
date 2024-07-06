# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load models and encoders
with open('models/reg_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)
with open('models/clf_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)
with open('models/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title('Copper Industry Prediction')

task = st.selectbox('Select Task', ['Regression', 'Classification'])

if task == 'Regression':
    # Input fields for regression
    input_data = {}
    for col in ['quantity_tons', 'customer', 'country', 'item_type', 'application', 'thickness', 'width', 'material_ref', 'product_ref', 'delivery_date']:
        input_data[col] = st.text_input(f'Enter {col}')
    
    # Preprocess inputs
    input_df = pd.DataFrame([input_data])
    encoded_inputs = encoder.transform(input_df.select_dtypes(include=[object])).toarray()
    input_df = pd.concat([input_df.select_dtypes(exclude=[object]), pd.DataFrame(encoded_inputs)], axis=1)
    input_df = scaler.transform(input_df)
    
    # Predict selling price
    pred_price = reg_model.predict(input_df)
    st.write('Predicted Selling Price:', np.expm1(pred_price[0]))

else:
    # Input fields for classification
    input_data = {}
    for col in ['quantity_tons', 'customer', 'country', 'item_type', 'application', 'thickness', 'width', 'material_ref', 'product_ref', 'delivery_date']:
        input_data[col] = st.text_input(f'Enter {col}')
    
    # Preprocess inputs
    input_df = pd.DataFrame([input_data])
    encoded_inputs = encoder.transform(input_df.select_dtypes(include=[object])).toarray()
    input_df = pd.concat([input_df.select_dtypes(exclude=[object]), pd.DataFrame(encoded_inputs)], axis=1)
    input_df = scaler.transform(input_df)
    
    # Predict status
    pred_status = clf_model.predict(input_df)
    st.write('Predicted Status:', 'WON' if pred_status[0] == 1 else 'LOST')
