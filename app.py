import streamlit as st
import pandas as pd
import joblib
import json

# Made by Muhammad Dava Pasha (mdavap) @ Dicoding

data = pd.read_csv('./data.csv', sep=';').drop('Status', axis=1)
columns = data.columns.values

with open('column_definitions.json', 'r') as file:
    column_definitions = json.load(file)

column_definitions = column_definitions['columns']
model = joblib.load('./model/model.joblib')
scaler = joblib.load('./model/scaler.joblib')

st.title("Dropout Prediction by Muhammad Dava Pasha (mdavap) @ Dicoding")

input_prediction = {}

for col in columns:
    definitions = column_definitions[col]
    data_type = definitions['type']
    description = definitions['description']
    if data_type == 'categorical':
        categories = definitions['categories']
        target_category = {category: value for value, category in categories.items()}
        input_prediction[col] = int(target_category[st.selectbox(f'{col} ({description})', target_category)])
    elif data_type == 'numerical':
        if 'range' in definitions:
            input_prediction[col] = st.slider(f'{col} ({description})', 0, int(definitions['range']))
        else:
            min_value = data[col].min()
            max_value = data[col].max()
            input_prediction[col] = st.slider(f'{col} ({description})', min_value, max_value)

if st.button('Predict'):
    input = pd.DataFrame()

    # Fix the order
    for order in columns:
        input[order] = [input_prediction[order]]

    # Scaling
    scaled = scaler.transform(input)

    result = model.predict(scaled)[0]
   
    st.write(f'Will he drop out? {"Yes" if result == 0 else "No"}')