# Copper Industry Prediction

## Project Description
This project aims to solve the challenges faced by the copper industry in pricing and lead classification using machine learning models. The solution includes a regression model to predict the selling price and a classification model to predict lead status (WON/LOST). An interactive web application is created using Streamlit to allow users to input data and get predictions.

## Project Structure
copper_industry_prediction/
│
├── data/
│ └── copper_data.csv
│
├── models/
│ ├── reg_model.pkl
│ ├── clf_model.pkl
│ ├── encoder.pkl
│ └── scaler.pkl
│
├── notebooks/
│ └── data_preprocessing_and_model_building.ipynb
│
├── app.py
├── requirements.txt
└── README.md

## Instructions
1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the Jupyter notebook `data_preprocessing_and_model_building.ipynb` to preprocess data and build models.
4. Run the Streamlit app using `streamlit run app.py`.
git clone <repository_url>
cd copper_industry_prediction
pip install -r requirements.txt
jupyter notebook notebooks/data_preprocessing_and_model_building.ipynb
streamlit run app.py
