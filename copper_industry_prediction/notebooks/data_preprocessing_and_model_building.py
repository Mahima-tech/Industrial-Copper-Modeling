import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest, RandomForestRegressor, ExtraTreesClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data
data_path = 'C:/Users/Lenovo/Desktop/copper_industry_prediction/data/copper_data.csv'

# Check if the file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"The file {data_path} does not exist.")

# Print the first few lines of the file for debugging
with open(data_path, 'r') as file:
    content = file.readlines()
    if not content:
        raise ValueError("The file is empty")
    for line in content[:5]:  # Print the first 5 lines for inspection
        print(line)

# Read the data with error handling
try:
    data = pd.read_csv(data_path, encoding='utf-8')
    if data.empty:
        raise ValueError("The file is empty")
except pd.errors.EmptyDataError as e:
    print(f"EmptyDataError: {e}")
    data = pd.DataFrame()  # Initialize an empty DataFrame
except pd.errors.ParserError as e:
    print(f"ParserError: {e}")
except Exception as e:
    print(f"Error: {e}")

# Check if data is loaded successfully
if not data.empty:
    # Convert rubbish values to null
    data['material_ref'] = data['material_ref'].apply(lambda x: np.nan if str(x).startswith('00000') else x)

    # Handle missing values
    data.fillna(data.median(), inplace=True)

    # Outlier detection and removal
    iso = IsolationForest(contamination=0.1)
    outliers = iso.fit_predict(data.select_dtypes(include=[np.number]))
    data = data[outliers != -1]

    # Handle skewness
    data['selling_price'] = np.log1p(data['selling_price'])

    # Encode categorical variables
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(data.select_dtypes(include=[object])).toarray()
    data = pd.concat([data.select_dtypes(exclude=[object]), pd.DataFrame(encoded_features)], axis=1)

    # Visualize outliers and skewness
    sns.boxplot(data['selling_price'])
    plt.show()

    # Split data for regression
    X_reg = data.drop(['selling_price'], axis=1)
    y_reg = data['selling_price']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # Train regression model
    reg_model = RandomForestRegressor()
    reg_model.fit(X_train_reg, y_train_reg)
    y_pred_reg = reg_model.predict(X_test_reg)

    # Evaluate regression model
    print('RMSE:', np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))
    print('R² Score:', r2_score(y_test_reg, y_pred_reg))

    # Save the regression model
    with open('C:/Users/Lenovo/Desktop/copper_industry_prediction/models/reg_model.pkl', 'wb') as f:
        pickle.dump(reg_model, f)

    # Split data for classification
    data = data[data['status'].isin(['WON', 'LOST'])]
    X_clf = data.drop(['status'], axis=1)
    y_clf = data['status'].map({'WON': 1, 'LOST': 0})
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    # Train classification model
    clf_model = ExtraTreesClassifier()
    clf_model.fit(X_train_clf, y_train_clf)
    y_pred_clf = clf_model.predict(X_test_clf)

    # Evaluate classification model
    print('Accuracy:', accuracy_score(y_test_clf, y_pred_clf))
    print('Precision:', precision_score(y_test_clf, y_pred_clf))
    print('Recall:', recall_score(y_test_clf, y_pred_clf))
    print('F1 Score:', f1_score(y_test_clf, y_pred_clf))
    print('AUC:', roc_auc_score(y_test_clf, y_pred_clf))

    # Save the classification model
    with open('C:/Users/Lenovo/Desktop/copper_industry_prediction/models/clf_model.pkl', 'wb') as f:
        pickle.dump(clf_model, f)

    # Save the encoder
    with open('C:/Users/Lenovo/Desktop/copper_industry_prediction/models/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    # Save the scaler
    scaler = StandardScaler()
    scaler.fit(X_train_reg)
    with open('C:/Users/Lenovo/Desktop/copper_industry_prediction/models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
else:
    print("Data not loaded. Please check the file and try again.")
