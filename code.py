# customer_order_prediction.py

import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

# Load data
def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a CSV file and return a cleaned DataFrame."""
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

# Feature Engineering
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features such as AveragePrice and TotalQuantity."""
    df['TotalQuantity'] = df.groupby('BillNo')['Quantity'].transform('sum')
    df['AveragePrice'] = df.groupby('BillNo')['Price'].transform('mean')
    return df

# Train the model
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Train the Random Forest model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Evaluate the model using Mean Squared Error."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Flask API
app = Flask(__name__)

@app.route('/predict_order', methods=['POST'])
def predict_order():
    """API endpoint to predict customer order amount."""
    data = request.get_json()
    features = [[data['TotalQuantity'], data['AveragePrice']]]
    prediction = model.predict(features)
    return jsonify({'predicted_order': prediction[0]})

if __name__ == '__main__':
    # Load and prepare the dataset
    df = load_data('data/sample_data.csv')
    df = feature_engineering(df)

    # Define features and target variable
    y = df.groupby('BillNo')['Price'].sum()
    X = df[['TotalQuantity', 'AveragePrice']]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')

    # Save the trained model
    joblib.dump(model, 'model.joblib')

    # Start the API
    app.run(debug=True)
