# iNeuron-Internship-Customer-Order-Predction.

# **Customer Order Prediction**

### **Project Overview**
The **Customer Order Prediction** project uses historical transaction data to predict future customer purchases. By analyzing past buying patterns, the system helps businesses optimize their inventory management and improve customer satisfaction.

### **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Evaluation](#model-evaluation)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## **Dataset**
The dataset used for this project includes transaction records with the following attributes:
- **BillNo**: Unique identifier for the transaction.
- **Itemname**: Name of the item purchased.
- **Quantity**: Number of items purchased.
- **Price**: Price of the item.
- **CustomerID**: Unique identifier for the customer.

A sample dataset is provided in `data/sample_data.csv`.

---

## **Features**
- **Data Cleaning**: Handling missing values and removing duplicates.
- **Feature Engineering**: Creating new features like `TotalQuantity` and `AveragePrice`.
- **Model Training**: A Random Forest Regressor is used to predict customer order amounts.
- **API Development**: A Flask API allows users to make predictions based on input features.

---

## **Technologies Used**
- **Languages**: Python
- **Libraries**: Pandas, Scikit-learn, Flask, NumPy, Matplotlib, Seaborn
- **Deployment**: Flask, Docker, AWS/GCP
- **Version Control**: Git, GitHub

---

## **Project Architecture**
```plaintext
+---------------------+        +---------------------+
|     Data Sources     |        |  External Data/API  |
|   (CSV, Database)    |        |                     |
+----------+----------+        +----------+----------+
           |                              |
           v                              v
+----------+------------------------------+-----------+
|                Data Ingestion                      |
+-----------------+-------------------+------------------+
                  |                   |
                  v                   v
     +---------------------+         +-----------------------+
     |     Data Cleaning    |         |  Feature Engineering  |
     +---------------------+         +-----------------------+
                  |                   |
                  v                   v
       +-----------------------------------+
       |    Machine Learning Model (ML)    |
       +-----------------------------------+
                  |
                  v
     +-------------------------------+
     |        Model Storage           |
     +-------------------------------+
                  |
                  v
+--------------------------------------------+
|        API Layer (Flask / FastAPI)         |
+--------------------------------------------+
                  |
                  v
+-----------------------------------------------+
|               Client Applications             |
+-----------------------------------------------+
```

---

## **Installation**

### **1. Clone the repository**
```bash
git clone https://github.com/yourusername/customer-order-prediction.git
cd customer-order-prediction
```

### **2. Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```

### **3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## **Usage**

### **1. Train the model**
You can run the `train.py` script to train the model:
```bash
python train.py
```

### **2. Start the Flask API**
To serve the model predictions via the Flask API, run:
```bash
python app.py
```

The API will be available at `http://localhost:5000`.

---

## **API Documentation**

### **Endpoint**: `/predict_order`
- **Method**: `POST`
- **Description**: Predicts the total order amount based on input features.
- **Input**: A JSON object with the following structure:
    ```json
    {
      "TotalQuantity": 10,
      "AveragePrice": 5.25
    }
    ```
- **Output**: A JSON object with the predicted order amount:
    ```json
    {
      "predicted_order": 52.4
    }
    ```

---

## **Model Evaluation**

### **Metrics**
- **Mean Squared Error (MSE)** is used to evaluate the model's performance on test data.

```python
from sklearn.metrics import mean_squared_error

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse
```

### **Results**
- **Training MSE**: 0.0234
- **Testing MSE**: 0.0317

---

## **Future Work**
- **Hyperparameter Optimization**: Implementing grid search for improved model accuracy.
- **Real-Time Predictions**: Enable the system to process real-time data for dynamic predictions.
- **Additional Features**: Incorporate customer demographics or time-based features.

---

