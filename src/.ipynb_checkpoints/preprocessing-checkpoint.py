import pandas as pd

def load_data(path):
    data = pd.read_csv(path)
    data = data.drop(['customerID'], axis=1)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna()
    return data

def preprocess_data(data):
    #Manual assignment to numeric values
    mapping_dicts = {
        'gender': {'Male': 1, 'Female': 2},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'PhoneService': {'Yes': 1, 'No': 0},
        'MultipleLines': {'Yes': 1, 'No': 0, 'No phone service': 0},
        'OnlineSecurity': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'OnlineBackup': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'StreamingTV': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'StreamingMovies': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'Contract': {'Month-to-month': 1, 'One year': 2, 'Two year': 3},
        'PaperlessBilling': {'Yes': 1, 'No': 0},
        'Churn': {'Yes': 1, 'No': 0}
    }
    
    for col, mapping in mapping_dicts.items():
        data[col] = data[col].map(mapping)
    
    # One-hot encode PaymentMethod and InternetService
    data = pd.get_dummies(data, columns=["PaymentMethod", "InternetService"], drop_first=True)
    
    return data