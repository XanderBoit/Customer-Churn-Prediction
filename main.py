import sys
import os
sys.path.append(os.path.abspath("."))
import pandas as pd
from src.preprocessing import load_data, preprocess_data
from src.split_utils import split_data
from src.model_utils import train_model, predict_with_threshold, evaluate_model, shap_summary_plot
from catboost import CatBoostClassifier

def main():
    # 1. Load data
    data = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.xls")

    # 2. Preprocess data
    data = preprocess_data(data)

    # 3. Seperate features and target. Split data into train, valid, test
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(data)

    # 4. Initialize model (adjust params as needed)
    model = CatBoostClassifier(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=100
    )

    # 6. Train model
    model = train_model(model, X_train, y_train, X_valid, y_valid)

    # 7. Predict with threshold (e.g., 0.45)
    prediction, y_proba = predict_with_threshold(model, X_test, threshold=0.45)

    # 8. Evaluate results
    evaluate_model(model, X_train, y_train, X_test, y_test, prediction, y_proba)

    # 9. SHAP plot for feature importance
    shap_summary_plot(model, X_test)

if __name__ == "__main__":
    main()