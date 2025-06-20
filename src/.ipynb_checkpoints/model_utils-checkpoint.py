import matplotlib.pyplot as plt
import shap
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)


def train_model(model, X_train, Y_train, X_valid=None, Y_valid=None):
    """
    Train the model. If validation data is provided, use it for eval set.
    """
    if X_valid is not None and Y_valid is not None:
        model.fit(X_train, Y_train, eval_set=[(X_valid, Y_valid)], verbose=False)
    else:
        model.fit(X_train, Y_train)
    return model


def predict_with_threshold(model, X_test, threshold=0.5):
    """
    Predict class labels based on probability threshold.
    Returns predicted labels and predicted probabilities for positive class.
    """
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    return preds, proba


def evaluate_model(model, X_train, Y_train, X_test, Y_test, prediction, y_proba):
    """
    Evaluate the model with accuracy, classification report, confusion matrix, ROC curve,
    and compare with baseline dummy classifier and logistic regression.
    """
    print("Accuracy: ", accuracy_score(Y_test, prediction))

    # Baseline dummy classifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, Y_train)
    print("Dummy Classifier Accuracy:", dummy.score(X_test, Y_test))

    # Logistic regression baseline
    logreg = LogisticRegression(max_iter=3000,solver='saga')
    logreg.fit(X_train, Y_train)
    print("Logistic Regression Accuracy:", logreg.score(X_test, Y_test))

    # Classification report
    print("\nClassification Report:\n", classification_report(Y_test, prediction))

    # Confusion matrix plot
    cm = confusion_matrix(Y_test, prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # ROC-AUC and curve plot
    roc_auc = roc_auc_score(Y_test, y_proba)
    print("ROC-AUC:", roc_auc)
    RocCurveDisplay.from_predictions(Y_test, y_proba)
    plt.title("ROC Curve")
    plt.show()


def shap_summary_plot(model, X_test, max_display=5):
    """
    Create a SHAP summary plot for feature importance.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, max_display=max_display)
    plt.tight_layout()
