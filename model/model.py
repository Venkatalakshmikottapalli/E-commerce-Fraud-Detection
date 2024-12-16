import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data function
def load_data(filepath):
    return pd.read_csv(filepath)

# Save model function
def save_model(model, filepath, compress_level=3):
    """
    Save the model to a specified filepath with compression.

    Parameters:
    - model: The trained model to save.
    - filepath: Full path to save the model (including filename).
    - compress_level: Compression level (1-9). Higher values increase compression but take longer.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure the directory exists
    joblib.dump(model, filepath, compress=compress_level)  # Save with compression

# Test model function
def test_model(X, y, model):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, model.predict_proba(X), multi_class='ovr', average='macro')
    conf_matrix = confusion_matrix(y, y_pred)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    print(f"Accuracy of Random Forest Model: {accuracy:.4f}")
    print(f"ROC-AUC of the Random Forest Model: {roc_auc:.4f}")

    print("Classification Report:")
    print(classification_report(y, y_pred))

    # Plot the normalized confusion matrix
    plt.figure(figsize=(5, 3))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap='Blues', 
                xticklabels=['Normal', 'Suspicious', 'Fraud'], 
                yticklabels=['Normal', 'Suspicious', 'Fraud'])
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return accuracy, roc_auc

if __name__ == "__main__":
    # Load data
    data = load_data("fraud_data.csv")

    # Prepare train and test data
    X = data.drop('FraudLabel', axis=1)  # Replace 'FraudLabel' with the actual target column name
    y = data['FraudLabel']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate and train the Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Test the model
    accuracy, roc_auc = test_model(X_test, y_test, rf_model)

    # Save the trained model
    model_path = os.path.join(os.getcwd(), 'fraud_rf_model.pkl')
    save_model(rf_model, model_path, compress_level=3)
    print(f"Model saved to {model_path}")
