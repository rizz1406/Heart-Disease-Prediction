import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    print("Loading the dataset")
    data = pd.read_csv(file_path)
    X = data.drop(columns=["target"])
    y = data["target"]
    return X, y

# Function to preprocess the data (scaling the features)
def preprocess_data(X):
    """
    Scale the features using StandardScaler.
    Args:
        X (DataFrame): Feature data.
    Returns:
        ndarray: Scaled features.
    """
    print("Preprocessing the data...")
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Function to train the model
def train_model(X_train, y_train):
    """
    Train a Random Forest model.
    Args:
        X_train (ndarray): Training features.
        y_train (ndarray): Training labels.
    Returns:
        RandomForestClassifier: Trained model.
    """
    print("Training the model...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model's performance
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    Args:
        model: Trained model.
        X_test (ndarray): Test features.
        y_test (ndarray): True labels.
    Returns:
        tuple: Accuracy score and classification report.
    """
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def main():
    dataset_file = r"C:\Users\DELL\Desktop\pythonProjects\personalpythonProjects\Heart Disease Prediction\heart.csv"  # Use raw string (r) for Windows paths
    # Check if the file exists
    if not os.path.exists(dataset_file):
        print(f"Dataset file '{dataset_file}' not found!")
        return
    
    print("=====================================")
    print("Heart Disease Prediction Model")
    print("=====================================")

    # Load and preprocess the data
    X, y = load_data(dataset_file)
    X_scaled = preprocess_data(X)
    
    # Split the data into train and test sets (80% train, 20% test)
    print("Splitting the data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)


    #Display the results
    print("\n==================== Results ====================")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)
    print("=====================================")

if __name__ == "__main__":
    main()
    
    


