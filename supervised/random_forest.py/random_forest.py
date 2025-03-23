import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_data(csv_file):
    """Load network traffic data from a CSV file."""
    df = pd.read_csv(csv_file)
    # Assume the label column is named 'label'
    X = df.drop(columns=['label'])
    y = df['label']
    return X, y

def main():
    csv_file = 'path_to_your_network_traffic_data.csv'  # Replace with your CSV file path

    # Load data
    X, y = load_data(csv_file)

    # Encode labels if they are not numerical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the classifier
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # Save the model and scaler
    joblib.dump(model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

if __name__ == '__main__':
    main()
