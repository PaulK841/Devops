# === Model Training and Experimentation ===

# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

# Load dataset
data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Define a function to train a model and return its accuracy
def train_model(n_estimators, max_depth):
    # Train a RandomForestClassifier
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model with n_estimators={n_estimators}, max_depth={max_depth} achieved accuracy={accuracy:.4f}")
    return model, accuracy

# Train models with different hyperparameter configurations
print("Training Model 1...")
model_1, acc_1 = train_model(n_estimators=20, max_depth=5)

print("\nTraining Model 2...")
model_2, acc_2 = train_model(n_estimators=100, max_depth=10)

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

# get this code from dagshub
import dagshub
dagshub.init(repo_owner='arthur-gtgn', repo_name='mlflow-with-daghub', mlflow=True)

data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Set MLflow experiment name. This will get created if it doesn't exist
experiment_name = "RandomForestExperiment_2"
mlflow.set_experiment(experiment_name)

# Define a function to train a model, log parameters and metrics to MLflow
def train_and_log_model(n_estimators, max_depth):
    with mlflow.start_run(): # <--
        # Train a RandomForestClassifier
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log parameters, metrics, and the model itself
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "random_forest_model_2")

        print(f"Logged RandomForest model with n_estimators={n_estimators}, max_depth={max_depth}, accuracy={accuracy:.4f}")

# Train and log Model 1
print("Training and Logging Model 1...")
train_and_log_model(n_estimators=20, max_depth=5)

# Train and log Model 2
print("\nTraining and Logging Model 2...")
train_and_log_model(n_estimators=100, max_depth=10)

# Instructions to visualize results
print("\nTo view the results, run the following command in your terminal:")
print("mlflow ui")
print("Then navigate to http://127.0.0.1:5000 to explore the experiment results.")

