import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from pygit2 import Repository
import os   
import json 


try:
    branch_name = Repository('.').head.shorthand
except Exception:
    branch_name = "local" 


mlflow.set_tracking_uri('http://35.223.208.118:8100')

def train_model():
    """
    Trains a model on the current version of the data, logs to MLflow,
    and saves a metrics file for CML.
    """
    print("--- Starting training run ---")
 
    print("Loading local data files...")
    try:
        train_path = 'data/train.csv'
        test_path = 'data/test.csv'
    except Exception as e:
        print(f"Error loading data files: {e}")
        return 

    print("Data loaded. Preparing training and test sets...")
    X_train = pd.read_csv(train_path).drop('species', axis=1)
    y_train = pd.read_csv(train_path)['species']
    X_test = pd.read_csv(test_path).drop('species', axis=1)
    y_test = pd.read_csv(test_path)['species']

    with mlflow.start_run(run_name=f"run-{branch_name}"):
        print(f"MLflow run started for branch: {branch_name}")

        mlflow.autolog()
        
        mlflow.log_param("branch_name", branch_name)

        print("Training Logistic Regression model...")
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
        model.fit(X_train, y_train)

        print("Evaluating model...")
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy}")

        mlflow.log_metric("accuracy", accuracy)

        print("Saving metrics for CML report...")
        
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)

        metrics_data = {'accuracy': accuracy}

        with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        print(f"Successfully wrote metrics to {os.path.join(results_dir, 'metrics.json')}")

    print("--- Training run finished ---")

if __name__ == "__main__":
    train_model()