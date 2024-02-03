from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient
 
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model")

X_train = pd.read_csv('/home/andrey/project/datasets/X_train.csv')

y_train = pd.read_csv('/home/andrey/project/datasets/y_train.csv', header = None)
y_train = y_train[0]

model = LinearRegression()

with mlflow.start_run():
    mlflow.sklearn.log_model(model,
                             artifact_path="lr",
                             registered_model_name="lr")
    mlflow.log_artifact(local_path="/home/andrey/project/scripts/train_model.py",
                        artifact_path="train_model code")
    mlflow.end_run()

model.fit(X_train, y_train)

with open('/home/andrey/project/models/model.pickle', 'wb') as f:
    pickle.dump(model, f)
