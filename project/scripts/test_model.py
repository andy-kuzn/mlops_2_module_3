from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
import pandas as pd

X_test = pd.read_csv('/home/andrey/project/datasets/X_test.csv')

y_test = pd.read_csv('/home/andrey/project/datasets/y_test.csv', header = None)
y_test = y_test[0]

model = LinearRegression()
with open('/home/andrey/project/models/model.pickle', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)
print(score)
