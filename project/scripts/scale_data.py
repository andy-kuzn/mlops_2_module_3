from sklearn.preprocessing import StandardScaler
import pandas as pd

X_train = pd.read_csv('/home/andrey/project/datasets/X_train.csv')
X_test = pd.read_csv('/home/andrey/project/datasets/X_test.csv')

scaler = StandardScaler()
scaler.fit(X_train)

X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

X_train.to_csv('/home/andrey/project/datasets/X_train.csv', index=None)
X_test.to_csv('/home/andrey/project/datasets/X_test.csv', index=None)
