import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/andrey/project/datasets/data.csv')

X = df[['text_length', 'reply_count']]
y = df['like_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train.to_csv('/home/andrey/project/datasets/X_train.csv', index=None)
y_train.to_csv('/home/andrey/project/datasets/y_train.csv', index=None, header=None)
X_test.to_csv('/home/andrey/project/datasets/X_test.csv', index=None)
y_test.to_csv('/home/andrey/project/datasets/y_test.csv', index=None, header=None)
