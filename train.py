import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow, pickle

df = pd.read_csv('data/iris.csv')
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    mlflow.log_metric('accuracy', acc)
    pickle.dump(model, open('model/model.pkl', 'wb'))
    print(f"✅ Model trained with accuracy: {acc:.2f}")
