import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    'study_hours': [1, 2, 3, 2, 5, 6, 7, 8, 4, 3, 6, 8, 9, 1, 4],
    'sleep_hours': [4, 5, 5, 7, 6, 7, 8, 8, 5, 8, 5, 7, 8, 8, 7],
    'passed':      [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['study_hours', 'sleep_hours']]
y = df['passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.0f}%")

new_student = pd.DataFrame([[6, 7]], columns=['study_hours', 'sleep_hours'])

result = model.predict(new_student)[0]
prob = model.predict_proba(new_student)[0][1]

status = "Pass 🎉" if result == 1 else "Fail ❌"
print(f"Prediction: {status} (Confidence: {prob * 100:.1f}%)")
