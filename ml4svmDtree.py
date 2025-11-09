import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = {
    'Height': [150, 160, 165, 170, 175, 180, 185, 190, 155, 168],
    'Weight': [50, 55, 60, 65, 70, 75, 80, 90, 52, 62],
    'Gender': [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  # 0=Female, 1=Male
}
df = pd.DataFrame(data)

X = df[['Height', 'Weight']]
y = df['Gender']

for test_size in [0.2, 0.3]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    dt = DecisionTreeClassifier()
    svm = SVC()

    for model, name in [(dt, "Decision Tree"), (svm, "SVM")]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name} ({int((1-test_size)*100)}-{int(test_size*100)}) Accuracy:", accuracy_score(y_test, y_pred))
