import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

data = {
    'StudyHours': [2,3,4,5,6,7,8,9,10,11],
    'Attendance': [40,50,60,65,70,80,85,90,95,100],
    'Pass': [0,0,0,1,1,1,1,1,1,1]
}
df = pd.DataFrame(data)

X = df[['StudyHours', 'Attendance']]
y = df['Pass']

for test_size in [0.2, 0.3]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    rf = RandomForestClassifier()
    ada = AdaBoostClassifier()

    for model, name in [(rf, "Random Forest"), (ada, "AdaBoost")]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name} ({int((1-test_size)*100)}-{int(test_size*100)}) Accuracy:", accuracy_score(y_test, y_pred))
