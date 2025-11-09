import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [25000, 28000, 32000, 36000, 40000, 44000, 47000, 50000, 54000, 60000]
}
df = pd.DataFrame(data)

X = df[['Experience']]
y = df['Salary']

for test_size in [0.2, 0.3]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\nSplit {int((1-test_size)*100)}-{int(test_size*100)}")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R²:", r2_score(y_test, y_pred))

    plt.scatter(X, y, color='blue')
    plt.plot(X, model.predict(X), color='red')
    plt.title("Experience vs Salary (Linear Regression)")
    plt.show()
