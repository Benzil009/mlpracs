import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create mini dataset
data = {
    'Age': [22, 25, 47, 52, 46, 56, 21, 30, 42, 35],
    'Salary': [25000, 32000, 50000, 52000, 48000, 61000, 23000, 40000, 45000, 38000],
    'Experience': [1, 2, 5, 7, 5, 9, 1, 3, 6, 4]
}
df = pd.DataFrame(data)

print(df.describe())
sns.pairplot(df)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
