import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("/content/CustomersDataSet")

df[:3]

x = df.drop("Spending Score (1-100)",axis=1)
y = df["Spending Score (1-100)"]

x

X = x.drop("CustomerID",axis=1)
X

X = pd.get_dummies(X, drop_first=True)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

y_train


models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1, max_iter = 1000),
    "Ridge Regression": Ridge(alpha=0.1, max_iter = 1000)
}

from sklearn.metrics import r2_score, mean_squared_error
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "R2 Score": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }

results_df = pd.DataFrame(results).T
print(results_df)