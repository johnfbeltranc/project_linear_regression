# Author: John Fredy Beltran Cuellar
# Date: 10/04/2025
# Goal: realizar el projecto de linear regression

from utils import db_connect
engine = db_connect()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ================================
# Step 1. Carga del dataset
# ================================
df_raw = pd.read_csv("https://breathecode.herokuapp.com/asset/internal-link?id=929&path=medical_insurance_cost.csv")

print("Info dataset original:")
df_raw.info()
display(df_raw.head())

# ================================
# Step 2. Preprocessing
# ================================
df_baking = df_raw.copy()

# Convertir columnas categóricas al tipo correcto
df_baking['sex'] = df_baking['sex'].astype('category')
df_baking['smoker'] = df_baking['smoker'].astype('category')
df_baking['region'] = df_baking['region'].astype('category')

# Crear datasets train/test DESPUÉS de baking
df_train, df_test = train_test_split(df_baking, test_size=0.1, random_state=2025)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print("Tamaños train/test:", df_train.shape, df_test.shape)

# ================================
# Step 3. EDA
# ================================

# Estadísticas descriptivas
display(df_train.describe(include='number').T)
display(df_train.describe(include='category').T)

# Histogramas numéricos
df_train.hist(figsize=(10, 8))
plt.suptitle("Histogramas de variables numéricas", y=1.02)
plt.tight_layout()
plt.show()

# Countplots categóricos
for col in ['sex', 'smoker', 'region']:
    sns.countplot(data=df_train, x=col)
    plt.title(f"Distribución de {col}")
    plt.show()

# Relación entre variables
sns.pairplot(df_train, corner=True)
plt.show()

sns.pairplot(df_train, hue='smoker', corner=True)
plt.show()

# ================================
# Step 4. Machine Learning
# ================================

# Separar features y target
X_train = df_train.drop(columns='charges')
y_train = df_train['charges']

X_test = df_test.drop(columns='charges')
y_test = df_test['charges']

print("Shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Columnas numéricas y categóricas
num_cols = X_train.select_dtypes('number').columns
cat_cols = X_train.select_dtypes('category').columns

# Pipelines de transformación
num_proc = Pipeline(steps=[
    ('scaler', MinMaxScaler())   # también se puede probar StandardScaler
])
cat_proc = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))
])

processor = ColumnTransformer(transformers=[
    ('num', num_proc, num_cols),
    ('cat', cat_proc, cat_cols)
])

# ---- Linear Regression ----
reg_lr = Pipeline(steps=[
    ('preprocess', processor),
    ('model', LinearRegression())
])
reg_lr.fit(X_train, y_train)
y_hat_lr = reg_lr.predict(X_test)

print("\n--- Linear Regression ---")
print(f"MSE : {mean_squared_error(y_test, y_hat_lr):.2f}")
print(f"MAE : {mean_absolute_error(y_test, y_hat_lr):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_hat_lr)):.2f}")

# Visualización predicciones vs reales
plt.scatter(x=y_test, y=y_hat_lr, alpha=0.6, color='b')
plt.xlabel("Real charges")
plt.ylabel("Predicted charges")
plt.title("Linear Regression: Predicción vs Realidad")
plt.grid(True)
plt.show()

# ---- Lasso Regression ----
reg_lasso = Pipeline(steps=[
    ('preprocess', processor),
    ('model', Lasso(alpha=100))
])
reg_lasso.fit(X_train, y_train)
y_hat_lasso = reg_lasso.predict(X_test)

print("\n--- Lasso Regression ---")
print(f"MSE : {mean_squared_error(y_test, y_hat_lasso):.2f}")
print(f"MAE : {mean_absolute_error(y_test, y_hat_lasso):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_hat_lasso)):.2f}")

# ---- Ridge Regression ----
reg_ridge = Pipeline(steps=[
    ('preprocess', processor),
    ('model', Ridge(alpha=100))
])
reg_ridge.fit(X_train, y_train)
y_hat_ridge = reg_ridge.predict(X_test)

print("\n--- Ridge Regression ---")
print(f"MSE : {mean_squared_error(y_test, y_hat_ridge):.2f}")
print(f"MAE : {mean_absolute_error(y_test, y_hat_ridge):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_hat_ridge)):.2f}")

# ================================
# Step 5. Predicción de nuevo cliente
# ================================
new_client = pd.DataFrame({
    'age':[19],
    'sex':['male'],
    'bmi':[35.53],
    'children':[0],
    'smoker':['no'],
    'region':['northwest']
})
pred_cost = reg_lr.predict(new_client)
print("\nCosto estimado para nuevo cliente:", round(pred_cost[0], 2))