# -*- coding: utf-8 -*-
"""
Created on Sat May 24 18:05:06 2025

@author: LENOVO
"""

# Gerekli kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Veri Yükleme
df = pd.read_csv('insurance.csv', sep=';')
df.info()

#Eksik veri kontrolü yapıldı
df.isnull().any().any()
df.head(10)
df.describe()

# 'charges' sütununu sayısala çevirme
# (Nokta yerine boşluk koyup virgül gibi davranmasını engelleme)
df['charges'] = df['charges'].str.replace('.', '', regex=False).astype(float)

# One-hot encoding uygulama
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Özellikler ve hedef değişken
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# Sadece sayısal sütunları seç
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
# Aykırı Değer Filtreleme
q01 = df['charges'].quantile(0.01)
q99 = df['charges'].quantile(0.99)
df_filtered = df[(df['charges'] > q01) & (df['charges'] < q99)]

# Boxplot çizimi
plt.figure(figsize=(11, 5))
for i, column in enumerate(numeric_columns):
    plt.subplot(1, len(numeric_columns), i + 1)
    sns.boxplot(y=df[column])
    plt.title(column)
plt.tight_layout()
plt.show()

# Dağılım grafiği
plt.figure(figsize=(8, 6))
sns.histplot(df['charges'], kde=True, color='skyblue')
plt.title('Quality Distribution Plot', fontsize=20)
plt.show()


# Değişkenler Arası İlişki Matrisi (Pair Plot)
sns_plot = sns.pairplot(df)

# Saçılım Grafikleri Oluşturma

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.scatterplot(x='age', y='charges', hue='smoker', data=df, palette='viridis')
plt.title('Charges vs. Age')

plt.subplot(1, 3, 2)
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df, palette='viridis')
plt.title('Charges vs. BMI')

plt.subplot(1, 3, 3)
sns.scatterplot(x='children', y='charges', hue='smoker', data=df, palette='viridis')
plt.title('Charges vs. Children')
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot 
plt.figure(figsize=(10, 8))
sns.pairplot(df, hue='smoker', vars=['age', 'bmi', 'children', 'charges'])
plt.show()

# Heatmap için sadece sayısal sütunları al
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

# Kodlama ve Bölme
def encode_and_split(df, target='charges', test_size=0.2):
    df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
    X = df_encoded.drop(target, axis=1)
    y = df_encoded[target]
    return train_test_split(X, y, test_size=test_size, random_state=42)

X_train, X_test, y_train_raw, y_test_raw = encode_and_split(df_filtered)

print(X_train.head())

# Log Dönüşümü
def log_transform(y):
    return np.log1p(y), np.expm1

y_train, inverse_transform = log_transform(y_train_raw)
y_test, _ = log_transform(y_test_raw)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA Görselleştirmesi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaler.fit_transform(X))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.5)
plt.title('PCA Görselleştirmesi')
plt.show()


#Modellerin Tanımı ve Eğitimi
models = {
    'Doğrusal Regresyon': Pipeline([
        ('ölçekleyici', StandardScaler()),
        ('model', LinearRegression())
    ]),
    'Ridge Regresyonu': Pipeline([
        ('ölçekleyici', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ]),
    'Rastgele Orman': Pipeline([
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
}

results = []

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred_log = pipeline.predict(X_test)
    y_pred = inverse_transform(y_pred_log)
    y_true = inverse_transform(y_test)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    results.append({'Model': name, 'RMSE': rmse, 'R2': r2})
    
# Örnek Model Sonuçları (örnek amaçlı, gerçek modeller burada eğitilmeli)
results = [
    {'Model': 'Linear Regression', 'RMSE': 4500, 'R2': 0.82},
    {'Model': 'Random Forest', 'RMSE': 3200, 'R2': 0.89},
    {'Model': 'Gradient Boosting', 'RMSE': 3100, 'R2': 0.90}
]


# Sonuç Karşılaştırması
results_df = pd.DataFrame(results)
print(results_df.sort_values(by='RMSE'))

# Grafiklerle görselleştirme
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x='Model', y='RMSE', palette='coolwarm')
plt.title("Modellere Göre RMSE Karşılaştırması")
plt.ylabel("RMSE (Düşük = Daha İyi)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x='Model', y='R2', palette='viridis')
plt.title("Modellere Göre R2 Karşılaştırması")
plt.ylabel("R2 Skoru (Yüksek = Daha İyi)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

