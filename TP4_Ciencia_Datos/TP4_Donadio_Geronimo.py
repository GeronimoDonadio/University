# importamos las librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.getcwd() # El path actual, donde hay que ubicar el excel adjuntado

# cargamos el dataset
df = pd.read_excel('1.xlsx')

# verificamos la cantidad de registros y columnas
print(df.info())
print(f"total de filas: {df.shape[0]}")
print(f"total de columnas: {df.shape[1]}")

# resumen estadistico
print(df.describe()) # variables numericas
print(df.describe(include='object')) # variables categoricas

# analisis grafico

# distribucion de las variables numericas
# Configuración del estilo de fondo oscuro
plt.style.use('dark_background')
# histograma para 'points'
plt.figure(figsize=(12, 6))
sns.histplot(df['points'], bins=20, kde=True)
plt.title('distribución de puntos')
plt.xlabel('puntos')
plt.ylabel('frecuencia')
plt.show()

# histograma para 'price'
plt.figure(figsize=(12, 6))
sns.histplot(df['price'], bins=20, kde=True)
plt.title('distribución de precios')
plt.xlabel('precio')
plt.ylabel('frecuencia')
plt.show()

# boxplot para 'points'
plt.figure(figsize=(12, 6))
sns.boxplot(x='points', data=df)
plt.title('boxplot de puntos')
plt.show()

# boxplot para 'price'
plt.figure(figsize=(12, 6))
sns.boxplot(x='price', data=df)
plt.title('boxplot de precios')
plt.show()

# mejoramos la visualizacion de price

# excluir los precios mayores a 1000 para mejorar la visibilidad
df_filtered = df[df['price'] < 1500]
plt.figure(figsize=(12, 6))
sns.histplot(df_filtered['price'], bins=50, kde=True)
plt.title('distribución de precios (sin outliers)')
plt.xlabel('precio')
plt.ylabel('frecuencia')
plt.show()

# histograma de precios con escala logarítmica en el eje x
plt.figure(figsize=(12, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.xscale('log')  # cambiar a escala logarítmica en el eje x
plt.title('distribución de precios (escala logarítmica en eje x)')
plt.xlabel('precio (log)')
plt.ylabel('frecuencia')
plt.show()

# transforcion logaritmica de la variable 'price'

# aplicar la transformación logarítmica a la variable price
df['log_price'] = np.log(df['price'] + 1)  # se añade +1 para evitar log(0)

# ver la distribución de log_price
df['log_price'].hist(bins=50)

# comparar el sesgo (skewness) antes y después de la transformación logarítmica
print('sesgo original:', df['price'].skew())
print('sesgo log-transformada:', df['log_price'].skew())

# variables categoricas
# Gráfico de barras para 'country' (mostrar los 10 países con más reseñas)
plt.figure(figsize=(12, 6))
df['country'].value_counts().head(10).plot(kind='bar')
plt.title('Cantidad de reseñas por país (Top 10)', fontsize=15)
plt.xlabel('País', fontsize=12)
plt.ylabel('Cantidad de reseñas', fontsize=12)
# Ajuste del espaciado y rotación de etiquetas
plt.xticks(rotation=45, fontsize=10, ha='right')  # Rotar etiquetas 45 grados y alinear a la derecha
plt.tight_layout()  # Ajustar automáticamente para evitar cortes

plt.show()

# Gráfico de barras para 'variety' (mostrar las 10 variedades más comunes)
plt.figure(figsize=(12, 6))
df['variety'].value_counts().head(10).plot(kind='bar')
plt.title('Variedades de Uva más comunes (Top 10)', fontsize=15)
plt.xlabel('Variedad de Uva', fontsize=12)
plt.ylabel('Cantidad de reseñas', fontsize=12)
# Ajuste del espaciado y rotación de etiquetas
plt.xticks(rotation=45, fontsize=10, ha='right')  # Rotar etiquetas 45 grados y alinear a la derecha
plt.tight_layout()  # Ajustar automáticamente para evitar cortes

plt.show()

# Gráfico de barras para 'winery' (mostrar los 10 productores más comunes)
plt.figure(figsize=(12, 6))
df['winery'].value_counts().head(10).plot(kind='bar')
plt.title('Bodegas más reseñadas (Top 10)', fontsize=15)
plt.xlabel('Bodega', fontsize=12)
plt.ylabel('Cantidad de reseñas', fontsize=12)
# Ajuste del espaciado y rotación de etiquetas
plt.xticks(rotation=45, fontsize=10, ha='right')  # Rotar etiquetas 45 grados y alinear a la derecha
plt.tight_layout()  # Ajustar automáticamente para evitar cortes

plt.show()

# Boxplot de puntos por país (los 10 países con más reseñas)
plt.figure(figsize=(12, 6))
top_countries = df['country'].value_counts().index[:10]
sns.boxplot(x='country', y='points', data=df[df['country'].isin(top_countries)])
plt.title('Distribución de puntos por país (Top 10 países)', fontsize=15)
plt.xlabel('País', fontsize=12)
plt.ylabel('Puntos', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# Boxplot de precios por variedad de uva (las 10 variedades más comunes)
plt.figure(figsize=(12, 6))
top_varieties = df['variety'].value_counts().index[:10]
sns.boxplot(x='variety', y='price', data=df[df['variety'].isin(top_varieties)])
plt.title('Distribución de precios por variedad de uva (Top 10 variedades)', fontsize=15)
plt.xlabel('Variedad de Uva', fontsize=12)
plt.ylabel('Precio', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# Gráfico para las variedades de vino más caras y más baratas
plt.figure(figsize=(12, 6))
avg_price_by_variety = df.groupby('variety')['price'].mean().sort_values(ascending=False).head(10)
avg_price_by_variety.plot(kind='bar', color='skyblue')
plt.title('Variedades de Vino más Caras', fontsize=15)
plt.xlabel('Variedad', fontsize=12)
plt.ylabel('Precio Promedio', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Gráfico de dispersión para analizar relación entre precio y puntos
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='price', y='points', alpha=0.6)
plt.title('Relación entre Precio y Puntos', fontsize=15)
plt.xlabel('Precio', fontsize=12)
plt.ylabel('Puntos', fontsize=12)
plt.xscale('log')  # Escala logarítmica para mejorar la visualización de precios extremos
plt.tight_layout()
plt.show()

# Tratamiento de valores faltantes

# Identficar nulls
print(df.isnull().sum())
missing_percentage = df.isnull().mean() * 100
print(missing_percentage)

# Pocos valores faltantes 1%
df = df.dropna(subset=['country', 'province', 'variety'])

# Valores faltantes significativos 6-7%
# Rellenar los valores faltantes en columnas categóricas con "Desconocido"
df['region_1'] = df['region_1'].fillna('Desconocido')
df['taster_name'] = df['taster_name'].fillna('Desconocido')
df['taster_twitter_handle'] = df['taster_twitter_handle'].fillna('Desconocido')
# Rellenar los valores faltantes en 'price' con la mediana
df['price'] = df['price'].fillna(df['price'].median())
df['log_price'] = np.log(df['price'] + 1)  # Actualizar la variable log_price

# Grandes cantidades de valores faltantes
df.drop(['designation', 'region_2'], axis=1, inplace=True)

# Verificacion de valores faltantes
print(df.isnull().sum())