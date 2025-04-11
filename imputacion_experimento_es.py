import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Paso 1: Crear un dataset aleatorio
np.random.seed(1977)
n_muestras = 1000
v1 = np.random.normal(loc=50, scale=10, size=n_muestras)
v2 = 0.5 * v1 + np.random.normal(loc=0, scale=5, size=n_muestras)

# Introducir 20% de valores nulos en v1 y 10% en v2
v1_nulos = v1.copy()
v2_nulos = v2.copy()
v1_nulos[np.random.choice(n_muestras, size=int(0.2 * n_muestras), replace=False)] = np.nan
v2_nulos[np.random.choice(n_muestras, size=int(0.1 * n_muestras), replace=False)] = np.nan

df = pd.DataFrame({'v1': v1_nulos, 'v2': v2_nulos})
df_original = pd.DataFrame({'v1': v1, 'v2': v2})

# Paso 2: Imputación por media
imputador_media = SimpleImputer(strategy='mean')
df_media_imputado = pd.DataFrame(imputador_media.fit_transform(df), columns=['v1', 'v2'])

# Paso 3: Imputación por emparejamiento de medias predictivas (PMM aproximado)
imputador_pmm = IterativeImputer(random_state=0, sample_posterior=True)
df_pmm_imputado = pd.DataFrame(imputador_pmm.fit_transform(df), columns=['v1', 'v2'])

# Paso 4: Visualizar histogramas
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle("Histogramas de v1 y v2 (Original, Imputación Media, PMM)", fontsize=16)

sns.histplot(df_original['v1'], bins=30, ax=axes[0, 0], kde=True, color='skyblue')
axes[0, 0].set_title("v1 Original")
sns.histplot(df_original['v2'], bins=30, ax=axes[0, 1], kde=True, color='skyblue')
axes[0, 1].set_title("v2 Original")
sns.histplot(df_media_imputado['v1'], bins=30, ax=axes[1, 0], kde=True, color='orange')
axes[1, 0].set_title("v1 Imputado (Media)")
sns.histplot(df_media_imputado['v2'], bins=30, ax=axes[1, 1], kde=True, color='orange')
axes[1, 1].set_title("v2 Imputado (Media)")
sns.histplot(df_pmm_imputado['v1'], bins=30, ax=axes[2, 0], kde=True, color='green')
axes[2, 0].set_title("v1 Imputado (PMM)")
sns.histplot(df_pmm_imputado['v2'], bins=30, ax=axes[2, 1], kde=True, color='green')
axes[2, 1].set_title("v2 Imputado (PMM)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Paso 5: Comparación de correlaciones
correlaciones = {
    "Original": df_original.corr().iloc[0, 1],
    "Imputación Media": df_media_imputado.corr().iloc[0, 1],
    "Imputación PMM": df_pmm_imputado.corr().iloc[0, 1],
}

print("Correlaciones:")
for metodo, valor in correlaciones.items():
    print(f"{metodo}: {valor:.4f}")
