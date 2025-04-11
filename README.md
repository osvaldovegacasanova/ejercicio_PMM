# Imputación de Valores Faltantes: Media vs Predictive Mean Matching (PMM)

Este repositorio contiene una demostración práctica en Python sobre cómo tratar valores faltantes utilizando dos métodos comunes:

- **Imputación por la media**
- **Emparejamiento de medias predictivo (Predictive Mean Matching - PMM)** (mediante `IterativeImputer` de `scikit-learn`)

---

## 🎯 Objetivo

Explorar cómo cada método afecta:

- La distribución de los datos (vía histogramas)
- La correlación entre las variables originales y las imputadas

---

## 📊 Descripción del experimento

1. Se generan dos variables correlacionadas `v1` y `v2`.
2. Se introduce artificialmente un 20% de valores nulos en `v1` y un 10% en `v2`.
3. Se imputan los valores nulos usando:
   - La media de cada variable
   - Una aproximación a PMM usando `IterativeImputer` con `sample_posterior=True`
4. Se grafican histogramas de las distribuciones antes y después de imputar.
5. Se comparan las correlaciones.

---

## 🧪 Requisitos

```bash
pip install -r requirements.txt
