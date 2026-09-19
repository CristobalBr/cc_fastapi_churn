# Telco Customer Churn - FastAPI

Proyecto correspondiente a la asignatura Cloud Computing del Diploma en Data Science de la Universidad Adolfo Ibáñez.

## Objetivo

Desarrollar un modelo de Machine Learning de clasificación binaria para predecir el abandono de clientes (`Churn`) y publicarlo como un servicio de inferencia mediante FastAPI.

## Dataset

Se utiliza el dataset público IBM Telco Customer Churn.

- Observaciones: 7.043 clientes
- Variable objetivo: `Churn`
- Tipo de problema: clasificación binaria
- Clases: `Yes` / `No`

Fuente:

https://github.com/IBM/telco-customer-churn-on-icp4d

## Estructura del proyecto

- `app/`: aplicación FastAPI y esquemas Pydantic.
- `model/`: pipeline entrenado y metadatos.
- `notebooks/`: análisis exploratorio y experimentación.
- `tests/`: pruebas automatizadas de la API.
- `docs/`: evidencias de funcionamiento.
- `data/`: dataset y documentación de origen.
- `train.py`: entrenamiento y serialización del modelo.

## Estado

Proyecto en desarrollo.