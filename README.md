# Modelo Computacional de Mantenimiento Predictivo para Ambulancias Tipo II
## Lima Metropolitana — 2024-2025

**Autor:** Osores Aguilar, Diego Jesús  
**Institución:** Universidad Nacional Mayor de San Marcos — FIEE  
**Escuela Profesional:** Ingeniería Biomédica  
**Asesor:** Dr. Cabezas Huerta, Franklin Alfredo  
**Año:** 2026  

---

## Descripción

Repositorio digital de la tesis para optar al Título Profesional de Ingeniero Biomédico. Contiene el código fuente, datasets simulados, resultados, figuras y plantilla semanal asociados al modelo computacional basado en datos para el mantenimiento predictivo con soporte preventivo de ambulancias médicas urbanas Tipo II en Lima Metropolitana.

Los datos incluidos en este repositorio son sintéticos y fueron generados mediante simulación paramétrica-estocástica reproducible. No corresponden a registros institucionales reales del SAMU.

---

## Estructura del repositorio

| Archivo o carpeta | Descripción |
|---|---|
| `generar_dataset_ambulancias.py` | Fases 1-2: generación del dataset simulado y construcción de observaciones por ventanas temporales |
| `modelo_predictivo_ambulancias.py` | Fases 3-4: entrenamiento de modelos, validación retrospectiva y generación de métricas |
| `lineamientos_tecnicos.py` | Fase 5: generación de lineamientos técnicos, criterios de alerta temprana y soporte preventivo |
| `generar_plantilla_semanal.py` | Generación de plantilla semanal editable de programación de mantenimiento |
| `datos_simulados/` | Datasets simulados generados en formato CSV |
| `resultados/` | Resultados del modelo, métricas, salidas de soporte preventivo y plantilla semanal |
| `figuras_tesis/` | Figuras generadas para la tesis |

---

## Requisitos

- Python 3.12.10
- pandas
- numpy
- scikit-learn
- matplotlib
- openpyxl

Instalación de dependencias:
pip install pandas numpy scikit-learn matplotlib openpyxl


---

## Instrucciones de ejecución

Ejecutar en el siguiente orden:

1. Generar el dataset simulado:
python generar_dataset_ambulancias.py

2. Entrenar y validar el modelo:
python modelo_predictivo_ambulancias.py

3. Generar lineamientos técnicos:
python lineamientos_tecnicos.py

4. Generar la plantilla semanal:
python generar_plantilla_semanal.py

---

## Resultados finales reproducibles

La versión final del repositorio corresponde al dataset simulado utilizado en la tesis:

- Flota simulada: 33 ambulancias urbanas Tipo II.
- Dataset final: 3 102 observaciones.
- Entrenamiento 2024: 1 452 observaciones.
- Validación 2025: 1 650 observaciones.
- Variable objetivo positiva total: 18,9%.
- Y = 1 en entrenamiento: 16,7%.
- Y = 1 en validación: 20,8%.
- Ventana histórica: W = 60 días.
- Horizonte de predicción: T = 14 días.
- Paso temporal: semanal.
- Umbral de clasificación: τ = 0,30.
- Modelo principal seleccionado: Random Forest.

Resultados del modelo Random Forest en validación retrospectiva 2025:

| Métrica | Valor |
|---|---:|
| Precisión | 21,18% |
| Sensibilidad | 91,86% |
| Especificidad | 9,95% |
| Exactitud | 27,03% |
| F1-Score | 34,42% |
| AUC-ROC | 56,03% |
| MAE | 45,39% |
| Disponibilidad proyectada | 9,58% |

Estos resultados corresponden a una demostración metodológica con datos simulados y no deben interpretarse como desempeño operativo real del SAMU.

---

## Reproducibilidad

La generación del dataset utiliza una semilla fija:
np.random.seed(42)

---

## Referencia bibliográfica

Osores Aguilar, D. J. (2026). *Modelo computacional basado en datos 
para el mantenimiento predictivo con soporte preventivo de ambulancias 
médicas urbanas Tipo II en Lima Metropolitana* 
[Tesis, Universidad Nacional Mayor de San Marcos]. 
Repositorio GitHub. https://github.com/DJOA5799/tesis-mantenimiento-predictivo-ambulancias
