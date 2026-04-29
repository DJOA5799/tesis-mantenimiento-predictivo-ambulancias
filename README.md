# Modelo Computacional de Mantenimiento Predictivo para Ambulancias Tipo II
## Lima Metropolitana — 2024-2025

**Autor:** Osores Aguilar, Diego Jesús  
**Institución:** Universidad Nacional Mayor de San Marcos — FIEE  
**Escuela Profesional:** Ingeniería Biomédica  
**Asesor:** Mg. Cabezas Huerta, Franklin Alfredo  
**Año:** 2026  

---

## Descripción

Repositorio digital de la tesis para optar al Título Profesional de 
Ingeniero Biomédico. Contiene el código fuente completo del modelo 
computacional predictivo para el mantenimiento de ambulancias médicas 
urbanas Tipo II del SAMU en Lima Metropolitana.

---

## Estructura del repositorio

| Archivo | Descripción |
|---|---|
| `generar_dataset_ambulancias.py` | Fase 1-2: Generación del dataset simulado |
| `modelo_predictivo_ambulancias.py` | Fase 3-4: Modelo predictivo y validación |
| `lineamientos_tecnicos.py` | Fase 5: Lineamientos técnicos y soporte preventivo |
| `datos_simulados/` | Carpeta con los datasets generados (CSV) |
| `resultados/` | Carpeta con los resultados del modelo (CSV) |

---

## Requisitos

- Python 3.12.10
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Instalación de dependencias:
pip install pandas numpy scikit-learn matplotlib seaborn


---

## Instrucciones de ejecución

Ejecutar en el siguiente orden:

1. Generar el dataset simulado:
python generar_dataset_ambulancias.py

2. Entrenar y validar el modelo:
python modelo_predictivo_ambulancias.py

3. Generar lineamientos técnicos:
python lineamientos_tecnicos.py


---

## Referencia bibliográfica

Osores Aguilar, D. J. (2026). *Modelo computacional basado en datos 
para el mantenimiento predictivo con soporte preventivo de ambulancias 
médicas urbanas Tipo II en Lima Metropolitana* 
[Tesis, Universidad Nacional Mayor de San Marcos]. 
Repositorio GitHub. https://github.com/DJOA5799/tesis-mantenimiento-predictivo-ambulancias