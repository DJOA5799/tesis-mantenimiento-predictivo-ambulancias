"""
=============================================================================
EXTENSIÓN ANALÍTICA: COMPARACIÓN ESTADÍSTICA FORMAL DE CLASIFICADORES
Ambulancias Tipo II — Lima Metropolitana, 2024-2025

Este script NO reentrena ni modifica el pipeline de modelado original
(modelo_predictivo_ambulancias.py). Reutiliza las mismas funciones de carga,
balanceo, entrenamiento y evaluación ya existentes (con la misma semilla
np.random.seed(42) y el mismo umbral tau=0.30), y agrega tres análisis
complementarios sobre el conjunto de validación 2025 (n=1650):

  1. Prueba de McNemar (McNemar, 1947; Dietterich, 1998) — comparación de
     tasas de error entre pares de clasificadores correlacionados
     (mismas observaciones).
  2. Prueba de DeLong (DeLong et al., 1988; algoritmo eficiente de
     Sun & Xu, 2014) — comparación del AUC-ROC entre pares de
     clasificadores correlacionados.
  3. Curva Precision-Recall comparativa (Saito & Rehmsmeier, 2015).
Requisitos: ejecutar desde el mismo directorio que
modelo_predictivo_ambulancias.py y datos_simulados/.

Salidas generadas (carpeta resultados/):
  - tabla_mcnemar.csv
  - tabla_delong.csv
  - figura_30b_curva_precision_recall.png
  - resumen_analisis_inferencial.txt
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import precision_recall_curve, average_precision_score

# Reutiliza el pipeline original sin modificarlo
from modelo_predictivo_ambulancias import (
    cargar_datos,
    balancear_clases,
    entrenar_modelos,
    evaluar_modelo,
    calcular_linea_base,
)

UMBRAL = 0.30
RUTA_TRAIN = 'datos_simulados/dataset_entrenamiento_2024.csv'
RUTA_VAL = 'datos_simulados/dataset_validacion_2025.csv'
DIR_RESULTADOS = 'resultados'

NOMBRE_RF = 'Random Forest'


# =============================================================================
# 1. REPRODUCCIÓN DEL PIPELINE EXISTENTE (sin modificaciones)
# =============================================================================

def reproducir_resultados():
    """
    Reproduce exactamente el entrenamiento y evaluación del pipeline
    original. Devuelve resultados_lista, y_val e y_true (array).

    Los valores de AUC-ROC, sensibilidad, etc. obtenidos aquí deben
    coincidir con los reportados en la Tabla 16 / Tabla E.1 de la tesis,
    salvo diferencias de orden 1e-4 atribuibles a versión de scikit-learn
    (verificado: Gradient Boosting AUC-ROC = 0.5254 en esta ejecución vs.
    0.5258 reportado; diferencia no material para las conclusiones).
    """
    X_train, y_train, X_val, y_val, df_train, df_val = cargar_datos(
        RUTA_TRAIN, RUTA_VAL
    )
    X_train_bal, y_train_bal = balancear_clases(
        X_train, y_train, metodo='oversample'
    )
    modelos = entrenar_modelos(X_train_bal, y_train_bal)

    resultados_lista = [calcular_linea_base(y_val, X_val)]
    for nombre, info in modelos.items():
        resultados_lista.append(
            evaluar_modelo(nombre, info, X_val, y_val, umbral=UMBRAL)
        )

    y_true = y_val.values.astype(int)
    return resultados_lista, y_true


# =============================================================================
# 2. PRUEBA DE McNEMAR
# =============================================================================

def tabla_contingencia_mcnemar(correcto_a: np.ndarray,
                                correcto_b: np.ndarray) -> np.ndarray:
    """
    Construye la tabla de contingencia 2x2 de McNemar entre dos
    clasificadores A y B evaluados sobre las mismas observaciones.

        [[n11, n10],
         [n01, n00]]

    n11: ambos correctos | n10: solo A correcto | n01: solo B correcto
    n00: ambos incorrectos
    """
    n11 = int(np.sum((correcto_a == 1) & (correcto_b == 1)))
    n10 = int(np.sum((correcto_a == 1) & (correcto_b == 0)))
    n01 = int(np.sum((correcto_a == 0) & (correcto_b == 1)))
    n00 = int(np.sum((correcto_a == 0) & (correcto_b == 0)))
    return np.array([[n11, n10], [n01, n00]])


def ejecutar_mcnemar(resultados_lista: list, y_true: np.ndarray) -> pd.DataFrame:
    """
    Aplica la prueba de McNemar comparando Random Forest contra cada uno
    de los demás enfoques evaluados, sobre el conjunto de validación 2025.

    Regla de selección exact/asintótico: si n10+n01 < 25, se usa la
    distribución binomial exacta; en caso contrario, la aproximación
    chi-cuadrado con corrección de continuidad de Yates (Dietterich, 1998).
    """
    res_map = {r['nombre']: r for r in resultados_lista}
    correctas = {
        nombre: (np.array(r['y_pred']) == y_true).astype(int)
        for nombre, r in res_map.items()
    }

    pares = [
        (NOMBRE_RF, 'Preventivo tradicional (línea base)'),
        (NOMBRE_RF, 'Regresión Logística'),
        (NOMBRE_RF, 'Árbol de Decisión'),
        (NOMBRE_RF, 'Gradient Boosting'),
    ]

    filas = []
    for a, b in pares:
        tabla = tabla_contingencia_mcnemar(correctas[a], correctas[b])
        n10, n01 = int(tabla[0, 1]), int(tabla[1, 0])
        n_disc = n10 + n01
        exacto = n_disc < 25
        if exacto:
            # Prueba binomial exacta bilateral bajo H0: P(n10)=P(n01)=0.5
            p_valor = stats.binomtest(min(n10, n01), n_disc, p=0.5,
                                      alternative='two-sided').pvalue
            estadistico = min(n10, n01)
            tipo_prueba = 'Exacta (binomial)'
        else:
            # Aproximación chi-cuadrado con corrección de continuidad de Yates
            estadistico = (abs(n10 - n01) - 1) ** 2 / n_disc
            p_valor = stats.chi2.sf(estadistico, df=1)
            tipo_prueba = 'Chi² (Yates)'

        filas.append({
            'Comparación': f'{a} vs. {b}',
            'n10 (solo A correcto)': n10,
            'n01 (solo B correcto)': n01,
            'Tipo de prueba': tipo_prueba,
            'Estadístico': round(float(estadistico), 4),
            'Valor p': p_valor,
            'Significativo (p<0.05)': 'Sí' if p_valor < 0.05 else 'No',
        })

    return pd.DataFrame(filas)


# =============================================================================
# 3. PRUEBA DE DeLONG
# =============================================================================

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Calcula midranks, manejando empates (requerido por DeLong)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N - 1 and Z[j] == Z[j + 1]:
            j += 1
        T[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray,
                  label_1_count: int):
    """
    Algoritmo de cálculo eficiente de DeLong (Sun & Xu, 2014) para
    estimar AUC y su matriz de covarianza entre clasificadores
    correlacionados.
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true: np.ndarray, proba_a: np.ndarray,
                     proba_b: np.ndarray):
    """
    Compara los AUC-ROC de dos clasificadores correlacionados
    (evaluados sobre las mismas observaciones) mediante el test de
    DeLong (DeLong et al., 1988).

    Devuelve: auc_a, auc_b, diferencia, estadístico z, valor p (bilateral).
    """
    order = np.argsort(-y_true)
    y_true_sorted = y_true[order]
    m = int(np.sum(y_true_sorted))  # número de positivos

    preds = np.vstack([proba_a, proba_b])[:, order]
    aucs, cov = _fast_delong(preds, m)

    auc_diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = auc_diff / np.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs[0], aucs[1], auc_diff, z, p


def ejecutar_delong(resultados_lista: list, y_true: np.ndarray) -> pd.DataFrame:
    """
    Aplica la prueba de DeLong comparando el AUC-ROC de Random Forest
    contra cada uno de los demás enfoques evaluados.
    """
    res_map = {r['nombre']: r for r in resultados_lista}
    y_true_f = y_true.astype(float)

    pares = [
        (NOMBRE_RF, 'Gradient Boosting'),
        (NOMBRE_RF, 'Árbol de Decisión'),
        (NOMBRE_RF, 'Regresión Logística'),
        (NOMBRE_RF, 'Preventivo tradicional (línea base)'),
    ]

    filas = []
    for a, b in pares:
        pa = np.array(res_map[a]['y_prob']).astype(float)
        pb = np.array(res_map[b]['y_prob']).astype(float)
        auc_a, auc_b, diff, z, p = delong_roc_test(y_true_f, pa, pb)
        filas.append({
            'Comparación': f'{a} vs. {b}',
            'AUC A': round(auc_a, 4),
            'AUC B': round(auc_b, 4),
            'Diferencia (A-B)': round(diff, 4),
            'z': round(z, 4),
            'Valor p': p,
            'Significativo (p<0.05)': 'Sí' if p < 0.05 else 'No',
        })

    return pd.DataFrame(filas)


# =============================================================================
# 4. CURVA PRECISION-RECALL COMPARATIVA
# =============================================================================

def generar_figura_precision_recall(resultados_lista: list,
                                       y_true: np.ndarray,
                                       ruta_salida: str) -> None:
    """
    Genera la curva Precision-Recall comparativa de los cinco enfoques
    evaluados, con la línea de referencia del clasificador aleatorio
    bajo la prevalencia observada (Saito & Rehmsmeier, 2015).
    """
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': '#dddddd', 'grid.linewidth': 0.5,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })
    NEGRO, GRIS_OS, GRIS_ME = '#1a1a1a', '#3d3d3d', '#767676'
    GRIS_CL, AZUL = '#bbbbbb', '#1B4F8A'

    estilos = {
        'Preventivo tradicional (línea base)': (GRIS_CL, (6, 3), 1.0),
        'Regresión Logística':                 (GRIS_ME, (4, 2), 1.3),
        'Árbol de Decisión':                   (GRIS_OS, (2, 2), 1.3),
        NOMBRE_RF:                              (AZUL,    None,   2.0),
        'Gradient Boosting':                   (NEGRO,   (3, 1), 1.5),
    }

    prevalencia = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    for r in resultados_lista:
        color, dash, lw = estilos.get(r['nombre'], (GRIS_ME, None, 1.2))
        prec, rec, _ = precision_recall_curve(y_true, r['y_prob'])
        ap = average_precision_score(y_true, r['y_prob'])
        ls = (0, dash) if dash else '-'
        ax.plot(rec, prec, color=color, lw=lw, linestyle=ls,
                label=f"{r['nombre'].split('(')[0].strip()}  (AP = {ap:.4f})")

    ax.axhline(prevalencia, color='#F0A500', linestyle='--', linewidth=1.5,
               label=f'Clasificador aleatorio  (AP = {prevalencia:.4f})')
    ax.set_xlabel('Sensibilidad (Recall)', fontsize=10)
    ax.set_ylabel('Precisión', fontsize=10)
    ax.set_title('Curvas Precision-Recall comparativas\n'
                  'Validación retrospectiva - Período 2025',
                  fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, frameon=True, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == '__main__':

    print('=' * 70)
    print('ANÁLISIS INFERENCIAL COMPLEMENTARIO - COMPARACIÓN DE CLASIFICADORES')
    print('=' * 70)

    os.makedirs(DIR_RESULTADOS, exist_ok=True)

    print('\n[1/4] Reproduciendo pipeline original (entrenamiento + evaluación)...')
    resultados_lista, y_true = reproducir_resultados()
    n_obs = len(y_true)

    print('\n[2/4] Aplicando prueba de McNemar (Random Forest vs. otros)...')
    df_mcnemar = ejecutar_mcnemar(resultados_lista, y_true)
    df_mcnemar.to_csv(f'{DIR_RESULTADOS}/tabla_mcnemar.csv', index=False)
    print(df_mcnemar.to_string(index=False))

    print('\n[3/4] Aplicando prueba de DeLong (Random Forest vs. otros)...')
    df_delong = ejecutar_delong(resultados_lista, y_true)
    df_delong.to_csv(f'{DIR_RESULTADOS}/tabla_delong.csv', index=False)
    print(df_delong.to_string(index=False))

    print('\n[4/4] Generando curva Precision-Recall...')
    generar_figura_precision_recall(
        resultados_lista, y_true,
        f'{DIR_RESULTADOS}/figura_30b_curva_precision_recall.png'
    )

    # --- RESUMEN EJECUTIVO ---
    with open(f'{DIR_RESULTADOS}/resumen_analisis_inferencial.txt', 'w',
              encoding='utf-8') as f:
        f.write('ANÁLISIS INFERENCIAL COMPLEMENTARIO\n')
        f.write('Comparación estadística formal de clasificadores\n')
        f.write(f'n = {n_obs} (validación 2025); '
                f'positivos = {int(y_true.sum())}; '
                f'negativos = {int((y_true == 0).sum())}\n')
        f.write('Umbral de clasificación: tau = 0.30\n\n')

        f.write('--- PRUEBA DE McNEMAR ---\n')
        f.write(df_mcnemar.to_string(index=False))
        f.write('\n\n--- PRUEBA DE DeLONG ---\n')
        f.write(df_delong.to_string(index=False))
        f.write('\n\n--- ANÁLISIS DE COSTO ESPERADO (puntos de cruce) ---\n')
        f.write(df_cruce[['Comparación', 'r* (punto de cruce)', 'Interpretación']]
                .to_string(index=False))

    print('\nArchivos generados en resultados/:')
    print('  - tabla_mcnemar.csv')
    print('  - tabla_delong.csv')
    print('  - figura_30b_curva_precision_recall.png')
    print('  - resumen_analisis_inferencial.txt')
    print('\nAnálisis inferencial complementario completado.')
