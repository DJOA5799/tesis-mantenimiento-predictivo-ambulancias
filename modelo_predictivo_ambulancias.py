"""
=============================================================================
FASE 3 Y 4: MODELADO COMPUTACIONAL Y VALIDACIÓN RETROSPECTIVA
Modelo computacional de mantenimiento predictivo para ambulancias Tipo II
Lima Metropolitana — Período 2024-2025

Metodología:
- Clasificación binaria supervisada
- Balanceo de clases con class_weight='balanced'
- Validación retrospectiva (backtesting temporal)
- Entrenamiento: datos 2024 / Validación: datos 2025

Modelos evaluados:
- Regresión Logística (modelo de referencia interpretable)
- Random Forest (modelo predictivo seleccionado)
- Gradient Boosting (modelo comparativo conservador)
- Mantenimiento preventivo tradicional (línea base operativa)

Métricas de evaluación (OE4 — tesis):
- Precisión, Sensibilidad, MAE, AUC-ROC, Disponibilidad proyectada

Referencias:
- Prytz (2014), Taoufyq et al. (2025), Si et al. (2011)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler

# =============================================================================
# PARÁMETROS
# =============================================================================

FEATURES = [
    'n_pm_w',
    'n_cm_w',
    'n_total_mant_w',
    'dias_desde_ultima_interv',
    'n_eventos_vehicular_w',
    'n_eventos_electrico_w',
    'n_eventos_equipamiento_w',
    'n_episodios_downtime_w',
    'downtime_total_dias_w',
    'downtime_promedio_dias_w',
    'disponibilidad_w',
    'km_en_w',
    'servicios_en_w',
    'equipamiento_funcional'
]

TARGET = 'y_inoperativa_14'

# =============================================================================
# FUNCIÓN 1: CARGAR Y PREPARAR DATOS
# =============================================================================

def cargar_datos(ruta_train: str, ruta_val: str):
    """
    Carga los datasets de entrenamiento y validación.
    Aplica preprocesamiento básico.
    """
    print("Cargando datos...")
    df_train = pd.read_csv(ruta_train)
    df_val = pd.read_csv(ruta_val)

    print(f"  Entrenamiento (2024): {len(df_train):,} observaciones")
    print(f"  Validación    (2025): {len(df_val):,} observaciones")

    # Verificar columnas necesarias
    cols_requeridas = FEATURES + [TARGET]
    for col in cols_requeridas:
        if col not in df_train.columns:
            raise ValueError(f"Columna faltante: {col}")

    # Separar features y target
    X_train = df_train[FEATURES].fillna(0)
    y_train = df_train[TARGET]
    X_val = df_val[FEATURES].fillna(0)
    y_val = df_val[TARGET]

    print(f"\n  Distribución Y en entrenamiento:")
    print(f"    Clase 0: {(y_train==0).sum():,} ({(y_train==0).mean()*100:.1f}%)")
    print(f"    Clase 1: {(y_train==1).sum():,} ({(y_train==1).mean()*100:.1f}%)")

    return X_train, y_train, X_val, y_val, df_train, df_val


# =============================================================================
# FUNCIÓN 2: BALANCEO MANUAL DE CLASES (sin imbalanced-learn)
# =============================================================================

def balancear_clases(X_train: pd.DataFrame,
                      y_train: pd.Series,
                      metodo: str = 'oversample',
                      random_state: int = 42) -> tuple:
    """
    Balanceo manual de clases por oversampling aleatorio.
    Alternativa a SMOTE cuando imbalanced-learn no está disponible.
    random_state: semilla para reproducibilidad del muestreo (default=42).

    metodo:
        'oversample' : duplica aleatoriamente muestras de la clase minoritaria
        'undersample': reduce aleatoriamente la clase mayoritaria
    """
    X = X_train.copy()
    y = y_train.copy()

    idx_0 = y[y == 0].index
    idx_1 = y[y == 1].index

    n_0 = len(idx_0)
    n_1 = len(idx_1)

    print(f"\n  Balanceo de clases ({metodo}):")
    print(f"    Antes  — Clase 0: {n_0:,} | Clase 1: {n_1:,}")

    rng = np.random.RandomState(random_state)  # semilla local para reproducibilidad
    if metodo == 'oversample':
        # Duplicar muestras de clase minoritaria hasta igualar mayoritaria
        idx_1_resampled = rng.choice(idx_1, size=n_0, replace=True)
        idx_balanced = np.concatenate([idx_0, idx_1_resampled])
    else:
        # Reducir clase mayoritaria al tamaño de la minoritaria
        idx_0_resampled = rng.choice(idx_0, size=n_1, replace=False)
        idx_balanced = np.concatenate([idx_0_resampled, idx_1])

    rng.shuffle(idx_balanced)

    X_bal = X.loc[idx_balanced].reset_index(drop=True)
    y_bal = y.loc[idx_balanced].reset_index(drop=True)

    print(f"    Después — Clase 0: {(y_bal==0).sum():,} | Clase 1: {(y_bal==1).sum():,}")

    return X_bal, y_bal


# =============================================================================
# FUNCIÓN 3: ENTRENAR MODELOS
# =============================================================================

def entrenar_modelos(X_train: pd.DataFrame,
                      y_train: pd.Series) -> dict:
    """
    Entrena tres modelos de clasificación supervisada:
    1. Regresión Logística (modelo de referencia interpretable)
    2. Random Forest (modelo predictivo seleccionado)
    3. Gradient Boosting (modelo comparativo conservador)

    La Regresión Logística y Random Forest incorporan class_weight='balanced'.
    Gradient Boosting se entrena sobre el conjunto balanceado por oversampling.
    """
    print("\nEntrenando modelos...")

    # Escalar features para Regresión Logística
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    modelos = {}

    # --- Modelo 1: Regresión Logística (línea base) ---
    print("  [1/3] Regresión Logística...")
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        C=0.1
    )
    lr.fit(X_train_scaled, y_train)
    modelos['Regresión Logística'] = {
        'modelo': lr,
        'scaler': scaler,
        'requiere_escala': True
    }

    # -------------------------------------------------------------------------
    # NOTA DE REPRODUCIBILIDAD
    # Los valores de referencia reportados en la tesis (Tabla 15) corresponden
    # a la ejecución de validación del 30/04/2026 con los siguientes resultados:
    #   GB  → AUC-ROC: 0.5551 | Sensibilidad: 0.5625 | VP=90, FP=748, VN=842, FN=70
    #   RF  → AUC-ROC: 0.5514 | Sensibilidad: 0.8250 | VP=132, FP=1193, VN=397, FN=28
    #   RL  → AUC-ROC: 0.5498 | Sensibilidad: 1.0000 | VP=160, FP=1580, VN=10, FN=0
    # Importancia de variables (RF): servicios_en_w=18.99%, km_en_w=18.58%
    # Umbral de clasificación: τ = 0.30
    # Estos valores son los canónicos del documento de tesis.
    # Pequeñas variaciones en ejecuciones futuras son esperables por la
    # naturaleza estocástica del oversampling; la interpretación cualitativa
    # (orden de modelos, variables más importantes) permanece estable.
    # -------------------------------------------------------------------------

    # --- Modelo 2: Random Forest (modelo predictivo seleccionado) ---
    print("  [2/3] Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    modelos['Random Forest'] = {
        'modelo': rf,
        'scaler': None,
        'requiere_escala': False
    }

    # --- Modelo 3: Gradient Boosting (modelo comparativo conservador) ---
    print("  [3/3] Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train, y_train)
    modelos['Gradient Boosting'] = {
        'modelo': gb,
        'scaler': None,
        'requiere_escala': False
    }

    print("  Modelos entrenados exitosamente.")
    return modelos


# =============================================================================
# FUNCIÓN 4: EVALUAR MODELOS — MÉTRICAS OE4
# =============================================================================

def evaluar_modelo(nombre: str,
                    info_modelo: dict,
                    X_val: pd.DataFrame,
                    y_val: pd.Series,
                    umbral: float = 0.3) -> dict:
    """
    Evalúa un modelo con las métricas declaradas en el OE4 de la tesis:
    - Precisión
    - Sensibilidad (Recall)
    - MAE
    - AUC-ROC
    - Disponibilidad operativa proyectada

    umbral: punto de corte para clasificación (default 0.3 por desbalance)
    """
    modelo = info_modelo['modelo']
    scaler = info_modelo['scaler']
    requiere_escala = info_modelo['requiere_escala']

    # Preparar datos
    if requiere_escala:
        X_eval = scaler.transform(X_val)
    else:
        X_eval = X_val.values

    # Probabilidades y predicciones
    y_prob = modelo.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob >= umbral).astype(int)

    # --- Métricas OE4 ---
    precision = precision_score(y_val, y_pred, zero_division=0)
    sensibilidad = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_val, y_prob)

    # MAE sobre probabilidades (distancia entre probabilidad estimada y valor real)
    mae = mean_absolute_error(y_val, y_prob)

    # Disponibilidad operativa proyectada
    # Porcentaje de unidades clasificadas como operativas (y_pred=0)
    disponibilidad_proyectada = (y_pred == 0).mean()

    # Disponibilidad real observada
    disponibilidad_real = (y_val == 0).mean()

    # Matriz de confusión
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    exactitud = (tp + tn) / (tp + tn + fp + fn)
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0 

    resultados = {
        'nombre': nombre,
        'precision': round(precision, 4),
        'sensibilidad': round(sensibilidad, 4),
        'exactitud': round(exactitud, 4),
        'especificidad': round(especificidad, 4),
        'f1_score': round(f1, 4),
        'auc_roc': round(auc_roc, 4),
        'mae': round(mae, 4),
        'disponibilidad_proyectada': round(disponibilidad_proyectada, 4),
        'disponibilidad_real': round(disponibilidad_real, 4),
        'verdaderos_positivos': int(tp),
        'falsos_positivos': int(fp),
        'verdaderos_negativos': int(tn),
        'falsos_negativos': int(fn),
        'y_prob': y_prob,
        'y_pred': y_pred
    }

    return resultados


# =============================================================================
# FUNCIÓN 5: COMPARACIÓN CON LÍNEA BASE PREVENTIVA
# =============================================================================

def calcular_linea_base(y_val: pd.Series,
                          X_val: pd.DataFrame) -> dict:
    """
    Simula el esquema de mantenimiento preventivo tradicional como
    línea base de comparación (OE4).

    Lógica: El esquema preventivo interviene cada ~37 días sin considerar
    el estado real del activo. Se modela como un clasificador que predice
    inoperatividad cuando días_desde_ultima_interv > 30 días.
    """
    # El preventivo "alerta" cuando han pasado más de 30 días sin intervención
    umbral_dias = 30
    y_pred_preventivo = (X_val['dias_desde_ultima_interv'] > umbral_dias).astype(int)

    precision = precision_score(y_val, y_pred_preventivo, zero_division=0)
    sensibilidad = recall_score(y_val, y_pred_preventivo, zero_division=0)
    f1 = f1_score(y_val, y_pred_preventivo, zero_division=0)

    # AUC-ROC usando días como score inverso (más días = más riesgo)
    score_preventivo = X_val['dias_desde_ultima_interv'] / X_val['dias_desde_ultima_interv'].max()
    try:
        auc_roc = roc_auc_score(y_val, score_preventivo)
    except Exception:
        auc_roc = 0.5

    mae = mean_absolute_error(y_val, y_pred_preventivo.astype(float))
    disponibilidad_proyectada = (y_pred_preventivo == 0).mean()
    disponibilidad_real = (y_val == 0).mean()

    cm = confusion_matrix(y_val, y_pred_preventivo)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    exactitud = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'nombre': 'Preventivo tradicional (línea base)',
        'precision': round(precision, 4),
        'sensibilidad': round(sensibilidad, 4),
        'exactitud': round(exactitud, 4),
        'especificidad': round(especificidad, 4),
        'f1_score': round(f1, 4),
        'auc_roc': round(auc_roc, 4),
        'mae': round(mae, 4),
        'disponibilidad_proyectada': round(disponibilidad_proyectada, 4),
        'disponibilidad_real': round(disponibilidad_real, 4),
        'verdaderos_positivos': int(tp),
        'falsos_positivos': int(fp),
        'verdaderos_negativos': int(tn),
        'falsos_negativos': int(fn),
        'y_prob': score_preventivo.values,
        'y_pred': y_pred_preventivo.values
    }


# =============================================================================
# FUNCIÓN 6: IMPORTANCIA DE VARIABLES (Random Forest)
# =============================================================================

def analizar_importancia_variables(modelos: dict,
                                   modelo_base: str = 'Random Forest') -> pd.DataFrame:
    """
    Extrae la importancia de variables del modelo seleccionado.
    Por defecto utiliza Random Forest, modelo predictivo seleccionado para el soporte preventivo.
    """
    info_modelo = modelos[modelo_base]
    modelo = info_modelo['modelo']

    importancias = pd.DataFrame({
        'variable': FEATURES,
        'importancia': modelo.feature_importances_
    }).sort_values('importancia', ascending=False).reset_index(drop=True)

    importancias['modelo_base'] = modelo_base

    importancias['importancia_pct'] = (
        importancias['importancia'] / importancias['importancia'].sum() * 100
    ).round(2)

    return importancias


# =============================================================================
# FUNCIÓN 7: GENERAR TABLA DE RESULTADOS COMPARATIVA
# =============================================================================

def generar_tabla_resultados(resultados_lista: list) -> pd.DataFrame:
    """
    Genera tabla comparativa de métricas para todos los modelos.
    """
    filas = []
    for r in resultados_lista:
        filas.append({
            'Modelo': r['nombre'],
            'Precisión': f"{r['precision']:.4f}",
            'Sensibilidad': f"{r['sensibilidad']:.4f}",
            'Especificidad': f"{r['especificidad']:.4f}",
            'Exactitud': f"{r['exactitud']:.4f}",
            'F1-Score': f"{r['f1_score']:.4f}",
            'AUC-ROC': f"{r['auc_roc']:.4f}",
            'MAE': f"{r['mae']:.4f}",
            'Disp. Proyectada': f"{r['disponibilidad_proyectada']:.4f}",
            'VP': r['verdaderos_positivos'],
            'FP': r['falsos_positivos'],
            'VN': r['verdaderos_negativos'],
            'FN': r['falsos_negativos']
        })
    return pd.DataFrame(filas)


# =============================================================================
# FUNCIÓN 8: VISUALIZACIONES
# =============================================================================

def generar_visualizaciones(resultados_lista: list,
                              importancias: pd.DataFrame,
                              y_val: pd.Series) -> None:
    """
    Genera las visualizaciones principales para la tesis:
    1. Curvas ROC comparativas
    2. Importancia de variables (Top 10, valores canónicos)
    3. Comparación de métricas
    4. Distribución de probabilidades predichas (Gradient Boosting)

    Estilo científico — serif + escala de grises + azul para RF.
    Valores canónicos de referencia (Tabla 15 y Tabla 16 de la tesis):
      GB  AUC-ROC=0.5551, Sensibilidad=0.5625
      RF  AUC-ROC=0.5514, Sensibilidad=0.8250
      RL  AUC-ROC=0.5498, Sensibilidad=1.0000
      Preventivo AUC-ROC=0.5038
    """
    import matplotlib
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': '#dddddd', 'grid.linewidth': 0.5,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })
    NEGRO   = '#1a1a1a'; GRIS_OS = '#3d3d3d'; GRIS_ME = '#767676'
    GRIS_CL = '#bbbbbb'; AZUL    = '#1B4F8A'; ROJO    = '#7B1515'
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        'Resultados de la validación retrospectiva del modelo computacional predictivo\n'
        'Ambulancias Tipo II — Lima Metropolitana 2024-2025',
        fontsize=12, fontweight='bold', y=1.01
    )

    # --- GRÁFICO 1: Curvas ROC con valores canónicos de la tesis ---
    ax1 = axes[0, 0]
    ax1.plot([0, 1], [0, 1], color=GRIS_CL, lw=1.0, linestyle='--',
             label='Clasificador aleatorio  (AUC = 0,50)', zorder=1)
    estilos_roc = {
        'Preventivo tradicional (línea base)': (GRIS_CL, (6, 3), 1.0),
        'Regresión Logística':                 (GRIS_ME, (4, 2), 1.3),
        'Random Forest':                       (AZUL,    (3, 1), 1.5),
        'Gradient Boosting':                   (NEGRO,   None,   2.0),
    }
    for r in resultados_lista:
        color, dash, lw = estilos_roc.get(r['nombre'], (GRIS_ME, None, 1.2))
        fpr, tpr, _ = roc_curve(y_val, r['y_prob'])
        ls = (0, dash) if dash else '-'
        ax1.plot(fpr, tpr, color=color, lw=lw, linestyle=ls,
                 label=f"{r['nombre'].split('(')[0].strip()}  (AUC = {r['auc_roc']:.4f})",
                 zorder=3)
    ax1.set_xlabel('Tasa de Falsos Positivos  (1 – Especificidad)', fontsize=9)
    ax1.set_ylabel('Sensibilidad  (TVP)', fontsize=9)
    ax1.set_title('Curvas ROC — Comparación de modelos', fontsize=10, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=7.5, frameon=True)
    ax1.grid(True, alpha=0.30)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)
    ax1.set_aspect('equal')

    # --- GRÁFICO 2: Importancia de variables del modelo seleccionado ---
    ax2 = axes[0, 1]
    top_imp = importancias.head(10).sort_values('importancia_pct', ascending=True).copy()

    nombres_legibles = {
        'km_en_w': 'Kilometraje en W',
        'servicios_en_w': 'Servicios en W',
        'dias_desde_ultima_interv': 'Días desde última intervención',
        'disponibilidad_w': 'Disponibilidad en W',
        'downtime_promedio_dias_w': 'Downtime promedio en W',
        'downtime_total_dias_w': 'Downtime total en W',
        'n_eventos_electrico_w': 'Eventos eléctricos en W',
        'n_eventos_equipamiento_w': 'Eventos de equipamiento en W',
        'n_total_mant_w': 'Mantenimientos totales en W',
        'n_pm_w': 'Mantenimientos preventivos en W',
        'n_cm_w': 'Mantenimientos correctivos en W',
        'n_eventos_vehicular_w': 'Eventos vehiculares en W',
        'n_episodios_downtime_w': 'Episodios de downtime en W',
        'equipamiento_funcional': 'Equipamiento funcional'
    }

    top_imp['variable_legible'] = top_imp['variable'].map(nombres_legibles).fillna(top_imp['variable'])

    vlbls = top_imp['variable_legible'].tolist()
    vimps = top_imp['importancia_pct'].tolist()

    bars2 = ax2.barh(range(len(vlbls)), vimps,
                    color=NEGRO, edgecolor='white',
                    linewidth=0.8, height=0.62, zorder=3)

    for bar, val in zip(bars2, vimps):
        ax2.text(val + 0.15, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}%', va='center', fontsize=8,
                color=NEGRO, fontweight='bold')

    ax2.set_yticks(range(len(vlbls)))
    ax2.set_yticklabels(vlbls, fontsize=8)
    ax2.set_xlabel('Importancia relativa (%)', fontsize=9)
    ax2.set_title('Importancia de variables - Random Forest (Top 10)',
                fontsize=10, fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.30, zorder=0)
    ax2.set_xlim(0, max(vimps) * 1.18)

    # --- GRÁFICO 3: Comparativa de métricas ---
    ax3 = axes[1, 0]
    mets_keys = ['exactitud', 'especificidad', 'precision', 'sensibilidad', 'f1_score', 'auc_roc']
    etiq_met  = ['Exactitud', 'Especificidad', 'Precisión', 'Sensibilidad', 'F1-Score', 'AUC-ROC']
    x = np.arange(len(mets_keys))
    n_m   = len(resultados_lista)
    ancho = 0.72 / n_m
    cols_m  = [GRIS_CL, GRIS_ME, AZUL, NEGRO]
    htchs_m = ['', '///', '\\', '']
    for i, r in enumerate(resultados_lista):
        off  = (i - n_m / 2 + 0.5) * ancho
        vals = [r[m] * 100 for m in mets_keys]
        b = ax3.bar(x + off, vals, ancho * 0.92,
                    color=cols_m[min(i, len(cols_m)-1)],
                    edgecolor='white', linewidth=0.8,
                    label=r['nombre'].split('(')[0].strip(), zorder=3)
        for bar in b:
            bar.set_hatch(htchs_m[min(i, len(htchs_m)-1)])
    ax3.set_xticks(x)
    ax3.set_xticklabels(etiq_met, fontsize=9)
    ax3.set_ylabel('Valor (%)', fontsize=9)
    ax3.set_ylim(0, 110)
    ax3.set_title('Perfil de desempeño de los modelos evaluados - Periodo 2025 - Umbral 0,30',
                    fontsize=10, fontweight='bold')
    ax3.legend(fontsize=7.5, loc='upper right', frameon=True)
    ax3.grid(True, axis='y', alpha=0.30, zorder=0)

    # --- GRÁFICO 4: Distribución de probabilidades (Random Forest) ---
    ax4 = axes[1, 1]
    rf_res = next((r for r in resultados_lista if 'Random Forest' in r['nombre']), None)
    if rf_res:
        y_arr = np.array(y_val)
        ax4.hist(rf_res['y_prob'][y_arr == 0], bins=30, alpha=0.65,
                color=AZUL, label='Sin inoperatividad (Y=0)',
                density=True, edgecolor='white')
        ax4.hist(rf_res['y_prob'][y_arr == 1], bins=30, alpha=0.65,
                color=ROJO, label='Con inoperatividad (Y=1)',
                density=True, edgecolor='white')
        ax4.axvline(x=0.30, color='#F0A500', linestyle='--',
                    linewidth=2, label='Umbral = 0,30')
        ax4.set_xlabel('Probabilidad predicha de inoperatividad', fontsize=9)
        ax4.set_ylabel('Densidad', fontsize=9)
        ax4.set_title('Distribución de probabilidades predichas - Random Forest',
                    fontsize=10, fontweight='bold')
        ax4.legend(fontsize=8, frameon=True)
        ax4.grid(True, alpha=0.30)
    
    top_imp['variable_legible'] = top_imp['variable'].map(nombres_legibles).fillna(top_imp['variable'])
    vlbls = top_imp['variable_legible'].tolist()
    plt.tight_layout()
    plt.savefig('resultados_modelo.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("\n  Figura guardada: resultados_modelo.png")
    plt.close()


# =============================================================================
# FUNCIÓN 9: GENERAR SALIDAS DE SOPORTE PREVENTIVO (Entregable 2)
# =============================================================================

def generar_soporte_preventivo(df_val: pd.DataFrame,
                                 modelos: dict,
                                 X_val: pd.DataFrame,
                                 modelo_base: str = 'Random Forest',
                                 top_n: int = 10) -> pd.DataFrame:
    """
    Genera el ranking de ambulancias por nivel de riesgo operativo.
    Corresponde al Entregable 2 de la tesis: prototipo de soporte preventivo.

    Por defecto utiliza Random Forest, seleccionado para el soporte preventivo
    por su mayor sensibilidad operativa y capacidad para identificar eventos
    reales de inoperatividad en el horizonte de 14 días.

    Niveles de riesgo:
        ALTO   : probabilidad >= 0.50
        MEDIO  : probabilidad >= 0.25
        BAJO   : probabilidad < 0.25
    """
    info_modelo = modelos[modelo_base]
    modelo = info_modelo['modelo']
    scaler = info_modelo['scaler']
    requiere_escala = info_modelo['requiere_escala']

    if requiere_escala:
        X_eval = scaler.transform(X_val)
    else:
        X_eval = X_val.values

    y_prob = modelo.predict_proba(X_eval)[:, 1]

    df_soporte = df_val[['id_ambulancia', 't0']].copy()
    df_soporte['modelo_base'] = modelo_base
    df_soporte['prob_inoperatividad'] = y_prob.round(4)

    # Asignar nivel de riesgo
    def nivel_riesgo(p):
        if p >= 0.50:
            return 'ALTO'
        elif p >= 0.25:
            return 'MEDIO'
        else:
            return 'BAJO'

    df_soporte['nivel_riesgo'] = df_soporte['prob_inoperatividad'].apply(nivel_riesgo)

    # Añadir variables explicativas clave para soporte preventivo
    df_soporte['dias_desde_ultima_interv'] = X_val['dias_desde_ultima_interv'].values
    df_soporte['downtime_total_dias_w'] = X_val['downtime_total_dias_w'].values
    df_soporte['n_cm_w'] = X_val['n_cm_w'].values
    df_soporte['disponibilidad_w'] = X_val['disponibilidad_w'].values
    df_soporte['equipamiento_funcional'] = X_val['equipamiento_funcional'].values
    df_soporte['km_en_w'] = X_val['km_en_w'].values
    df_soporte['servicios_en_w'] = X_val['servicios_en_w'].values

    # Ordenar por probabilidad descendente
    df_soporte = df_soporte.sort_values('prob_inoperatividad', ascending=False)

    return df_soporte


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    print("="*60)
    print("FASE 3-4: MODELADO Y VALIDACIÓN RETROSPECTIVA")
    print("Ambulancias Tipo II — Lima Metropolitana")
    print("="*60)

    # --- CARGAR DATOS ---
    X_train, y_train, X_val, y_val, df_train, df_val = cargar_datos(
        'datos_simulados/dataset_entrenamiento_2024.csv',
        'datos_simulados/dataset_validacion_2025.csv'
    )

    # --- BALANCEO DE CLASES ---
    X_train_bal, y_train_bal = balancear_clases(X_train, y_train, metodo='oversample')

    # --- ENTRENAR MODELOS ---
    modelos = entrenar_modelos(X_train_bal, y_train_bal)

    # --- EVALUAR MODELOS EN VALIDACIÓN 2025 ---
    print("\nEvaluando modelos en datos de validación (2025)...")
    umbral = 0.30

    resultados_lista = []

    # Línea base preventiva
    lb = calcular_linea_base(y_val, X_val)
    resultados_lista.append(lb)

    # Modelos predictivos
    for nombre, info in modelos.items():
        r = evaluar_modelo(nombre, info, X_val, y_val, umbral=umbral)
        resultados_lista.append(r)

    # --- TABLA COMPARATIVA DE RESULTADOS ---
    print("\n" + "="*60)
    print("TABLA DE RESULTADOS COMPARATIVOS (Umbral = 0.30)")
    print("="*60)
    tabla = generar_tabla_resultados(resultados_lista)
    print(tabla.to_string(index=False))

    # --- IMPORTANCIA DE VARIABLES ---
    print("\n" + "="*60)
    print("IMPORTANCIA DE VARIABLES — Random Forest (Top 10)")
    print("="*60)
    importancias = analizar_importancia_variables(modelos, modelo_base='Random Forest')
    print(importancias.head(10).to_string(index=False))

    # --- IDENTIFICAR MEJOR MODELO ---
    modelos_pred = [r for r in resultados_lista if 'línea base' not in r['nombre']]
    mejor_modelo = max(modelos_pred, key=lambda x: x['auc_roc'])
    print(f"\n{'='*60}")
    print(f"MODELO CON MAYOR AUC-ROC: {mejor_modelo['nombre']}")
    print("MODELO PREDICTIVO SELECCIONADO PARA SOPORTE PREVENTIVO: Random Forest")
    print(f"  AUC-ROC    : {mejor_modelo['auc_roc']:.4f}")
    print(f"  Precisión  : {mejor_modelo['precision']:.4f}")
    print(f"  Sensibilidad: {mejor_modelo['sensibilidad']:.4f}")
    print(f"  MAE        : {mejor_modelo['mae']:.4f}")
    print(f"  Disp. proy.: {mejor_modelo['disponibilidad_proyectada']:.4f}")

    lb_res = resultados_lista[0]
    rf_res = next(r for r in resultados_lista if r['nombre'] == 'Random Forest')
    print(f"\nCOMPARACIÓN DEL MODELO SELECCIONADO FRENTE A LA LÍNEA BASE PREVENTIVA:")
    print(f"  AUC-ROC    : {lb_res['auc_roc']:.4f} → {rf_res['auc_roc']:.4f} "
          f"(+{rf_res['auc_roc']-lb_res['auc_roc']:.4f})")
    print(f"  Sensibilidad: {lb_res['sensibilidad']:.4f} → {rf_res['sensibilidad']:.4f} "
          f"(+{rf_res['sensibilidad']-lb_res['sensibilidad']:.4f})")

    # --- SOPORTE PREVENTIVO (Entregable 2) ---
    print(f"\n{'='*60}")
    print("PROTOTIPO DE SOPORTE PREVENTIVO")
    print("Ranking de ambulancias por nivel de riesgo (Top 15)")
    print("="*60)
    df_soporte = generar_soporte_preventivo(
    df_val, modelos, X_val, modelo_base='Random Forest'
    )

    # Mostrar corte más reciente
    ultimo_corte = df_soporte['t0'].max()
    df_corte = df_soporte[df_soporte['t0'] == ultimo_corte].copy()
    df_ultimo = df_corte.head(15)

    print(f"\nCorte temporal: {ultimo_corte}")
    print(df_ultimo[[
        'id_ambulancia', 'prob_inoperatividad', 'nivel_riesgo',
        'dias_desde_ultima_interv', 'downtime_total_dias_w',
        'n_cm_w', 'disponibilidad_w'
    ]].to_string(index=False))

    resumen_riesgo = df_corte['nivel_riesgo'].value_counts()
    print(f"\nResumen de riesgo al {ultimo_corte}:")
    for nivel in ['ALTO', 'MEDIO', 'BAJO']:
        n = resumen_riesgo.get(nivel, 0)
        print(f"  {nivel:5s}: {n:2d} ambulancias ({n/len(df_corte)*100:.0f}%)")

    # --- VISUALIZACIONES ---
    print(f"\n{'='*60}")
    print("Generando visualizaciones...")
    generar_visualizaciones(resultados_lista, importancias, y_val)

    # --- EXPORTAR RESULTADOS ---
    print("\nExportando resultados...")
    import os
    os.makedirs("resultados", exist_ok=True)

    tabla.to_csv("resultados/tabla_metricas_comparativas.csv", index=False)
    importancias.to_csv("resultados/importancia_variables.csv", index=False)
    df_soporte.to_csv("resultados/soporte_preventivo_completo.csv", index=False)

    print(f"  tabla_metricas_comparativas.csv")
    print(f"  importancia_variables.csv")
    print(f"  soporte_preventivo_completo.csv")
    print(f"  resultados_modelo.png")

    print("\n¡Fase 3-4 completada exitosamente!")
    print("Siguiente paso: Fase 5 — Lineamientos técnicos y redacción de resultados")
