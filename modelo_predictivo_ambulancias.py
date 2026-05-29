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
- Exactitud, especificidad, precisión, sensibilidad, F1-Score, MAE, AUC-ROC y disponibilidad proyectada

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
# FUNCIÓN 8B: FIGURAS INDIVIDUALES PARA TESIS
# =============================================================================

def _configurar_estilo_figuras_tesis():
    """
    Configura un estilo visual sobrio, institucional y consistente
    para las figuras individuales de la tesis.
    """
    import matplotlib
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': '#E6E6E6',
        'grid.linewidth': 0.6,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.titleweight': 'bold',
        'savefig.facecolor': 'white'
    })


def _asegurar_directorio_figuras():
    """Crea el directorio de salida para figuras individuales de tesis."""
    import os
    os.makedirs("figuras_tesis", exist_ok=True)


def _nombre_corto_modelo(nombre: str) -> str:
    """Devuelve nombres cortos para etiquetas de gráficos."""
    if 'Preventivo' in nombre:
        return 'Preventivo\n(base)'
    if 'Regresión' in nombre:
        return 'Reg.\nLogística'
    if 'Random Forest' in nombre:
        return 'Random\nForest'
    if 'Gradient' in nombre:
        return 'Gradient\nBoosting'
    return nombre


def _mapear_variable_legible(variable: str) -> str:
    """Convierte nombres técnicos de variables a etiquetas legibles."""
    nombres_legibles = {
        'n_pm_w': 'Mantenimientos preventivos en W',
        'n_cm_w': 'Mantenimientos correctivos en W',
        'n_total_mant_w': 'Mantenimientos totales en W',
        'dias_desde_ultima_interv': 'Días desde última intervención',
        'n_eventos_vehicular_w': 'Eventos vehiculares en W',
        'n_eventos_electrico_w': 'Eventos eléctricos en W',
        'n_eventos_equipamiento_w': 'Eventos de equipamiento en W',
        'n_episodios_downtime_w': 'Episodios de downtime en W',
        'downtime_total_dias_w': 'Downtime total en W',
        'downtime_promedio_dias_w': 'Downtime promedio en W',
        'disponibilidad_w': 'Disponibilidad en W',
        'km_en_w': 'Kilometraje en W',
        'servicios_en_w': 'Servicios prestados en W',
        'equipamiento_funcional': 'Equipamiento funcional'
    }
    return nombres_legibles.get(variable, variable)

def generar_figura_03_sigmoide_regresion_logistica() -> None:
    """
    Genera la Figura 3:
    Función sigmoide empleada en la regresión logística para transformar
    una combinación lineal de variables en probabilidad.

    Esta figura es conceptual y no depende del dataset de la tesis.
    """
    _configurar_estilo_figuras_tesis()
    _asegurar_directorio_figuras()

    AZUL = '#355C7D'
    GRIS_OSCURO = '#4B5358'
    GRIS_MEDIO = '#8E969B'
    GRIS_CLARO = '#E6E6E6'
    NEGRO = '#1F2326'

    z = np.linspace(-8, 8, 600)
    sigmoide = 1 / (1 + np.exp(-z))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    # Curva sigmoide
    ax.plot(
        z,
        sigmoide,
        color=NEGRO,
        linewidth=2.4,
        label=r'$\sigma(z)=\frac{1}{1+e^{-z}}$'
    )

    # Regiones de clasificación para umbral 0.50
    ax.fill_between(
        z,
        sigmoide,
        1,
        where=(z >= 0),
        color=AZUL,
        alpha=0.08,
        label=r'Región $\hat{y}=1$  ($\tau=0,50$)'
    )

    ax.fill_between(
        z,
        0,
        sigmoide,
        where=(z < 0),
        color=GRIS_MEDIO,
        alpha=0.10,
        label=r'Región $\hat{y}=0$  ($\tau=0,50$)'
    )

    # Líneas de referencia
    ax.axhline(
        y=0.50,
        color=GRIS_MEDIO,
        linestyle='--',
        linewidth=1.1
    )

    ax.axvline(
        x=0,
        color=GRIS_MEDIO,
        linestyle='--',
        linewidth=1.1
    )

    # Punto central
    ax.scatter(
        [0],
        [0.5],
        color=NEGRO,
        s=35,
        zorder=5
    )

    ax.annotate(
        r'$\sigma(0)=0,5$',
        xy=(0, 0.5),
        xytext=(1.3, 0.39),
        arrowprops=dict(
            arrowstyle='->',
            color=GRIS_OSCURO,
            lw=1.0
        ),
        fontsize=10,
        color=GRIS_OSCURO
    )

    ax.text(
        0.15,
        0.07,
        r'$z=0$',
        fontsize=9,
        color=GRIS_MEDIO
    )

    # Etiquetas y formato
    ax.set_title(
        'Función sigmoide en la regresión logística',
        fontsize=12,
        fontweight='bold'
    )

    ax.set_xlabel(
        r'Combinación lineal  $z$',
        fontsize=10
    )

    ax.set_ylabel(
        r'Probabilidad  $\hat{P}(Y=1 \mid x)$',
        fontsize=10
    )

    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.05, 1.05)

    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(['0', '0,25', '0,50', '0,75', '1,00'])

    ax.grid(True, alpha=0.35)
    ax.legend(
        fontsize=8.5,
        frameon=True,
        loc='lower right'
    )

    plt.tight_layout()
    plt.savefig(
        "figuras_tesis/figura_03_sigmoide_regresion_logistica.png",
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close()

    print("  Figura guardada: figuras_tesis/figura_03_sigmoide_regresion_logistica.png")

def generar_figura_04_balanceo_clases(df_train: pd.DataFrame) -> None:
    """
    Genera la Figura 4:
    Distribución de clases antes y después del balanceo mediante
    oversampling aleatorio con reemplazo.

    La figura se calcula directamente desde el conjunto de entrenamiento 2024,
    por lo que debe coincidir con la partición final usada por el modelo.
    """
    _configurar_estilo_figuras_tesis()
    _asegurar_directorio_figuras()

    AZUL = '#355C7D'
    ROJO = '#C96B63'
    GRIS_OSCURO = '#4B5358'
    GRIS_CLARO = '#E6E6E6'

    conteos_originales = df_train[TARGET].value_counts().sort_index()
    n_clase_0 = int(conteos_originales.get(0, 0))
    n_clase_1 = int(conteos_originales.get(1, 0))

    n_balanceado = max(n_clase_0, n_clase_1)
    conteos_balanceados = [n_balanceado, n_balanceado]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))

    # -------------------------------------------------------------------------
    # Panel izquierdo: distribución original
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    valores_originales = [n_clase_0, n_clase_1]
    colores = [AZUL, ROJO]

    barras1 = ax1.bar(
        [0, 1],
        valores_originales,
        color=colores,
        edgecolor='white',
        linewidth=0.8,
        width=0.55
    )

    total_original = sum(valores_originales)

    for barra, valor, color in zip(barras1, valores_originales, colores):
        ax1.text(
            barra.get_x() + barra.get_width() / 2,
            valor + total_original * 0.02,
            f"{valor:,}\n({valor / total_original * 100:.1f}%)",
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold',
            color=color
        )

    ax1.set_title(
        'Dataset de entrenamiento\n(distribución original)',
        fontsize=10,
        fontweight='bold'
    )
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Y = 0\n(operativa)', 'Y = 1\n(inoperativa)'], fontsize=8)
    ax1.set_ylabel('Número de observaciones', fontsize=9)
    ax1.grid(True, axis='y', alpha=0.35)

    desbalance = n_clase_0 / n_clase_1 if n_clase_1 > 0 else np.nan
    ax1.text(
        0.5,
        max(valores_originales) * 0.12,
        f"Desbalance ≈ {desbalance:.1f}:1",
        ha='center',
        fontsize=8,
        color=GRIS_OSCURO,
        style='italic'
    )

    # -------------------------------------------------------------------------
    # Panel derecho: distribución posterior al oversampling
    # -------------------------------------------------------------------------
    ax2 = axes[1]

    barras2 = ax2.bar(
        [0, 1],
        conteos_balanceados,
        color=colores,
        edgecolor='white',
        linewidth=0.8,
        width=0.55
    )

    total_balanceado = sum(conteos_balanceados)

    for barra, valor, color in zip(barras2, conteos_balanceados, colores):
        ax2.text(
            barra.get_x() + barra.get_width() / 2,
            valor + total_balanceado * 0.015,
            f"{valor:,}\n({valor / total_balanceado * 100:.1f}%)",
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold',
            color=color
        )

    ax2.set_title(
        'Dataset de entrenamiento\n(después del oversampling)',
        fontsize=10,
        fontweight='bold'
    )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Y = 0\n(operativa)', 'Y = 1\n(inoperativa)'], fontsize=8)
    ax2.set_ylabel('Número de observaciones', fontsize=9)
    ax2.grid(True, axis='y', alpha=0.35)

    ax2.text(
        0.5,
        n_balanceado * 0.12,
        "Ratio 1:1",
        ha='center',
        fontsize=8,
        color=GRIS_OSCURO,
        style='italic'
    )

    plt.suptitle(
        'Distribución de clases antes y después del balanceo\n'
        'mediante oversampling aleatorio con reemplazo',
        fontsize=11,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(
        "figuras_tesis/figura_04_balanceo_clases.png",
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close()

    print("  Figura guardada: figuras_tesis/figura_04_balanceo_clases.png")


def generar_figura_05_arbol_decision_ilustrativo() -> None:
    """
    Genera la Figura 5:
    Esquema ilustrativo de reglas de decisión aplicadas al problema
    de clasificación de inoperatividad.

    Esta figura es conceptual. No representa un árbol individual extraído
    del Random Forest final, sino la lógica general de particionamiento
    mediante reglas condicionales.
    """
    _configurar_estilo_figuras_tesis()
    _asegurar_directorio_figuras()

    from matplotlib.patches import FancyBboxPatch

    AZUL = '#355C7D'
    GRIS_OSCURO = '#4B5358'
    GRIS_MEDIO = '#8E969B'
    ROJO = '#C96B63'
    AMARILLO = '#F4D35E'
    VERDE = '#78BFA3'

    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    def caja(x, y, w, h, texto, color, fs=8.4, text_color='white'):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.05',
            facecolor=color,
            edgecolor='white',
            linewidth=1.3
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            texto,
            ha='center',
            va='center',
            fontsize=fs,
            fontweight='bold',
            color=text_color
        )

    def flecha(x1, y1, x2, y2, texto=None, dx=0, dy=0):
        ax.annotate(
            '',
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->',
                lw=1.2,
                color=GRIS_MEDIO
            )
        )
        if texto:
            ax.text(
                (x1 + x2) / 2 + dx,
                (y1 + y2) / 2 + dy,
                texto,
                fontsize=7.2,
                color=GRIS_MEDIO,
                style='italic',
                ha='center'
            )

    caja(3.65, 6.75, 2.7, 0.75, "Variable de uso operativo\nsupera umbral", AZUL)
    caja(1.0, 5.25, 2.8, 0.75, "Días desde última\nintervención elevados", GRIS_OSCURO)
    caja(6.2, 5.25, 2.8, 0.75, "Disponibilidad reciente\nreducida", GRIS_OSCURO)

    caja(0.35, 3.65, 2.1, 0.75, "BAJO\nriesgo", VERDE)
    caja(3.0, 3.65, 2.1, 0.75, "Downtime acumulado\nelevado", GRIS_OSCURO)
    caja(6.0, 3.65, 2.1, 0.75, "MEDIO\nriesgo", AMARILLO)
    caja(8.15, 3.65, 1.5, 0.75, "ALTO\nriesgo", ROJO)

    caja(2.55, 2.1, 1.85, 0.75, "MEDIO\nriesgo", AMARILLO)
    caja(4.75, 2.1, 1.85, 0.75, "ALTO\nriesgo", ROJO)

    flecha(5.0, 6.75, 2.4, 6.0, "No", dx=-0.15, dy=0.08)
    flecha(5.0, 6.75, 7.6, 6.0, "Sí", dx=0.15, dy=0.08)

    flecha(2.4, 5.25, 1.4, 4.4, "No", dx=-0.15)
    flecha(2.4, 5.25, 4.05, 4.4, "Sí", dx=0.10)

    flecha(7.6, 5.25, 6.95, 4.4, "No", dx=-0.10)
    flecha(7.6, 5.25, 8.85, 4.4, "Sí", dx=0.10)

    flecha(4.05, 3.65, 3.45, 2.85, "No", dx=-0.10)
    flecha(4.05, 3.65, 5.65, 2.85, "Sí", dx=0.10)

    ax.text(
        5.0,
        0.95,
        "Nodo de decisión: regla condicional sobre una variable explicativa | Nodo hoja: nivel de riesgo operativo",
        ha='center',
        fontsize=7.3,
        color=GRIS_MEDIO,
        style='italic'
    )

    plt.title(
        'Estructura ilustrativa de un árbol de decisión aplicado al problema de\n'
        'clasificación de inoperatividad en ambulancias Tipo II',
        fontsize=11,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(
        "figuras_tesis/figura_05_arbol_decision_ilustrativo.png",
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close()

    print("  Figura guardada: figuras_tesis/figura_05_arbol_decision_ilustrativo.png")


def generar_figura_06_bagging_random_forest() -> None:
    """
    Genera la Figura 6:
    Esquema de bagging en Random Forest: muestreo con reemplazo,
    entrenamiento de K árboles independientes y agregación por votación.
    """
    _configurar_estilo_figuras_tesis()
    _asegurar_directorio_figuras()

    from matplotlib.patches import FancyBboxPatch

    AZUL = '#355C7D'
    GRIS_OSCURO = '#4B5358'
    GRIS_MEDIO = '#8E969B'
    NEGRO = '#1F2326'

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    def caja(x, y, w, h, texto, color, fs=8.2):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.05',
            facecolor=color,
            edgecolor='white',
            linewidth=1.3
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            texto,
            ha='center',
            va='center',
            fontsize=fs,
            fontweight='bold',
            color='white'
        )

    def flecha(x1, y1, x2, y2):
        ax.annotate(
            '',
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->',
                lw=1.1,
                color=GRIS_MEDIO
            )
        )

    caja(0.45, 3.55, 1.35, 0.85, "Dataset\noriginal\nN obs.", AZUL)

    caja(2.25, 5.25, 1.8, 0.70, "Bootstrap\nmuestra 1", GRIS_OSCURO)
    caja(5.10, 5.25, 1.8, 0.70, "Bootstrap\nmuestra 2", GRIS_OSCURO)
    caja(7.95, 5.25, 1.8, 0.70, "Bootstrap\nmuestra 3", GRIS_OSCURO)

    caja(2.25, 3.55, 1.8, 0.75, "Árbol 1\nprofundidad ≤ 8", GRIS_OSCURO)
    caja(5.10, 3.55, 1.8, 0.75, "Árbol 2\nprofundidad ≤ 8", GRIS_OSCURO)
    caja(7.95, 3.55, 1.8, 0.75, "Árbol 3\nprofundidad ≤ 8", GRIS_OSCURO)
    caja(10.25, 3.55, 1.35, 0.75, "Árbol K\nK = 200", NEGRO)

    caja(3.1, 1.75, 6.7, 0.80,
         "Agregación por votación mayoritaria\nŷ = mode(h₁(x), h₂(x), ..., hₖ(x))",
         GRIS_OSCURO)

    caja(4.35, 0.55, 4.2, 0.75,
         "Predicción final: ŷ ∈ {0, 1}",
         NEGRO)

    # Flechas dataset -> bootstrap
    flecha(1.8, 4.0, 2.25, 5.60)
    flecha(1.8, 4.0, 5.10, 5.60)
    flecha(1.8, 4.0, 7.95, 5.60)

    # Flechas bootstrap -> árbol
    flecha(3.15, 5.25, 3.15, 4.30)
    flecha(6.00, 5.25, 6.00, 4.30)
    flecha(8.85, 5.25, 8.85, 4.30)

    # Dataset -> árboles
    flecha(1.8, 4.0, 2.25, 3.95)

    # Árboles -> agregación
    flecha(3.15, 3.55, 5.0, 2.55)
    flecha(6.00, 3.55, 6.00, 2.55)
    flecha(8.85, 3.55, 7.0, 2.55)
    flecha(10.9, 3.55, 8.1, 2.55)

    # Agregación -> predicción
    flecha(6.0, 1.75, 6.0, 1.30)

    ax.text(10.0, 4.1, "…", fontsize=16, color=GRIS_MEDIO, ha='center')

    plt.title(
        'Esquema de bagging en Random Forest: muestreo con reemplazo,\n'
        'entrenamiento de K árboles independientes y agregación por votación',
        fontsize=11,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(
        "figuras_tesis/figura_06_bagging_random_forest.png",
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close()

    print("  Figura guardada: figuras_tesis/figura_06_bagging_random_forest.png")


def generar_figura_07_gradient_boosting_conceptual() -> None:
    """
    Genera la Figura 7:
    Proceso secuencial de Gradient Boosting como modelo comparativo
    de ensamble.

    La figura evita mostrar hiperparámetros concretos para no confundir
    el marco teórico con la configuración final del experimento.
    """
    _configurar_estilo_figuras_tesis()
    _asegurar_directorio_figuras()

    from matplotlib.patches import FancyBboxPatch

    AZUL = '#355C7D'
    GRIS_OSCURO = '#4B5358'
    GRIS_MEDIO = '#8E969B'
    NEGRO = '#1F2326'
    ROJO = '#C96B63'

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    def caja(x, y, w, h, texto, color, fs=8.0):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.05',
            facecolor=color,
            edgecolor='white',
            linewidth=1.3
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            texto,
            ha='center',
            va='center',
            fontsize=fs,
            fontweight='bold',
            color='white'
        )

    def flecha(x1, y1, x2, y2):
        ax.annotate(
            '',
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->',
                lw=1.1,
                color=GRIS_MEDIO
            )
        )

    def flecha_error(x, y1, y2):
        ax.annotate(
            '',
            xy=(x, y2),
            xytext=(x, y1),
            arrowprops=dict(
                arrowstyle='->',
                lw=1.0,
                color=ROJO,
                linestyle='--'
            )
        )

    caja(0.55, 2.35, 1.65, 0.85, "Modelo inicial\nF₀(x)", AZUL)
    caja(2.75, 2.35, 1.65, 0.85, "Árbol débil h₁(x)\nresiduales", GRIS_OSCURO)
    caja(4.95, 2.35, 1.65, 0.85, "Árbol débil h₂(x)\nerror restante", GRIS_OSCURO)
    caja(7.15, 2.35, 1.65, 0.85, "Árbol débil hₘ(x)\ncorrección iterativa", GRIS_OSCURO)
    caja(9.70, 2.35, 1.65, 0.85, "Predicción\nfinal Fₘ(x)", NEGRO)

    flecha(2.20, 2.78, 2.75, 2.78)
    flecha(4.40, 2.78, 4.95, 2.78)
    flecha(6.60, 2.78, 7.15, 2.78)
    flecha(8.80, 2.78, 9.70, 2.78)

    flecha_error(3.55, 1.05, 2.35)
    flecha_error(5.75, 1.05, 2.35)
    flecha_error(7.95, 1.05, 2.35)

    ax.text(3.55, 0.78, "errores del\nmodelo previo",
            ha='center', fontsize=7.2, color=ROJO, style='italic')
    ax.text(5.75, 0.78, "nuevos\nresiduales",
            ha='center', fontsize=7.2, color=ROJO, style='italic')
    ax.text(7.95, 0.78, "corrección\nsecuencial",
            ha='center', fontsize=7.2, color=ROJO, style='italic')

    ax.text(
        6.0,
        1.55,
        "Cada árbol débil se entrena para reducir el error residual del ensamble anterior.",
        ha='center',
        fontsize=8,
        color=GRIS_OSCURO,
        style='italic'
    )

    plt.title(
        'Proceso secuencial de Gradient Boosting como modelo comparativo:\n'
        'corrección iterativa de errores mediante árboles débiles',
        fontsize=11,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(
        "figuras_tesis/figura_07_gradient_boosting_conceptual.png",
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close()

    print("  Figura guardada: figuras_tesis/figura_07_gradient_boosting_conceptual.png")


def generar_figura_08_matriz_confusion_teorica() -> None:
    """
    Genera la Figura 8:
    Estructura de la matriz de confusión y métricas derivadas para
    clasificación binaria.
    """
    _configurar_estilo_figuras_tesis()
    _asegurar_directorio_figuras()

    from matplotlib.patches import FancyBboxPatch

    AZUL = '#355C7D'
    ROJO = '#C96B63'
    VERDE = '#78BFA3'
    AMARILLO = '#F4D35E'
    GRIS_OSCURO = '#4B5358'
    GRIS_MEDIO = '#8E969B'
    NEGRO = '#1F2326'

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.3))
    ax1, ax2 = axes

    # -------------------------------------------------------------------------
    # Panel izquierdo: matriz de confusión
    # -------------------------------------------------------------------------
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)
    ax1.axis('off')

    def caja_matriz(x, y, w, h, texto, color, fs=8.2, text_color='white'):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.04',
            facecolor=color,
            edgecolor='white',
            linewidth=1.2
        )
        ax1.add_patch(rect)
        ax1.text(
            x + w / 2,
            y + h / 2,
            texto,
            ha='center',
            va='center',
            fontsize=fs,
            color=text_color,
            fontweight='bold'
        )

    ax1.text(2.65, 4.55, "Predicho Y = 1", ha='center', fontsize=8.5, fontweight='bold')
    ax1.text(4.05, 4.55, "Predicho Y = 0", ha='center', fontsize=8.5, fontweight='bold')
    ax1.text(0.58, 3.55, "Real\nY = 1", ha='center', va='center', fontsize=8.5, fontweight='bold')
    ax1.text(0.58, 2.05, "Real\nY = 0", ha='center', va='center', fontsize=8.5, fontweight='bold')

    caja_matriz(
        1.65, 3.00, 1.6, 1.1,
        "VP\nVerdadero positivo\ninoperatividad\npredicha y ocurrida",
        VERDE,
        fs=7.1
    )
    caja_matriz(
        3.25, 3.00, 1.6, 1.1,
        "FN\nFalso negativo\ninoperatividad\nno predicha",
        ROJO,
        fs=7.1
    )
    caja_matriz(
        1.65, 1.50, 1.6, 1.1,
        "FP\nFalso positivo\nalerta sin\ninoperatividad",
        AMARILLO,
        fs=7.1
    )
    caja_matriz(
        3.25, 1.50, 1.6, 1.1,
        "VN\nVerdadero negativo\noperatividad\ncorrectamente predicha",
        AZUL,
        fs=7.1
    )

    ax1.set_title(
        'Estructura de la matriz de confusión',
        fontsize=10,
        fontweight='bold'
    )

    # -------------------------------------------------------------------------
    # Panel derecho: métricas derivadas
    # -------------------------------------------------------------------------
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis('off')

    ax2.set_title(
        'Métricas derivadas de la matriz de confusión',
        fontsize=10,
        fontweight='bold'
    )

    metricas = [
        ("Precisión", r"$\frac{VP}{VP + FP}$", "Proporción de alertas positivas correctas", AZUL),
        ("Sensibilidad\n(recall)", r"$\frac{VP}{VP + FN}$", "Proporción de inoperatividades detectadas", ROJO),
        ("Especificidad", r"$\frac{VN}{VN + FP}$", "Proporción de operatividades correctamente identificadas", VERDE),
        ("F1-Score", r"$\frac{2 \cdot VP}{2 \cdot VP + FP + FN}$", "Media armónica entre precisión y sensibilidad", GRIS_OSCURO),
    ]

    y0 = 5.0
    for i, (nombre, formula, descripcion, color) in enumerate(metricas):
        y = y0 - i * 1.15

        ax2.text(
            0.1, y,
            nombre,
            fontsize=8.4,
            fontweight='bold',
            color=color,
            va='center'
        )

        ax2.text(
            3.0, y,
            formula,
            fontsize=13,
            color=NEGRO,
            va='center'
        )

        ax2.text(
            6.25, y,
            descripcion,
            fontsize=7.6,
            color=GRIS_MEDIO,
            va='center',
            style='italic'
        )

        ax2.plot([0.1, 9.6], [y - 0.42, y - 0.42],
                 color='#E6E6E6',
                 lw=0.8)

    plt.suptitle(
        'Matriz de confusión y métricas de evaluación para\n'
        'clasificación binaria desbalanceada',
        fontsize=11,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(
        "figuras_tesis/figura_08_matriz_confusion_teorica.png",
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close()

    print("  Figura guardada: figuras_tesis/figura_08_matriz_confusion_teorica.png")

def generar_figura_10_umbral_precision_sensibilidad(resultados_lista: list,
                                                     y_val: pd.Series) -> None:
    """
    Genera la Figura 10:
    Relación precisión-sensibilidad según el umbral de clasificación
    y comparativa de sensibilidad entre los modelos evaluados.

    La curva del panel izquierdo se calcula con las probabilidades del
    modelo Random Forest, seleccionado como modelo predictivo principal.
    """
    _configurar_estilo_figuras_tesis()
    _asegurar_directorio_figuras()

    AZUL = '#355C7D'
    AZUL_MEDIO = '#6C8EAD'
    GRIS_OSCURO = '#4B5358'
    GRIS_MEDIO = '#8E969B'
    GRIS_CLARO = '#C9CED3'
    ROJO = '#C96B63'
    AMARILLO = '#F4D35E'

    rf_res = next((r for r in resultados_lista if r['nombre'] == 'Random Forest'), None)
    if rf_res is None:
        print("  No se encontró Random Forest para generar la Figura 10.")
        return

    y_prob = np.asarray(rf_res['y_prob'])
    y_true = np.asarray(y_val)

    thresholds = np.linspace(0.01, 0.99, 99)
    precisiones = []
    sensibilidades = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        precisiones.append(precision_score(y_true, y_pred_t, zero_division=0))
        sensibilidades.append(recall_score(y_true, y_pred_t, zero_division=0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # -------------------------------------------------------------------------
    # Panel 1: precisión y sensibilidad según umbral
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(thresholds, sensibilidades, color=AZUL, lw=2.0,
             label='Sensibilidad (recall)')
    ax1.plot(thresholds, precisiones, color=GRIS_OSCURO, lw=1.8,
             linestyle='--', label='Precisión')

    ax1.axvline(x=0.30, color=ROJO, linestyle=':', lw=1.6,
                label='Umbral seleccionado (τ = 0,30)')
    ax1.axvline(x=0.50, color=GRIS_CLARO, linestyle='--', lw=1.2,
                label='Umbral estándar (τ = 0,50)')

    idx_03 = np.argmin(np.abs(thresholds - 0.30))

    ax1.scatter([0.30], [sensibilidades[idx_03]], color=AZUL,
                s=28, zorder=5)
    ax1.scatter([0.30], [precisiones[idx_03]], color=GRIS_OSCURO,
                s=28, zorder=5)

    ax1.annotate(f"Sens. = {sensibilidades[idx_03]:.2f}",
                 xy=(0.30, sensibilidades[idx_03]),
                 xytext=(0.38, min(sensibilidades[idx_03] + 0.08, 1.02)),
                 arrowprops=dict(arrowstyle='-', color=GRIS_MEDIO, lw=1),
                 fontsize=8, color=GRIS_OSCURO)

    ax1.annotate(f"Prec. = {precisiones[idx_03]:.2f}",
                 xy=(0.30, precisiones[idx_03]),
                 xytext=(0.38, max(precisiones[idx_03] - 0.10, 0.04)),
                 arrowprops=dict(arrowstyle='-', color=GRIS_MEDIO, lw=1),
                 fontsize=8, color=GRIS_OSCURO)

    ax1.set_title('Relación entre precisión y sensibilidad\nsegún el umbral de clasificación',
                  fontsize=10)
    ax1.set_xlabel('Umbral de clasificación (τ)', fontsize=9)
    ax1.set_ylabel('Valor de la métrica', fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.35)
    ax1.legend(fontsize=7.5, frameon=True, loc='best')

    # -------------------------------------------------------------------------
    # Panel 2: sensibilidad comparativa
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    nombres = [_nombre_corto_modelo(r['nombre']) for r in resultados_lista]
    sensibilidades_modelos = [r['sensibilidad'] for r in resultados_lista]

    colores = [GRIS_CLARO, GRIS_MEDIO, AZUL, GRIS_OSCURO]
    hatches = ['', '///', '\\\\', '']

    x = np.arange(len(nombres))
    barras = ax2.bar(x, sensibilidades_modelos,
                     color=colores[:len(nombres)],
                     edgecolor='white',
                     linewidth=0.8)

    for barra, hatch in zip(barras, hatches):
        barra.set_hatch(hatch)

    for barra, valor in zip(barras, sensibilidades_modelos):
        ax2.text(barra.get_x() + barra.get_width() / 2,
                 valor + 0.025,
                 f"{valor:.4f}",
                 ha='center',
                 va='bottom',
                 fontsize=8,
                 fontweight='bold',
                 color=GRIS_OSCURO)

    ax2.axhline(y=resultados_lista[0]['sensibilidad'],
                color=GRIS_CLARO,
                linestyle='--',
                lw=1)

    ax2.set_xticks(x)
    ax2.set_xticklabels(nombres, fontsize=8)
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel('Sensibilidad', fontsize=9)
    ax2.set_title('Sensibilidad comparativa entre modelos\nValidación 2025 | Umbral τ = 0,30',
                  fontsize=10)
    ax2.grid(True, axis='y', alpha=0.35)

    plt.suptitle(
        'Relación precisión-sensibilidad según el umbral de clasificación\n'
        'y comparativa de sensibilidad entre los modelos evaluados',
        fontsize=11,
        fontweight='bold',
        y=1.03
    )

    plt.tight_layout()
    plt.savefig("figuras_tesis/figura_10_umbral_precision_sensibilidad.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    print("  Figura guardada: figuras_tesis/figura_10_umbral_precision_sensibilidad.png")


def generar_figura_18_construccion_dataset(df_train: pd.DataFrame,
                                            df_val: pd.DataFrame) -> None:
    """
    Genera la Figura 18:
    Diagrama de flujo del proceso de construcción del dataset mediante
    ventanas temporales deslizantes.
    """
    _configurar_estilo_figuras_tesis()
    _asegurar_directorio_figuras()

    from matplotlib.patches import FancyBboxPatch

    AZUL = '#355C7D'
    GRIS_OSCURO = '#4B5358'
    GRIS_MEDIO = '#8E969B'
    ROJO = '#C96B63'
    VERDE = '#78BFA3'
    NEGRO = '#1F2326'

    total_obs = len(df_train) + len(df_val)
    n_unidades = pd.concat([
        df_train[['id_ambulancia']],
        df_val[['id_ambulancia']]
    ], ignore_index=True)['id_ambulancia'].nunique()

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def caja(x, y, w, h, texto, color, fs=8.6):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.04',
            facecolor=color,
            edgecolor='white',
            linewidth=1.4
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, texto,
                ha='center', va='center',
                fontsize=fs, fontweight='bold',
                color='white')

    def flecha(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->',
                                    lw=1.2,
                                    color=GRIS_MEDIO))

    caja(0.7, 8.4, 2.0, 0.75, "Tabla\nMantenimiento", GRIS_OSCURO)
    caja(4.0, 8.4, 2.0, 0.75, "Tabla\nDowntime", GRIS_OSCURO)
    caja(7.3, 8.4, 2.0, 0.75, "Tabla\nUso operativo", GRIS_OSCURO)

    caja(2.1, 6.95, 5.8, 0.82,
         "Integración por ambulancia y período\n(cortes semanales)", AZUL)
    caja(1.8, 5.50, 6.4, 0.82,
         "Para cada corte temporal t₀:\ncalcular variables explicativas en W = 60 días previos",
         GRIS_OSCURO)
    caja(1.5, 4.05, 7.0, 0.88,
         "Determinar variable objetivo en T = 14 días siguientes\n"
         "Y = 1 si hubo inoperatividad | Y = 0 en caso contrario",
         ROJO)
    caja(2.2, 2.60, 5.6, 0.82,
         "Observación modelada:\n14 variables explicativas + variable objetivo Y ∈ {0,1}",
         VERDE)
    caja(1.8, 1.12, 6.4, 0.90,
         f"Dataset final: {total_obs:,} observaciones × 15 columnas\n"
         f"{n_unidades} ambulancias simuladas | np.random.seed(42)",
         NEGRO)

    flecha(1.7, 8.4, 3.6, 7.77)
    flecha(5.0, 8.4, 5.0, 7.77)
    flecha(8.3, 8.4, 6.4, 7.77)
    flecha(5.0, 6.95, 5.0, 6.32)
    flecha(5.0, 5.50, 5.0, 4.93)
    flecha(5.0, 4.05, 5.0, 3.42)
    flecha(5.0, 2.60, 5.0, 2.02)

    ax.text(5.0, 0.35,
            f"Partición temporal: entrenamiento 2024 ({len(df_train):,} obs.) | "
            f"validación 2025 ({len(df_val):,} obs.)",
            ha='center',
            va='center',
            fontsize=8,
            color=GRIS_OSCURO,
            style='italic')

    plt.title('Diagrama de flujo del proceso de construcción del dataset\n'
              'mediante ventanas temporales deslizantes',
              fontsize=11,
              fontweight='bold')

    plt.tight_layout()
    plt.savefig("figuras_tesis/figura_18_construccion_dataset.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    print("  Figura guardada: figuras_tesis/figura_18_construccion_dataset.png")


def generar_figura_19_pipeline_entrenamiento(resultados_lista: list,
                                             df_train: pd.DataFrame) -> None:
    """
    Genera la Figura 19:
    Diagrama de flujo del pipeline de entrenamiento de modelos de
    clasificación supervisada.
    """
    _configurar_estilo_figuras_tesis()
    _asegurar_directorio_figuras()

    from matplotlib.patches import FancyBboxPatch

    AZUL = '#355C7D'
    GRIS_OSCURO = '#4B5358'
    GRIS_MEDIO = '#8E969B'
    NEGRO = '#1F2326'
    VERDE = '#78BFA3'

    rf_res = next((r for r in resultados_lista if r['nombre'] == 'Random Forest'), None)
    auc_rf = rf_res['auc_roc'] if rf_res else 0.0
    sens_rf = rf_res['sensibilidad'] if rf_res else 0.0

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 4.2)
    ax.axis('off')

    def caja(x, y, w, h, texto, color, fs=8.2):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.04',
            facecolor=color,
            edgecolor='white',
            linewidth=1.4
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, texto,
                ha='center', va='center',
                fontsize=fs,
                color='white',
                fontweight='bold')

    def flecha(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->',
                                    lw=1.2,
                                    color=GRIS_MEDIO))

    y = 1.75

    caja(0.25, y, 1.75, 1.0,
         f"Dataset\ntrain 2024\n{len(df_train):,} obs.",
         AZUL)

    caja(2.35, y, 2.05, 1.0,
         "Balanceo\noversampling\nclase minoritaria",
         GRIS_OSCURO)

    caja(4.75, y, 1.85, 1.0,
         "Escalado\nsolo Regresión\nLogística",
         GRIS_OSCURO)

    caja(6.95, y, 2.30, 1.0,
         "Entrenamiento\n3 algoritmos\nRL | RF | GB",
         GRIS_OSCURO)

    caja(9.60, y, 2.10, 1.0,
         "Validación\nretrospectiva 2025\nτ = 0,30",
         GRIS_OSCURO)

    caja(12.05, y, 1.85, 1.0,
         "Modelo seleccionado\nRandom Forest",
         VERDE)

    flecha(2.00, y + 0.50, 2.35, y + 0.50)
    flecha(4.40, y + 0.50, 4.75, y + 0.50)
    flecha(6.60, y + 0.50, 6.95, y + 0.50)
    flecha(9.25, y + 0.50, 9.60, y + 0.50)
    flecha(11.70, y + 0.50, 12.05, y + 0.50)

    ax.text(1.10, 1.35, "np.random.seed(42)",
            ha='center', fontsize=7.2, color=GRIS_OSCURO, style='italic')
    ax.text(3.38, 1.22, "Balanceo reproducible\nsin modificar validación",
            ha='center', fontsize=7.2, color=GRIS_OSCURO, style='italic')
    ax.text(5.68, 1.22, "StandardScaler\nsolo para RL",
            ha='center', fontsize=7.2, color=GRIS_OSCURO, style='italic')
    ax.text(8.10, 1.05,
            "RL: C = 0,1\nRF: n = 200, depth = 8\nGB: n = 150, lr = 0,05",
            ha='center', fontsize=7.1, color=GRIS_OSCURO, style='italic')
    ax.text(10.65, 1.22,
            "Métricas: exactitud, especificidad,\nprecisión, sensibilidad, F1, AUC y MAE",
            ha='center', fontsize=7.1, color=GRIS_OSCURO, style='italic')
    ax.text(12.98, 1.22,
            f"AUC-ROC = {auc_rf:.4f}\nSens. = {sens_rf:.4f}",
            ha='center', fontsize=7.2, color=GRIS_OSCURO, style='italic')

    plt.title('Diagrama de flujo del pipeline de entrenamiento de los modelos\n'
              'de clasificación supervisada',
              fontsize=11,
              fontweight='bold')

    plt.tight_layout()
    plt.savefig("figuras_tesis/figura_19_pipeline_entrenamiento.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    print("  Figura guardada: figuras_tesis/figura_19_pipeline_entrenamiento.png")


def generar_figura_23_distribucion_variable_objetivo(df_train: pd.DataFrame,
                                                      df_val: pd.DataFrame) -> None:
    """
    Genera la Figura 23:
    Distribución de la variable objetivo por conjunto de datos.
    """
    _configurar_estilo_figuras_tesis()
    _asegurar_directorio_figuras()

    AZUL = '#355C7D'
    ROJO = '#C96B63'
    GRIS_OSCURO = '#4B5358'

    df_total = pd.concat([df_train, df_val], ignore_index=True)

    conjuntos = [
        ("Dataset completo", df_total),
        ("Entrenamiento 2024", df_train),
        ("Validación 2025", df_val)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, (titulo, df) in zip(axes, conjuntos):
        clase_0 = int((df[TARGET] == 0).sum())
        clase_1 = int((df[TARGET] == 1).sum())
        total = len(df)

        valores = [clase_0, clase_1]
        colores = [AZUL, ROJO]

        barras = ax.bar([0, 1], valores,
                        color=colores,
                        width=0.55,
                        edgecolor='white',
                        linewidth=0.8)

        for barra, valor, color in zip(barras, valores, colores):
            ax.text(barra.get_x() + barra.get_width() / 2,
                    valor + total * 0.015,
                    f"{valor:,}\n({valor / total * 100:.1f}%)",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold',
                    color=color)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Y = 0\n(operativa)', 'Y = 1\n(inoperativa)'],
                           fontsize=8)
        ax.set_title(f"{titulo}\n(n = {total:,})",
                     fontsize=10,
                     fontweight='bold')
        ax.set_ylabel('Observaciones', fontsize=9)
        ax.grid(True, axis='y', alpha=0.35)
        ax.tick_params(axis='y', labelsize=8)

    plt.suptitle('Distribución de la variable objetivo por conjunto de datos\n'
                 'Prevalencia de inoperatividad (Y = 1) en cada período',
                 fontsize=11,
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig("figuras_tesis/figura_23_distribucion_variable_objetivo.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    print("  Figura guardada: figuras_tesis/figura_23_distribucion_variable_objetivo.png")


def generar_figura_24_mapa_correlaciones(df_train: pd.DataFrame,
                                          df_val: pd.DataFrame) -> None:
    """
    Genera la Figura 24:
    Mapa de calor de correlaciones entre las 14 variables explicativas.
    """
    _configurar_estilo_figuras_tesis()
    _asegurar_directorio_figuras()

    df_total = pd.concat([
        df_train[FEATURES],
        df_val[FEATURES]
    ], ignore_index=True)

    corr = df_total.corr()

    etiquetas = [_mapear_variable_legible(v) for v in FEATURES]

    fig, ax = plt.subplots(figsize=(11, 8.7))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(FEATURES)))
    ax.set_yticks(np.arange(len(FEATURES)))
    ax.set_xticklabels(etiquetas, rotation=45, ha='right', fontsize=7.3)
    ax.set_yticklabels(etiquetas, fontsize=7.3)

    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            valor = corr.iloc[i, j]
            ax.text(j, i, f"{valor:.2f}",
                    ha='center',
                    va='center',
                    fontsize=6.2,
                    color='white' if abs(valor) > 0.60 else '#1F2326')

    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label('Coeficiente de correlación de Pearson', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title('Mapa de calor de correlaciones entre las 14 variables explicativas\n'
                 f'Dataset de observaciones | n = {len(df_total):,} | Período 2024-2025',
                 fontsize=10,
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig("figuras_tesis/figura_24_mapa_correlaciones.png",
                dpi=320, bbox_inches='tight')
    plt.close()

    print("  Figura guardada: figuras_tesis/figura_24_mapa_correlaciones.png")


def generar_figura_25_matrices_confusion(resultados_lista: list) -> None:
    """
    Genera la Figura 25:
    Matrices de confusión comparativas de los cuatro enfoques evaluados.
    """
    _configurar_estilo_figuras_tesis()
    _asegurar_directorio_figuras()

    modelos_orden = [
        'Preventivo tradicional (línea base)',
        'Regresión Logística',
        'Random Forest',
        'Gradient Boosting'
    ]

    AZUL = '#355C7D'
    GRIS_OSCURO = '#4B5358'

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))

    for ax, nombre in zip(axes, modelos_orden):
        r = next((x for x in resultados_lista if x['nombre'] == nombre), None)

        if r is None:
            ax.axis('off')
            continue

        tp = r['verdaderos_positivos']
        fp = r['falsos_positivos']
        tn = r['verdaderos_negativos']
        fn = r['falsos_negativos']

        # Orden operacional:
        # Filas: clase real positiva/negativa.
        # Columnas: predicción positiva/negativa.
        matriz = np.array([
            [tp, fn],
            [fp, tn]
        ])

        im = ax.imshow(matriz, cmap='Blues')

        total = matriz.sum()
        etiquetas = np.array([
            ['VP', 'FN'],
            ['FP', 'VN']
        ])

        for i in range(2):
            for j in range(2):
                valor = matriz[i, j]
                pct = valor / total * 100 if total > 0 else 0
                ax.text(j, i,
                        f"{etiquetas[i, j]}\n{valor:,}\n({pct:.1f}%)",
                        ha='center',
                        va='center',
                        fontsize=8,
                        fontweight='bold',
                        color='white' if valor > matriz.max() * 0.45 else GRIS_OSCURO)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred. Y=1', 'Pred. Y=0'], fontsize=7.5)
        ax.set_yticklabels(['Real Y=1', 'Real Y=0'], fontsize=7.5)

        ax.set_title(f"{_nombre_corto_modelo(nombre)}\n"
                     f"Sens.={r['sensibilidad']:.4f} | AUC={r['auc_roc']:.4f}",
                     fontsize=8.4,
                     fontweight='bold')

    plt.suptitle('Matrices de confusión de los cuatro enfoques evaluados\n'
                 'Período de validación 2025 | n = 1 750 | Umbral τ = 0,30',
                 fontsize=11,
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig("figuras_tesis/figura_25_matrices_confusion.png",
                dpi=320, bbox_inches='tight')
    plt.close()

    print("  Figura guardada: figuras_tesis/figura_25_matrices_confusion.png")


def generar_figuras_individuales_tesis(resultados_lista: list,
                                        df_train: pd.DataFrame,
                                        df_val: pd.DataFrame,
                                        y_val: pd.Series) -> None:
    """
    Ejecuta la generación de figuras individuales para la tesis.

    Estas figuras reemplazan versiones manuales o antiguas y se generan
    directamente desde los resultados y datasets finales del flujo computacional
    cuando corresponde.
    """
    print("\nGenerando figuras individuales para la tesis...")

    # Figuras teóricas y metodológicas del marco teórico/metodología
    generar_figura_03_sigmoide_regresion_logistica()
    generar_figura_04_balanceo_clases(df_train)
    generar_figura_05_arbol_decision_ilustrativo()
    generar_figura_06_bagging_random_forest()
    generar_figura_07_gradient_boosting_conceptual()
    generar_figura_08_matriz_confusion_teorica()
        
    # Figuras metodológicas y de resultados previamente automatizadas
    generar_figura_10_umbral_precision_sensibilidad(resultados_lista, y_val)
    generar_figura_18_construccion_dataset(df_train, df_val)
    generar_figura_19_pipeline_entrenamiento(resultados_lista, df_train)
    generar_figura_23_distribucion_variable_objetivo(df_train, df_val)
    generar_figura_24_mapa_correlaciones(df_train, df_val)
    generar_figura_25_matrices_confusion(resultados_lista)

    print("  Figuras individuales generadas en la carpeta: figuras_tesis/")

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

    generar_figuras_individuales_tesis(
        resultados_lista=resultados_lista,
        df_train=df_train,
        df_val=df_val,
        y_val=y_val
    )

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
    print(f"  figuras_tesis/figura_03_sigmoide_regresion_logistica.png")
    print(f"  figuras_tesis/figura_04_balanceo_clases.png")
    print(f"  figuras_tesis/figura_05_arbol_decision_ilustrativo.png")
    print(f"  figuras_tesis/figura_06_bagging_random_forest.png")
    print(f"  figuras_tesis/figura_07_gradient_boosting_conceptual.png")
    print(f"  figuras_tesis/figura_08_matriz_confusion_teorica.png")
    print(f"  figuras_tesis/figura_10_umbral_precision_sensibilidad.png")
    print(f"  figuras_tesis/figura_18_construccion_dataset.png")
    print(f"  figuras_tesis/figura_19_pipeline_entrenamiento.png")
    print(f"  figuras_tesis/figura_23_distribucion_variable_objetivo.png")
    print(f"  figuras_tesis/figura_24_mapa_correlaciones.png")
    print(f"  figuras_tesis/figura_25_matrices_confusion.png")

    print("\n¡Fase 3-4 completada exitosamente!")
    print("Siguiente paso: Fase 5 — Lineamientos técnicos y redacción de resultados")
