"""
=============================================================================
GENERADOR DE DATASET SIMULADO
Modelo computacional de mantenimiento predictivo para ambulancias Tipo II
Lima Metropolitana — Período 2024-2025

Referencia metodológica:
- NTS N.º 051-MINSA/OGDN-V.01 (MINSA, 2006)
- Prytz (2014) — estructura de variables y ventanas temporales
- Taoufyq et al. (2025) — enfoque data-driven
- Si et al. (2011) — variable objetivo RUL/clasificación binaria

Parámetros del modelo:
- W = 60 días (ventana histórica)
- T = 14 días (horizonte de predicción)
- Cortes temporales: periodicidad semanal
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# =============================================================================
# SEMILLA DE REPRODUCIBILIDAD
# =============================================================================
np.random.seed(42)

# =============================================================================
# PARÁMETROS DE SIMULACIÓN
# =============================================================================

# Flota
N_AMBULANCIAS = 35
FECHA_INICIO = datetime(2024, 1, 1)
FECHA_FIN = datetime(2025, 12, 31)

# Parámetros del modelo
W = 60   # Ventana histórica en días
T = 14   # Horizonte de predicción en días
PASO_CORTE = 7  # Periodicidad semanal de cortes t0

# Parámetros de mantenimiento (basados en NTS 051 y literatura)
INTERVALO_PM_MEDIA = 37    # Días promedio entre mantenimientos preventivos
INTERVALO_PM_STD = 8       # Desviación estándar del intervalo PM

# Parámetros de inoperatividad
# Defensoría del Pueblo (2023a): ~20-25% del parque inoperativo
PROB_INOPERATIVIDAD_BASE = 0.18   # Probabilidad base mensual de evento de inoperatividad
DURACION_DOWNTIME_MEDIA = 3.5     # Días promedio de inoperatividad por evento
DURACION_DOWNTIME_STD = 2.0       # Desviación estándar

# Subsistemas (distribución de eventos según NTS 051)
SUBSISTEMAS = ['vehicular', 'electrico', 'equipamiento']
PROB_SUBSISTEMA = [0.55, 0.28, 0.17]  # Distribución de probabilidad por subsistema

# Tasa de falla del equipamiento obligatorio
PROB_FALLA_EQUIPAMIENTO = 0.08  # Probabilidad mensual de indisponibilidad

# Uso operativo (Lima Metropolitana — estimación SAMU)
KM_MENSUAL_MEDIA = 2800
KM_MENSUAL_STD = 600
SERVICIOS_MENSUALES_MEDIA = 45
SERVICIOS_MENSUALES_STD = 12

# =============================================================================
# FUNCIÓN 1: GENERAR IDENTIFICADORES DE AMBULANCIAS
# =============================================================================

def generar_flota(n: int) -> list:
    """
    Genera identificadores únicos para la flota de ambulancias.
    Formato: AMB-TIPO2-XXX
    """
    return [f"AMB-T2-{str(i+1).zfill(3)}" for i in range(n)]


# =============================================================================
# FUNCIÓN 2: GENERAR TABLA DE EVENTOS DE MANTENIMIENTO
# =============================================================================

def generar_tabla_mantenimiento(flota: list,
                                 fecha_inicio: datetime,
                                 fecha_fin: datetime) -> pd.DataFrame:
    """
    Genera registros históricos de eventos de mantenimiento
    (preventivo y correctivo) para cada ambulancia de la flota.

    Columnas:
        id_ambulancia       : Identificador de la unidad
        fecha_inicio        : Fecha de inicio del evento
        fecha_fin           : Fecha de fin del evento
        tipo_mantenimiento  : 'preventivo' o 'correctivo'
        subsistema          : 'vehicular', 'electrico' o 'equipamiento'
        deriva_inoperatividad: True si el evento generó downtime
        duracion_dias       : Duración del evento en días
        observaciones       : Descripción textual del evento
    """
    registros = []
    total_dias = (fecha_fin - fecha_inicio).days

    descripciones_pm = {
        'vehicular': [
            'Cambio de aceite y filtros',
            'Revisión de frenos y suspensión',
            'Inspección del sistema de refrigeración',
            'Rotación y balanceo de neumáticos',
            'Revisión general del motor'
        ],
        'electrico': [
            'Revisión del sistema eléctrico de cabina',
            'Mantenimiento del sistema de iluminación',
            'Verificación de conexiones eléctricas',
            'Revisión del sistema de carga de equipos'
        ],
        'equipamiento': [
            'Verificación funcional del monitor desfibrilador',
            'Revisión del oxímetro de pulso portátil',
            'Inspección del equipamiento básico',
            'Calibración de equipos de monitorización'
        ]
    }

    descripciones_cm = {
        'vehicular': [
            'Reparación de fuga en sistema de frenos',
            'Cambio de batería vehicular',
            'Reparación del sistema de dirección',
            'Cambio de correa de distribución',
            'Reparación de fuga de aceite'
        ],
        'electrico': [
            'Reparación del sistema de iluminación de emergencia',
            'Reemplazo de fusibles del sistema eléctrico',
            'Reparación del inversor de corriente de cabina',
            'Corrección de falla en sistema de carga'
        ],
        'equipamiento': [
            'Reemplazo de monitor desfibrilador por falla técnica',
            'Reparación del oxímetro de pulso',
            'Reemplazo de componente de equipamiento básico',
            'Corrección de falla en equipo de monitorización'
        ]
    }

    for amb in flota:
        # Generar perfil de riesgo heterogéneo por ambulancia
        factor_riesgo = np.random.uniform(0.7, 1.5)
        intervalo_pm = max(20, np.random.normal(INTERVALO_PM_MEDIA, INTERVALO_PM_STD))

        fecha_cursor = fecha_inicio + timedelta(days=np.random.randint(0, 15))

        while fecha_cursor <= fecha_fin:
            # ---- MANTENIMIENTO PREVENTIVO ----
            subsistema_pm = np.random.choice(SUBSISTEMAS, p=PROB_SUBSISTEMA)
            duracion_pm = max(0.5, np.random.normal(1.2, 0.4))
            fecha_fin_pm = fecha_cursor + timedelta(days=duracion_pm)

            registros.append({
                'id_ambulancia': amb,
                'fecha_inicio': fecha_cursor.date(),
                'fecha_fin': fecha_fin_pm.date(),
                'tipo_mantenimiento': 'preventivo',
                'subsistema': subsistema_pm,
                'deriva_inoperatividad': False,
                'duracion_dias': round(duracion_pm, 1),
                'observaciones': np.random.choice(descripciones_pm[subsistema_pm])
            })

            # ---- MANTENIMIENTO CORRECTIVO (probabilístico) ----
            prob_cm = (PROB_INOPERATIVIDAD_BASE / 30) * intervalo_pm * factor_riesgo
            n_cm = np.random.poisson(prob_cm * 0.6)

            for _ in range(n_cm):
                offset_cm = np.random.randint(1, int(intervalo_pm))
                fecha_cm = fecha_cursor + timedelta(days=offset_cm)
                if fecha_cm > fecha_fin:
                    break

                subsistema_cm = np.random.choice(SUBSISTEMAS, p=PROB_SUBSISTEMA)
                duracion_cm = max(0.5, np.random.normal(2.8, 1.2))
                fecha_fin_cm = fecha_cm + timedelta(days=duracion_cm)
                deriva = np.random.random() < (0.65 * factor_riesgo)

                registros.append({
                    'id_ambulancia': amb,
                    'fecha_inicio': fecha_cm.date(),
                    'fecha_fin': fecha_fin_cm.date(),
                    'tipo_mantenimiento': 'correctivo',
                    'subsistema': subsistema_cm,
                    'deriva_inoperatividad': deriva,
                    'duracion_dias': round(duracion_cm, 1),
                    'observaciones': np.random.choice(descripciones_cm[subsistema_cm])
                })

            # Avanzar al siguiente PM
            siguiente_pm = intervalo_pm + np.random.normal(0, 5)
            fecha_cursor += timedelta(days=max(15, siguiente_pm))

    df = pd.DataFrame(registros)
    df = df.sort_values(['id_ambulancia', 'fecha_inicio']).reset_index(drop=True)
    return df


# =============================================================================
# FUNCIÓN 3: GENERAR TABLA DE INOPERATIVIDAD (DOWNTIME)
# =============================================================================

def generar_tabla_downtime(df_mantenimiento: pd.DataFrame,
                            flota: list,
                            fecha_inicio: datetime,
                            fecha_fin: datetime) -> pd.DataFrame:
    """
    Genera registros de inoperatividad derivados de eventos correctivos
    y eventos espontáneos no registrados en mantenimiento.

    Columnas:
        id_ambulancia           : Identificador de la unidad
        fecha_inicio_downtime   : Inicio del período fuera de servicio
        fecha_fin_downtime      : Fin del período fuera de servicio
        duracion_dias           : Duración en días
        causa                   : Tipo de causa del downtime
    """
    registros = []

    causas = ['falla_mecanica', 'falla_electrica', 'falla_equipamiento', 'administrativa']
    prob_causas = [0.45, 0.25, 0.17, 0.13]

    # Downtimes derivados de correctivos con deriva_inoperatividad=True
    cm_con_deriva = df_mantenimiento[
        (df_mantenimiento['tipo_mantenimiento'] == 'correctivo') &
        (df_mantenimiento['deriva_inoperatividad'] == True)
    ]

    for _, row in cm_con_deriva.iterrows():
        duracion = max(0.5, np.random.normal(DURACION_DOWNTIME_MEDIA, DURACION_DOWNTIME_STD))
        fecha_ini = pd.to_datetime(row['fecha_inicio'])
        fecha_fn = fecha_ini + timedelta(days=duracion)

        if fecha_fn.date() <= fecha_fin.date():
            causa_subsistema = {
                'vehicular': 'falla_mecanica',
                'electrico': 'falla_electrica',
                'equipamiento': 'falla_equipamiento'
            }.get(row['subsistema'], 'falla_mecanica')

            registros.append({
                'id_ambulancia': row['id_ambulancia'],
                'fecha_inicio_downtime': fecha_ini.date(),
                'fecha_fin_downtime': fecha_fn.date(),
                'duracion_dias': round(duracion, 1),
                'causa': causa_subsistema
            })

    # Downtimes espontáneos (no capturados en mantenimiento)
    for amb in flota:
        factor_riesgo = np.random.uniform(0.6, 1.4)
        n_espontaneos = np.random.poisson(3 * factor_riesgo)

        for _ in range(n_espontaneos):
            dias_offset = np.random.randint(0, (fecha_fin - fecha_inicio).days)
            fecha_ini = fecha_inicio + timedelta(days=dias_offset)
            duracion = max(1.0, np.random.exponential(4.0))
            fecha_fn = fecha_ini + timedelta(days=duracion)

            if fecha_fn <= fecha_fin:
                registros.append({
                    'id_ambulancia': amb,
                    'fecha_inicio_downtime': fecha_ini.date(),
                    'fecha_fin_downtime': fecha_fn.date(),
                    'duracion_dias': round(duracion, 1),
                    'causa': np.random.choice(causas, p=prob_causas)
                })

    df = pd.DataFrame(registros)
    df = df.sort_values(['id_ambulancia', 'fecha_inicio_downtime']).reset_index(drop=True)
    return df


# =============================================================================
# FUNCIÓN 4: GENERAR TABLA DE USO OPERATIVO
# =============================================================================

def generar_tabla_uso(flota: list,
                       fecha_inicio: datetime,
                       fecha_fin: datetime) -> pd.DataFrame:
    """
    Genera registros mensuales de uso operativo por ambulancia.

    Columnas:
        id_ambulancia       : Identificador de la unidad
        fecha               : Fecha del registro (fin de mes)
        kilometraje_periodo : Kilómetros recorridos en el período
        numero_servicios    : Servicios prestados en el período
        km_acumulado        : Kilometraje acumulado desde inicio
    """
    registros = []

    for amb in flota:
        # Perfil de uso heterogéneo por ambulancia
        factor_uso = np.random.uniform(0.7, 1.4)
        km_acumulado = np.random.randint(15000, 80000)  # Km iniciales al 01/01/2024

        fecha_cursor = fecha_inicio.replace(day=1)

        while fecha_cursor <= fecha_fin:
            km_periodo = max(500, np.random.normal(
                KM_MENSUAL_MEDIA * factor_uso,
                KM_MENSUAL_STD
            ))
            servicios = max(10, int(np.random.normal(
                SERVICIOS_MENSUALES_MEDIA * factor_uso,
                SERVICIOS_MENSUALES_STD
            )))

            km_acumulado += km_periodo

            registros.append({
                'id_ambulancia': amb,
                'fecha': fecha_cursor.date(),
                'kilometraje_periodo': round(km_periodo, 0),
                'numero_servicios': servicios,
                'km_acumulado': round(km_acumulado, 0)
            })

            # Avanzar al siguiente mes
            if fecha_cursor.month == 12:
                fecha_cursor = fecha_cursor.replace(year=fecha_cursor.year + 1, month=1)
            else:
                fecha_cursor = fecha_cursor.replace(month=fecha_cursor.month + 1)

    df = pd.DataFrame(registros)
    df = df.sort_values(['id_ambulancia', 'fecha']).reset_index(drop=True)
    return df


# =============================================================================
# FUNCIÓN 5: CALCULAR VARIABLES EXPLICATIVAS EN VENTANA W
# =============================================================================

def calcular_variables_ventana(id_amb: str,
                                t0: datetime,
                                df_mant: pd.DataFrame,
                                df_down: pd.DataFrame,
                                df_uso: pd.DataFrame,
                                W: int) -> dict:
    """
    Para una ambulancia y un corte temporal t0,
    calcula todas las variables explicativas en la ventana (t0-W, t0].
    """
    fecha_inicio_w = t0 - timedelta(days=W)

    # --- Filtrar datos de la ambulancia en la ventana ---
    mant_amb = df_mant[df_mant['id_ambulancia'] == id_amb].copy()
    mant_amb['fecha_inicio'] = pd.to_datetime(mant_amb['fecha_inicio'])

    down_amb = df_down[df_down['id_ambulancia'] == id_amb].copy()
    down_amb['fecha_inicio_downtime'] = pd.to_datetime(down_amb['fecha_inicio_downtime'])

    uso_amb = df_uso[df_uso['id_ambulancia'] == id_amb].copy()
    uso_amb['fecha'] = pd.to_datetime(uso_amb['fecha'])

    # Filtrar por ventana temporal
    mant_w = mant_amb[
        (mant_amb['fecha_inicio'] > fecha_inicio_w) &
        (mant_amb['fecha_inicio'] <= t0)
    ]
    down_w = down_amb[
        (down_amb['fecha_inicio_downtime'] > fecha_inicio_w) &
        (down_amb['fecha_inicio_downtime'] <= t0)
    ]
    uso_w = uso_amb[
        (uso_amb['fecha'] > fecha_inicio_w) &
        (uso_amb['fecha'] <= t0)
    ]

    # --- Variables de mantenimiento ---
    n_pm = len(mant_w[mant_w['tipo_mantenimiento'] == 'preventivo'])
    n_cm = len(mant_w[mant_w['tipo_mantenimiento'] == 'correctivo'])
    n_total_mant = n_pm + n_cm

    # Días desde última intervención
    if len(mant_w) > 0:
        ultima_interv = mant_w['fecha_inicio'].max()
        dias_desde_ultima = (t0 - ultima_interv).days
    else:
        dias_desde_ultima = W  # Sin intervención en ventana = W días

    # --- Variables por subsistema ---
    n_vehicular = len(mant_w[mant_w['subsistema'] == 'vehicular'])
    n_electrico = len(mant_w[mant_w['subsistema'] == 'electrico'])
    n_equipamiento = len(mant_w[mant_w['subsistema'] == 'equipamiento'])

    # --- Variables de inoperatividad ---
    n_episodios_down = len(down_w)
    downtime_total = down_w['duracion_dias'].sum() if len(down_w) > 0 else 0.0
    downtime_promedio = down_w['duracion_dias'].mean() if len(down_w) > 0 else 0.0

    # Disponibilidad en W
    disponibilidad_w = max(0, (W - downtime_total) / W)

    # --- Variables de uso operativo ---
    km_en_w = uso_w['kilometraje_periodo'].sum() if len(uso_w) > 0 else 0.0
    servicios_en_w = uso_w['numero_servicios'].sum() if len(uso_w) > 0 else 0

    # --- Condición funcional del equipamiento obligatorio ---
    n_fallas_eq = len(mant_w[
        (mant_w['subsistema'] == 'equipamiento') &
        (mant_w['tipo_mantenimiento'] == 'correctivo')
    ])
    equipamiento_funcional = 1 if n_fallas_eq == 0 else 0

    return {
        'id_ambulancia': id_amb,
        't0': t0.date(),
        # Variables de mantenimiento
        'n_pm_w': n_pm,
        'n_cm_w': n_cm,
        'n_total_mant_w': n_total_mant,
        'dias_desde_ultima_interv': dias_desde_ultima,
        # Variables por subsistema
        'n_eventos_vehicular_w': n_vehicular,
        'n_eventos_electrico_w': n_electrico,
        'n_eventos_equipamiento_w': n_equipamiento,
        # Variables de inoperatividad
        'n_episodios_downtime_w': n_episodios_down,
        'downtime_total_dias_w': round(downtime_total, 2),
        'downtime_promedio_dias_w': round(downtime_promedio, 2),
        'disponibilidad_w': round(disponibilidad_w, 4),
        # Variables de uso
        'km_en_w': round(km_en_w, 0),
        'servicios_en_w': int(servicios_en_w),
        # Condición del equipamiento
        'equipamiento_funcional': equipamiento_funcional
    }


# =============================================================================
# FUNCIÓN 6: CALCULAR VARIABLE OBJETIVO EN HORIZONTE T
# =============================================================================

def calcular_variable_objetivo(id_amb: str,
                                 t0: datetime,
                                 df_down: pd.DataFrame,
                                 T: int) -> int:
    """
    Calcula la variable objetivo binaria y_inoperativa_14:
        1 si hubo inoperatividad en (t0, t0+T]
        0 si no hubo inoperatividad en ese horizonte
    """
    fecha_fin_horizonte = t0 + timedelta(days=T)

    down_amb = df_down[df_down['id_ambulancia'] == id_amb].copy()
    down_amb['fecha_inicio_downtime'] = pd.to_datetime(down_amb['fecha_inicio_downtime'])

    eventos_horizonte = down_amb[
        (down_amb['fecha_inicio_downtime'] > t0) &
        (down_amb['fecha_inicio_downtime'] <= fecha_fin_horizonte)
    ]

    return 1 if len(eventos_horizonte) > 0 else 0


# =============================================================================
# FUNCIÓN 7: CONSTRUIR EL DATASET FINAL DE OBSERVACIONES
# =============================================================================

def construir_dataset(flota: list,
                       df_mant: pd.DataFrame,
                       df_down: pd.DataFrame,
                       df_uso: pd.DataFrame,
                       fecha_inicio: datetime,
                       fecha_fin: datetime,
                       W: int,
                       T: int,
                       paso: int) -> pd.DataFrame:
    """
    Construye el dataset final de observaciones.
    Cada observación es una ambulancia × corte temporal t0.

    El rango válido de cortes es:
        - Desde: fecha_inicio + W (necesitamos W días de historia)
        - Hasta: fecha_fin - T (necesitamos T días futuros para el objetivo)
    """
    print("Construyendo dataset de observaciones...")

    fecha_inicio_cortes = fecha_inicio + timedelta(days=W)
    fecha_fin_cortes = fecha_fin - timedelta(days=T)

    observaciones = []
    t0_cursor = fecha_inicio_cortes

    n_cortes = 0
    while t0_cursor <= fecha_fin_cortes:
        n_cortes += 1
        t0_cursor += timedelta(days=paso)

    print(f"  Ambulancias: {len(flota)}")
    print(f"  Cortes temporales válidos: {n_cortes}")
    print(f"  Observaciones totales esperadas: {len(flota) * n_cortes}")

    t0_cursor = fecha_inicio_cortes
    corte_num = 0

    while t0_cursor <= fecha_fin_cortes:
        corte_num += 1
        if corte_num % 10 == 0:
            print(f"  Procesando corte {corte_num}/{n_cortes}: {t0_cursor.date()}")

        for amb in flota:
            # Calcular variables explicativas en ventana W
            vars_exp = calcular_variables_ventana(
                amb, t0_cursor, df_mant, df_down, df_uso, W
            )
            # Calcular variable objetivo en horizonte T
            y = calcular_variable_objetivo(amb, t0_cursor, df_down, T)
            vars_exp['y_inoperativa_14'] = y

            observaciones.append(vars_exp)

        t0_cursor += timedelta(days=paso)

    df_dataset = pd.DataFrame(observaciones)
    df_dataset = df_dataset.sort_values(['id_ambulancia', 't0']).reset_index(drop=True)

    return df_dataset


# =============================================================================
# FUNCIÓN 8: GENERAR REPORTE DE ESTADÍSTICAS DEL DATASET
# =============================================================================

def generar_reporte(df_mant: pd.DataFrame,
                     df_down: pd.DataFrame,
                     df_uso: pd.DataFrame,
                     df_dataset: pd.DataFrame) -> None:
    """
    Imprime estadísticas descriptivas del dataset generado.
    """
    print("\n" + "="*60)
    print("REPORTE DEL DATASET SIMULADO")
    print("="*60)

    print(f"\n--- TABLAS FUENTE ---")
    print(f"Eventos de mantenimiento : {len(df_mant):,} registros")
    print(f"  - Preventivos          : {len(df_mant[df_mant['tipo_mantenimiento']=='preventivo']):,}")
    print(f"  - Correctivos          : {len(df_mant[df_mant['tipo_mantenimiento']=='correctivo']):,}")
    print(f"Registros de downtime    : {len(df_down):,} registros")
    print(f"Registros de uso         : {len(df_uso):,} registros")

    print(f"\n--- DATASET FINAL ---")
    print(f"Total de observaciones   : {len(df_dataset):,}")
    print(f"Ambulancias únicas       : {df_dataset['id_ambulancia'].nunique()}")
    print(f"Cortes temporales        : {df_dataset['t0'].nunique()}")
    print(f"Período cubierto         : {df_dataset['t0'].min()} — {df_dataset['t0'].max()}")

    print(f"\n--- VARIABLE OBJETIVO ---")
    dist_y = df_dataset['y_inoperativa_14'].value_counts()
    pct_pos = df_dataset['y_inoperativa_14'].mean() * 100
    print(f"Clase 0 (sin inoperatividad) : {dist_y.get(0,0):,} ({100-pct_pos:.1f}%)")
    print(f"Clase 1 (con inoperatividad) : {dist_y.get(1,0):,} ({pct_pos:.1f}%)")
    print(f"Desbalance de clases         : {(100-pct_pos)/pct_pos:.1f}:1")

    print(f"\n--- VARIABLES EXPLICATIVAS (estadísticas) ---")
    cols_stats = [
        'n_pm_w', 'n_cm_w', 'dias_desde_ultima_interv',
        'n_episodios_downtime_w', 'downtime_total_dias_w',
        'disponibilidad_w', 'km_en_w', 'servicios_en_w'
    ]
    print(df_dataset[cols_stats].describe().round(2).to_string())

    print(f"\n--- PARTICIÓN ENTRENAMIENTO / VALIDACIÓN ---")
    train = df_dataset[pd.to_datetime(df_dataset['t0']).dt.year == 2024]
    val = df_dataset[pd.to_datetime(df_dataset['t0']).dt.year == 2025]
    print(f"Entrenamiento (2024)  : {len(train):,} observaciones ({len(train)/len(df_dataset)*100:.1f}%)")
    print(f"Validación    (2025)  : {len(val):,} observaciones ({len(val)/len(df_dataset)*100:.1f}%)")
    print(f"Y=1 en entrenamiento  : {train['y_inoperativa_14'].mean()*100:.1f}%")
    print(f"Y=1 en validación     : {val['y_inoperativa_14'].mean()*100:.1f}%")


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    print("="*60)
    print("GENERANDO DATASET SIMULADO PARA TESIS")
    print("Ambulancias Tipo II — Lima Metropolitana 2024-2025")
    print("="*60)

    # 1. Generar flota
    print("\n[1/5] Generando identificadores de flota...")
    flota = generar_flota(N_AMBULANCIAS)
    print(f"      {len(flota)} ambulancias generadas: {flota[0]} ... {flota[-1]}")

    # 2. Generar tablas fuente
    print("\n[2/5] Generando tabla de mantenimiento...")
    df_mantenimiento = generar_tabla_mantenimiento(flota, FECHA_INICIO, FECHA_FIN)
    print(f"      {len(df_mantenimiento):,} eventos generados")

    print("\n[3/5] Generando tabla de downtime...")
    df_downtime = generar_tabla_downtime(df_mantenimiento, flota, FECHA_INICIO, FECHA_FIN)
    print(f"      {len(df_downtime):,} episodios generados")

    print("\n[4/5] Generando tabla de uso operativo...")
    df_uso = generar_tabla_uso(flota, FECHA_INICIO, FECHA_FIN)
    print(f"      {len(df_uso):,} registros generados")

    # 3. Construir dataset de observaciones
    print("\n[5/5] Construyendo dataset de observaciones por ventanas...")
    df_dataset = construir_dataset(
        flota, df_mantenimiento, df_downtime, df_uso,
        FECHA_INICIO, FECHA_FIN, W, T, PASO_CORTE
    )

    # 4. Generar reporte
    generar_reporte(df_mantenimiento, df_downtime, df_uso, df_dataset)

    # 5. Exportar a CSV
    print("\n--- EXPORTANDO ARCHIVOS ---")
    os.makedirs("datos_simulados", exist_ok=True)

    df_mantenimiento.to_csv("datos_simulados/tabla_mantenimiento.csv", index=False)
    df_downtime.to_csv("datos_simulados/tabla_downtime.csv", index=False)
    df_uso.to_csv("datos_simulados/tabla_uso.csv", index=False)
    df_dataset.to_csv("datos_simulados/dataset_observaciones.csv", index=False)

    # Exportar particiones
    df_train = df_dataset[pd.to_datetime(df_dataset['t0']).dt.year == 2024]
    df_val = df_dataset[pd.to_datetime(df_dataset['t0']).dt.year == 2025]
    df_train.to_csv("datos_simulados/dataset_entrenamiento_2024.csv", index=False)
    df_val.to_csv("datos_simulados/dataset_validacion_2025.csv", index=False)

    print(f"\nArchivos exportados en /datos_simulados/:")
    print(f"  tabla_mantenimiento.csv        ({len(df_mantenimiento):,} filas)")
    print(f"  tabla_downtime.csv             ({len(df_downtime):,} filas)")
    print(f"  tabla_uso.csv                  ({len(df_uso):,} filas)")
    print(f"  dataset_observaciones.csv      ({len(df_dataset):,} filas)")
    print(f"  dataset_entrenamiento_2024.csv ({len(df_train):,} filas)")
    print(f"  dataset_validacion_2025.csv    ({len(df_val):,} filas)")

    print("\n¡Dataset generado exitosamente!")
    print("Siguiente paso: Fase 3 — Modelado computacional (Random Forest / Logistic Regression)")
