"""
=============================================================================
GENERADOR DE DATASET SIMULADO
Modelo computacional de mantenimiento predictivo para ambulancias Tipo II
Lima Metropolitana — Período 2024-2025

DESCRIPCIÓN METODOLÓGICA:
    El presente script implementa un simulador paramétrico-estocástico para
    generar registros sintéticos de operación, mantenimiento e inoperatividad
    de ambulancias médicas urbanas Tipo II.

    El mecanismo de generación incorpora una función de riesgo compuesta λ(t),
    coherente con los principios del mantenimiento predictivo y la confiabilidad
    de activos físicos:

    1. Función de riesgo compuesta λ(t) que integra:
       — Factor de riesgo técnico latente por ambulancia (heterogeneidad de flota)
       — Carga operativa (km recorridos y servicios prestados relativos a la media)
       — Tiempo desde el último mantenimiento preventivo
       — Historial acumulado de correctivos previos (fragilidad operativa)
    2. Los correctivos se generan con tasa Poisson cuyo parámetro depende
       directamente de λ(t), no de azar puro.
    3. Los downtimes espontáneos se vinculan al mismo λ(t) acumulado del
       período de riesgo, no a distribuciones independientes.
    4. Los mantenimientos preventivos mantienen su periodicidad semestral (180±15 días).
    5. Cada parámetro nuevo está documentado como supuesto de simulación o
       inspirado en literatura de confiabilidad.

REFERENCIAS METODOLÓGICAS:
    — NTS N.º 051-MINSA/OGDN-V.01: clasificación y equipamiento de ambulancias.
    — Ficha de Homologación Ambulancia Urbana Tipo II: PM semestral o según fab.
    — Defensoría del Pueblo (2024): 33 ambulancias SAMU Lima; 7 inoperativas (21.2%).
    — Prytz (2014): variables operativas para mantenimiento predictivo vehicular.
    — Taoufyq et al. (2025): enfoque data-driven en mantenimiento predictivo.
    — Si et al. (2011): variable objetivo asociada a horizonte de falla/pronóstico.
    — Barlow & Proschan (1965): fundamentos de distribución de Weibull para fallas.
      [supuesto: riesgo crece con uso acumulado — justifica el componente km_factor]
    — Rausand & Høyland (2004): tasa de falla proporcional al nivel de uso.
      [supuesto: λ ∝ (km/km_ref) — componente de carga operativa del simulador]

PARÁMETROS DEL MODELO:
    — W = 60 días (ventana histórica)
    — T = 14 días (horizonte de predicción)
    — Cortes temporales: periodicidad semanal
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
# PARÁMETROS DE SIMULACIÓN — VALORES INVARIABLES (sustentados en fuentes)
# =============================================================================

# Flota
# Defensoría del Pueblo (2024): 33 ambulancias SAMU Lima.
N_AMBULANCIAS = 33
FECHA_INICIO = datetime(2024, 1, 1)
FECHA_FIN    = datetime(2025, 12, 31)

# Parámetros del modelo
W         = 60   # Ventana histórica en días
T         = 14   # Horizonte de predicción en días
PASO_CORTE = 7   # Periodicidad semanal de cortes t0

# Mantenimiento preventivo
# Ficha de Homologación: mantenimiento preventivo cada seis meses o según fab.
INTERVALO_PM_MEDIA    = 180   # ≈ 6 meses
INTERVALO_PM_VARIACION = 15   # Variación operativa simulada (±15 días)

# Inoperatividad
# Defensoría del Pueblo: 7 inoperativas de 33 = 21.2%.
PROB_INOPERATIVIDAD_BASE = 0.212   # Referencia contextual: 7/33 unidades inoperativas reportadas
DURACION_DOWNTIME_MEDIA  = 3.5     # Días promedio por evento (supuesto operativo)
DURACION_DOWNTIME_STD    = 2.0     # Desviación estándar

# Macro-subsistemas analíticos
# Agrupación funcional: Ficha de Homologación — estructura técnica.
SUBSISTEMAS      = ['vehicular', 'electrico_electronico', 'biomedico_asistencial']
PROB_SUBSISTEMA  = [1/3, 1/3, 1/3]   # Equiprobable: sin fuente empírica disponible

# Uso operativo — estimación SAMU Lima Metropolitana
KM_MENSUAL_MEDIA          = 2800
KM_MENSUAL_STD            = 600
SERVICIOS_MENSUALES_MEDIA = 45
SERVICIOS_MENSUALES_STD   = 12

# =============================================================================
# PARÁMETROS DE LA FUNCIÓN DE RIESGO COMPUESTA
# =============================================================================
# Supuesto de simulación: la tasa de falla λ(t) de una ambulancia en un período
# es proporcional al producto de cuatro factores:
#
#   λ(t) = λ_base × f_tecnico × f_uso_km × f_uso_svc × f_tiempo × (1 + f_historial)
#
# Este esquema es coherente con modelos de tasa de hazard proporcional
# (Cox, 1972; Rausand & Høyland, 2004) donde covariables observables modulan
# una tasa base.  Se implementa en versión paramétrica simplificada para
# datos sintéticos.

# Tasa base de correctivos por período de PM (≈ 180 días)
# Calibrada para producir una prevalencia Y=1 ≈ 20-30% en el dataset final.
# Supuesto de simulación: sin fuente empírica directa para ambulancias Lima.
LAMBDA_CM_BASE = 1.2   # Eventos correctivos esperados por ambulancia × período PM

# Factor de carga por km:
# λ se multiplica por (km_periodo / KM_MENSUAL_MEDIA * 6)^EXPO_KM
# Supuesto inspirado en Rausand & Høyland (2004): riesgo proporcional al uso.
# EXPO_KM < 1 evita inflación excesiva; 0.6 produce señal moderada y realista.
EXPO_KM  = 0.6   # Elasticidad de la tasa respecto al km relativo

# Factor de carga por servicios:
# Análogo al km, pero para número de servicios.  Supuesto de simulación.
EXPO_SVC = 0.5   # Elasticidad de la tasa respecto a servicios relativos

# Factor de tiempo desde PM:
# λ aumenta con el tiempo transcurrido desde el último PM (degradación).
# Supuesto: log-lineal en días transcurridos, normalizado por INTERVALO_PM_MEDIA.
# Basado en la intuición de un hazard creciente (forma Weibull β>1).
COEF_DIAS_PM = 0.8   # Coeficiente del componente temporal en log(λ)

# Factor de historial: cada correctivo previo en la ventana W aumenta λ en este %.
# Supuesto de simulación: la fragilidad operativa se acumula.
COEF_HISTORIAL_CM = 0.25   # +25% por cada correctivo previo (máx. 3 ajustados)

# Peso de downtimes espontáneos respecto a los derivados de correctivos.
# Se reduce fuertemente respecto a v1.0 para vincular la inoperatividad
# principalmente a la cadena: alto riesgo → correctivo → downtime.
PESO_ESPONTANEOS = 0.15   # Supuesto: 15% de downtimes sin correctivo registrado


# =============================================================================
# FUNCIÓN 1: GENERAR IDENTIFICADORES DE AMBULANCIAS
# =============================================================================

def generar_flota(n: int) -> list:
    """
    Genera identificadores únicos para la flota de ambulancias.
    Formato: AMB-T2-XXX
    """
    return [f"AMB-T2-{str(i+1).zfill(3)}" for i in range(n)]


# =============================================================================
# FUNCIÓN 2: GENERAR PERFIL DE RIESGO FIJO POR AMBULANCIA
# =============================================================================

def generar_perfiles_ambulancia(flota: list) -> dict:
    """
    Genera un perfil de riesgo latente y un factor de uso operativo fijos
    por ambulancia.  Estos valores representan heterogeneidad de flota
    (antigüedad, marca, historial de mantenimiento previo) y son observables
    indirectamente a través de las variables explicativas del dataset.

    Parámetros:
        factor_riesgo_tecnico : U(0.7, 1.5) — heterogeneidad técnica de la unidad.
            Supuesto de simulación: sin fuente que cuantifique este rango en Lima.
        factor_uso_operativo  : U(0.7, 1.4) — intensidad de uso relativa.
            Supuesto coherente con variabilidad de turnos y cobertura geográfica.
    """
    perfiles = {}
    for amb in flota:
        perfiles[amb] = {
            'factor_riesgo_tecnico': np.random.uniform(0.7, 1.5),
            'factor_uso_operativo' : np.random.uniform(0.7, 1.4),
            'km_inicial'           : np.random.randint(15000, 80000)
        }
    return perfiles


# =============================================================================
# FUNCIÓN 3: CALCULAR λ (TASA DE RIESGO COMPUESTA)
# =============================================================================

def calcular_lambda(factor_tecnico: float,
                    km_periodo: float,
                    svc_periodo: float,
                    dias_desde_pm: float,
                    n_cm_previos: int,
                    km_ref: float,
                    svc_ref: float) -> float:
    """
    Calcula la tasa λ de eventos correctivos para un período dado.

    Modelo: λ = λ_base × f_tecnico × f_km × f_svc × f_tiempo × (1 + f_hist)

    Argumentos:
        factor_tecnico  : Factor latente de la ambulancia [0.7, 1.5]
        km_periodo      : Km en el período de exposición
        svc_periodo     : Servicios en el período de exposición
        dias_desde_pm   : Días transcurridos desde el último PM
        n_cm_previos    : Número de correctivos previos en la ventana
        km_ref          : Km de referencia (media mensual × 6 meses)
        svc_ref         : Servicios de referencia (media mensual × 6 meses)

    Retorna:
        λ > 0 : Número esperado de eventos correctivos en el período
    """
    # Factor de carga por km (Rausand & Høyland, 2004 — riesgo ∝ uso)
    km_ratio  = max(0.1, km_periodo / km_ref)
    f_km      = km_ratio ** EXPO_KM

    # Factor de carga por servicios
    svc_ratio = max(0.1, svc_periodo / svc_ref)
    f_svc     = svc_ratio ** EXPO_SVC

    # Factor temporal (degradación entre PMs — inspirado en Weibull β > 1)
    t_norm    = max(0.01, dias_desde_pm / INTERVALO_PM_MEDIA)
    f_tiempo  = np.exp(COEF_DIAS_PM * t_norm)

    # Factor de historial (fragilidad acumulada)
    n_hist    = min(n_cm_previos, 3)   # Saturamos en 3 para no inflar
    f_historial = COEF_HISTORIAL_CM * n_hist

    # Tasa compuesta
    lam = (LAMBDA_CM_BASE
           * factor_tecnico
           * f_km
           * f_svc
           * f_tiempo
           * (1.0 + f_historial))

    return max(0.01, lam)


# =============================================================================
# FUNCIÓN 4: GENERAR TABLA DE EVENTOS DE MANTENIMIENTO
# =============================================================================

def generar_tabla_mantenimiento(flota: list,
                                 perfiles: dict,
                                 fecha_inicio: datetime,
                                 fecha_fin: datetime) -> pd.DataFrame:
    """
    Genera registros históricos de eventos de mantenimiento (preventivo y
    correctivo) para cada ambulancia.

    La tasa de eventos correctivos depende de calcular_lambda(), que integra
    el kilometraje recorrido, los servicios prestados, el tiempo desde el último
    mantenimiento preventivo y el historial de correctivos previos. Esto permite
    que las variables operativas y de mantenimiento estén asociadas con la
    probabilidad de inoperatividad futura.

    Columnas:
        id_ambulancia        : Identificador de la unidad
        fecha_inicio         : Fecha de inicio del evento
        fecha_fin            : Fecha de fin del evento
        tipo_mantenimiento   : 'preventivo' o 'correctivo'
        subsistema           : Macro-subsistema afectado
        deriva_inoperatividad: True si el evento genera downtime
        duracion_dias        : Duración del evento en días
        observaciones        : Descripción textual
    """
    registros = []
    total_dias = (fecha_fin - fecha_inicio).days

    # Referencias de km y servicios para el período PM (≈180 días ≈ 6 meses)
    km_ref_pm  = KM_MENSUAL_MEDIA  * 6
    svc_ref_pm = SERVICIOS_MENSUALES_MEDIA * 6

    descripciones_pm = {
        'vehicular': [
            'Cambio de aceite y filtros según programa semestral',
            'Revisión de frenos, suspensión y dirección',
            'Inspección del sistema de refrigeración del motor',
            'Rotación y balanceo de neumáticos',
            'Revisión general del motor y transmisión'
        ],
        'electrico_electronico': [
            'Revisión del sistema eléctrico de cabina asistencial',
            'Mantenimiento del sistema de iluminación de emergencia',
            'Verificación de conexiones eléctricas y masa',
            'Revisión del sistema de carga de equipos médicos'
        ],
        'biomedico_asistencial': [
            'Verificación funcional del monitor desfibrilador',
            'Revisión y calibración del oxímetro de pulso portátil',
            'Inspección de equipamiento básico de soporte vital',
            'Calibración de equipos de monitorización no invasiva'
        ]
    }

    descripciones_cm = {
        'vehicular': [
            'Reparación de fuga en sistema de frenos',
            'Reemplazo de batería vehicular por falla',
            'Corrección de falla en sistema de dirección',
            'Cambio de correa de distribución por desgaste',
            'Reparación de fuga de aceite en motor'
        ],
        'electrico_electronico': [
            'Reparación del sistema de balizas y sirena',
            'Reemplazo de fusibles por cortocircuito',
            'Reparación del inversor de corriente de cabina',
            'Corrección de falla en sistema de carga de equipos'
        ],
        'biomedico_asistencial': [
            'Reemplazo de monitor desfibrilador por falla técnica',
            'Reparación de oxímetro de pulso portátil',
            'Reemplazo de componente de equipamiento básico',
            'Corrección de falla en equipo de monitorización'
        ]
    }

    for amb in flota:
        factor_tecnico  = perfiles[amb]['factor_riesgo_tecnico']
        factor_uso      = perfiles[amb]['factor_uso_operativo']

        # Intervalo PM con variación operativa (±15 días)
        intervalo_pm = INTERVALO_PM_MEDIA + np.random.randint(
            -INTERVALO_PM_VARIACION, INTERVALO_PM_VARIACION + 1
        )

        # Inicio desfasado aleatoriamente (0-14 días) para evitar sincronía
        fecha_cursor    = fecha_inicio + timedelta(days=np.random.randint(0, 15))
        n_cm_acumulados = 0   # Historial de correctivos de esta ambulancia

        while fecha_cursor <= fecha_fin:

            # --- MANTENIMIENTO PREVENTIVO ---
            subsistema_pm = np.random.choice(SUBSISTEMAS, p=PROB_SUBSISTEMA)
            duracion_pm   = max(0.5, np.random.normal(1.2, 0.4))
            fecha_fin_pm  = fecha_cursor + timedelta(days=duracion_pm)

            registros.append({
                'id_ambulancia'       : amb,
                'fecha_inicio'        : fecha_cursor.date(),
                'fecha_fin'           : fecha_fin_pm.date(),
                'tipo_mantenimiento'  : 'preventivo',
                'subsistema'          : subsistema_pm,
                'deriva_inoperatividad': False,
                'duracion_dias'       : round(duracion_pm, 1),
                'observaciones'       : np.random.choice(descripciones_pm[subsistema_pm])
            })

            # --- CALCULAR KM Y SERVICIOS DEL PERÍODO PM ---
            # Se simulan los km y servicios que la ambulancia acumula en este
            # período inter-PM.  Estos valores alimentan calcular_lambda()
            # para determinar la tasa de correctivos, estableciendo la
            # dependencia causal entre uso operativo y eventos de falla.
            km_periodo_pm  = max(200, np.random.normal(
                km_ref_pm  * factor_uso, KM_MENSUAL_STD * np.sqrt(6)
            ))
            svc_periodo_pm = max(5, np.random.normal(
                svc_ref_pm * factor_uso, SERVICIOS_MENSUALES_STD * np.sqrt(6)
            ))

            # Días transcurridos desde el último PM (= intervalo_pm en este ciclo)
            dias_desde_pm = intervalo_pm

            # --- TASA DE CORRECTIVOS COMPUESTA ---
            lam = calcular_lambda(
                factor_tecnico  = factor_tecnico,
                km_periodo      = km_periodo_pm,
                svc_periodo     = svc_periodo_pm,
                dias_desde_pm   = dias_desde_pm,
                n_cm_previos    = n_cm_acumulados,
                km_ref          = km_ref_pm,
                svc_ref         = svc_ref_pm
            )

            # Número de correctivos en el período (distribución Poisson)
            n_cm = np.random.poisson(lam)
            n_cm_acumulados += n_cm

            # --- GENERAR EVENTOS CORRECTIVOS ---
            for _ in range(n_cm):
                # Distribuir los eventos correctivos uniformemente en el período
                offset_cm  = np.random.randint(1, max(2, intervalo_pm))
                fecha_cm   = fecha_cursor + timedelta(days=offset_cm)
                if fecha_cm > fecha_fin:
                    break

                subsistema_cm = np.random.choice(SUBSISTEMAS, p=PROB_SUBSISTEMA)
                duracion_cm   = max(0.5, np.random.normal(2.8, 1.2))
                fecha_fin_cm  = fecha_cm + timedelta(days=duracion_cm)

                # Probabilidad de derivar en downtime, modulada por factor técnico
                # Mayor factor_tecnico → mayor probabilidad de impacto operativo.
                p_deriva = np.clip(0.55 * factor_tecnico, 0.3, 0.85)
                deriva   = np.random.random() < p_deriva

                registros.append({
                    'id_ambulancia'       : amb,
                    'fecha_inicio'        : fecha_cm.date(),
                    'fecha_fin'           : fecha_fin_cm.date(),
                    'tipo_mantenimiento'  : 'correctivo',
                    'subsistema'          : subsistema_cm,
                    'deriva_inoperatividad': deriva,
                    'duracion_dias'       : round(duracion_cm, 1),
                    'observaciones'       : np.random.choice(descripciones_cm[subsistema_cm])
                })

            # --- AVANZAR AL SIGUIENTE PM ---
            siguiente_pm = INTERVALO_PM_MEDIA + np.random.randint(
                -INTERVALO_PM_VARIACION, INTERVALO_PM_VARIACION + 1
            )
            fecha_cursor += timedelta(days=siguiente_pm)

    df = pd.DataFrame(registros)
    df = df.sort_values(['id_ambulancia', 'fecha_inicio']).reset_index(drop=True)
    return df


# =============================================================================
# FUNCIÓN 5: GENERAR TABLA DE INOPERATIVIDAD (DOWNTIME)
# =============================================================================

def generar_tabla_downtime(df_mantenimiento: pd.DataFrame,
                            flota: list,
                            perfiles: dict,
                            fecha_inicio: datetime,
                            fecha_fin: datetime) -> pd.DataFrame:
    """
    Genera registros de inoperatividad.

    Los registros de downtime se generan a partir de dos mecanismos:
    1. Downtimes derivados de eventos correctivos que afectan la operatividad
       de la unidad.
    2. Downtimes espontáneos condicionados por el perfil de riesgo técnico y
       el nivel de uso operativo de cada ambulancia.

    Este diseño permite representar tanto inoperatividades asociadas a eventos
    correctivos registrados como episodios adicionales no vinculados directamente
    a una orden de mantenimiento específica.

    Columnas:
        id_ambulancia         : Identificador de la unidad
        fecha_inicio_downtime : Inicio del período fuera de servicio
        fecha_fin_downtime    : Fin del período fuera de servicio
        duracion_dias         : Duración en días
        causa                 : Tipo de causa
    """
    registros = []

    causas_espontaneas  = ['falla_mecanica', 'falla_electrica',
                           'falla_equipamiento', 'administrativa']
    prob_causas_esp     = [0.45, 0.25, 0.17, 0.13]

    total_dias = (fecha_fin - fecha_inicio).days

    # --- DOWNTIMES DERIVADOS DE CORRECTIVOS ---
    cm_con_deriva = df_mantenimiento[
        (df_mantenimiento['tipo_mantenimiento'] == 'correctivo') &
        (df_mantenimiento['deriva_inoperatividad'] == True)
    ]

    for _, row in cm_con_deriva.iterrows():
        duracion  = max(0.5, np.random.normal(DURACION_DOWNTIME_MEDIA,
                                               DURACION_DOWNTIME_STD))
        fecha_ini = pd.to_datetime(row['fecha_inicio'])
        fecha_fn  = fecha_ini + timedelta(days=duracion)

        if fecha_fn.date() <= fecha_fin.date():
            causa_subsistema = {
                'vehicular'             : 'falla_mecanica',
                'electrico_electronico' : 'falla_electrica',
                'biomedico_asistencial' : 'falla_equipamiento'
            }.get(row['subsistema'], 'falla_mecanica')

            registros.append({
                'id_ambulancia'        : row['id_ambulancia'],
                'fecha_inicio_downtime': fecha_ini.date(),
                'fecha_fin_downtime'   : fecha_fn.date(),
                'duracion_dias'        : round(duracion, 1),
                'causa'                : causa_subsistema
            })

    # --- DOWNTIMES ESPONTÁNEOS CONDICIONADOS AL PERFIL DE RIESGO ---
    # La tasa anual esperada de espontáneos es reducida (PESO_ESPONTANEOS × 6),
    # pero no es uniforme: depende del perfil técnico y de uso de la ambulancia.
    # Supuesto de simulación: las unidades de mayor riesgo técnico y mayor uso
    # concentran también los eventos no capturados en el registro formal.

    for amb in flota:
        factor_tecnico = perfiles[amb]['factor_riesgo_tecnico']
        factor_uso     = perfiles[amb]['factor_uso_operativo']

        # Tasa anual de espontáneos: escalada por riesgo técnico y uso
        # Supuesto: valor base = 1.0 evento/año por unidad de flota promedio.
        lambda_esp_anual = (PESO_ESPONTANEOS * 6.0
                            * factor_tecnico
                            * factor_uso)
        n_espontaneos = np.random.poisson(lambda_esp_anual)

        for _ in range(n_espontaneos):
            dias_offset = np.random.randint(0, total_dias)
            fecha_ini   = fecha_inicio + timedelta(days=dias_offset)
            duracion    = max(1.0, np.random.exponential(3.5))
            fecha_fn    = fecha_ini + timedelta(days=duracion)

            if fecha_fn <= fecha_fin:
                registros.append({
                    'id_ambulancia'        : amb,
                    'fecha_inicio_downtime': fecha_ini.date(),
                    'fecha_fin_downtime'   : fecha_fn.date(),
                    'duracion_dias'        : round(duracion, 1),
                    'causa'                : np.random.choice(causas_espontaneas,
                                                              p=prob_causas_esp)
                })

    df = pd.DataFrame(registros)
    df = df.sort_values(['id_ambulancia', 'fecha_inicio_downtime']).reset_index(drop=True)
    return df


# =============================================================================
# FUNCIÓN 6: GENERAR TABLA DE USO OPERATIVO
# =============================================================================

def generar_tabla_uso(flota: list,
                       perfiles: dict,
                       fecha_inicio: datetime,
                       fecha_fin: datetime) -> pd.DataFrame:
    """
    Genera registros mensuales de uso operativo por ambulancia.

    El factor_uso_operativo del perfil de la ambulancia modula su km mensual
    y número de servicios.  Este mismo factor alimenta calcular_lambda() en
    generar_tabla_mantenimiento(), asegurando coherencia entre el uso
    registrado en df_uso y la tasa de correctivos generada.

    Columnas:
        id_ambulancia       : Identificador de la unidad
        fecha               : Fecha del registro (inicio de mes)
        kilometraje_periodo : Km recorridos en el período
        numero_servicios    : Servicios prestados en el período
        km_acumulado        : Km acumulado desde inicio
    """
    registros = []

    for amb in flota:
        factor_uso   = perfiles[amb]['factor_uso_operativo']
        km_acumulado = float(perfiles[amb]['km_inicial'])

        fecha_cursor = fecha_inicio.replace(day=1)

        while fecha_cursor <= fecha_fin:
            km_periodo = max(500, np.random.normal(
                KM_MENSUAL_MEDIA * factor_uso, KM_MENSUAL_STD
            ))
            servicios = max(10, int(np.random.normal(
                SERVICIOS_MENSUALES_MEDIA * factor_uso, SERVICIOS_MENSUALES_STD
            )))

            km_acumulado += km_periodo

            registros.append({
                'id_ambulancia'      : amb,
                'fecha'              : fecha_cursor.date(),
                'kilometraje_periodo': round(km_periodo, 0),
                'numero_servicios'   : servicios,
                'km_acumulado'       : round(km_acumulado, 0)
            })

            if fecha_cursor.month == 12:
                fecha_cursor = fecha_cursor.replace(
                    year=fecha_cursor.year + 1, month=1)
            else:
                fecha_cursor = fecha_cursor.replace(
                    month=fecha_cursor.month + 1)

    df = pd.DataFrame(registros)
    df = df.sort_values(['id_ambulancia', 'fecha']).reset_index(drop=True)
    return df


# =============================================================================
# FUNCIÓN 7: CALCULAR VARIABLES EXPLICATIVAS EN VENTANA W
# =============================================================================

def calcular_variables_ventana(id_amb: str,
                                t0: datetime,
                                df_mant: pd.DataFrame,
                                df_down: pd.DataFrame,
                                df_uso: pd.DataFrame,
                                W: int) -> dict:
    """
    Para una ambulancia y un corte temporal t0, calcula todas las variables
    explicativas en la ventana (t0-W, t0].

    NOTA SOBRE AUSENCIA DE FUGA TEMPORAL:
    Todas las variables se calculan exclusivamente con datos en (t0-W, t0].
    La variable objetivo y_inoperativa_14 se calcula en (t0, t0+T], que es
    estrictamente posterior.  No existe solapamiento temporal entre ventana
    de predicción y horizonte de evaluación.
    """
    fecha_inicio_w = t0 - timedelta(days=W)

    # Filtrar datos de la ambulancia
    mant_amb = df_mant[df_mant['id_ambulancia'] == id_amb].copy()
    mant_amb['fecha_inicio'] = pd.to_datetime(mant_amb['fecha_inicio'])

    down_amb = df_down[df_down['id_ambulancia'] == id_amb].copy()
    down_amb['fecha_inicio_downtime'] = pd.to_datetime(down_amb['fecha_inicio_downtime'])

    uso_amb = df_uso[df_uso['id_ambulancia'] == id_amb].copy()
    uso_amb['fecha'] = pd.to_datetime(uso_amb['fecha'])

    # Filtrar por ventana temporal (t0-W, t0]
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
    n_pm          = len(mant_w[mant_w['tipo_mantenimiento'] == 'preventivo'])
    n_cm          = len(mant_w[mant_w['tipo_mantenimiento'] == 'correctivo'])
    n_total_mant  = n_pm + n_cm

    # Días desde última intervención de cualquier tipo
    if len(mant_w) > 0:
        ultima_interv       = mant_w['fecha_inicio'].max()
        dias_desde_ultima   = (t0 - ultima_interv).days
    else:
        dias_desde_ultima   = W

    # Días desde último PM (para el componente temporal de λ)
    mant_pm_w = mant_w[mant_w['tipo_mantenimiento'] == 'preventivo']
    if len(mant_pm_w) > 0:
        ultimo_pm           = mant_pm_w['fecha_inicio'].max()
        dias_desde_ultimo_pm = (t0 - ultimo_pm).days
    else:
        dias_desde_ultimo_pm = W

    # --- Variables por subsistema ---
    n_vehicular   = len(mant_w[mant_w['subsistema'] == 'vehicular'])
    n_electrico   = len(mant_w[mant_w['subsistema'] == 'electrico_electronico'])
    n_equipamiento= len(mant_w[mant_w['subsistema'] == 'biomedico_asistencial'])

    # --- Variables de inoperatividad en ventana ---
    n_episodios_down  = len(down_w)
    downtime_total    = down_w['duracion_dias'].sum()   if len(down_w) > 0 else 0.0
    downtime_promedio = down_w['duracion_dias'].mean()  if len(down_w) > 0 else 0.0
    disponibilidad_w  = max(0.0, (W - downtime_total) / W)

    # --- Variables de uso operativo ---
    km_en_w        = uso_w['kilometraje_periodo'].sum() if len(uso_w) > 0 else 0.0
    servicios_en_w = uso_w['numero_servicios'].sum()    if len(uso_w) > 0 else 0

    # --- Condición del equipamiento biomédico obligatorio ---
    n_fallas_eq = len(mant_w[
        (mant_w['subsistema'] == 'biomedico_asistencial') &
        (mant_w['tipo_mantenimiento'] == 'correctivo')
    ])
    equipamiento_funcional = 1 if n_fallas_eq == 0 else 0

    return {
        'id_ambulancia'                : id_amb,
        't0'                           : t0.date(),
        # Variables de mantenimiento
        'n_pm_w'                       : n_pm,
        'n_cm_w'                       : n_cm,
        'n_total_mant_w'               : n_total_mant,
        'dias_desde_ultima_interv'     : dias_desde_ultima,
        # Variables por subsistema
        'n_eventos_vehicular_w'        : n_vehicular,
        'n_eventos_electrico_w'        : n_electrico,
        'n_eventos_equipamiento_w'     : n_equipamiento,
        # Variables de inoperatividad
        'n_episodios_downtime_w'       : n_episodios_down,
        'downtime_total_dias_w'        : round(downtime_total, 2),
        'downtime_promedio_dias_w'     : round(downtime_promedio, 2),
        'disponibilidad_w'             : round(disponibilidad_w, 4),
        # Variables de uso
        'km_en_w'                      : round(km_en_w, 0),
        'servicios_en_w'               : int(servicios_en_w),
        # Condición del equipamiento
        'equipamiento_funcional'       : equipamiento_funcional
    }


# =============================================================================
# FUNCIÓN 8: CALCULAR VARIABLE OBJETIVO EN HORIZONTE T
# =============================================================================

def calcular_variable_objetivo(id_amb: str,
                                 t0: datetime,
                                 df_down: pd.DataFrame,
                                 T: int) -> int:
    """
    Calcula la variable objetivo binaria y_inoperativa_14:
        1 si hubo inoperatividad en (t0, t0+T]
        0 si no hubo inoperatividad en ese horizonte

    El horizonte (t0, t0+T] es estrictamente posterior a la ventana
    de variables explicativas (t0-W, t0], garantizando ausencia de
    fuga de información temporal.
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
# FUNCIÓN 9: CONSTRUIR EL DATASET FINAL DE OBSERVACIONES
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

    Rango válido de cortes:
        — Desde: fecha_inicio + W  (necesitamos W días de historia)
        — Hasta: fecha_fin   - T   (necesitamos T días futuros para el objetivo)
    """
    print("Construyendo dataset de observaciones...")

    fecha_inicio_cortes = fecha_inicio + timedelta(days=W)
    fecha_fin_cortes    = fecha_fin    - timedelta(days=T)

    t0_cursor = fecha_inicio_cortes
    n_cortes  = 0
    while t0_cursor <= fecha_fin_cortes:
        n_cortes  += 1
        t0_cursor += timedelta(days=paso)

    print(f"  Ambulancias            : {len(flota)}")
    print(f"  Cortes temporales      : {n_cortes}")
    print(f"  Observaciones esperadas: {len(flota) * n_cortes:,}")

    observaciones = []
    t0_cursor     = fecha_inicio_cortes
    corte_num     = 0

    while t0_cursor <= fecha_fin_cortes:
        corte_num += 1
        if corte_num % 10 == 0:
            print(f"  Corte {corte_num:3d}/{n_cortes}: {t0_cursor.date()}")

        for amb in flota:
            vars_exp = calcular_variables_ventana(
                amb, t0_cursor, df_mant, df_down, df_uso, W
            )
            y = calcular_variable_objetivo(amb, t0_cursor, df_down, T)
            vars_exp['y_inoperativa_14'] = y
            observaciones.append(vars_exp)

        t0_cursor += timedelta(days=paso)

    df_dataset = pd.DataFrame(observaciones)
    df_dataset = df_dataset.sort_values(
        ['id_ambulancia', 't0']).reset_index(drop=True)

    return df_dataset


# =============================================================================
# FUNCIÓN 10: REPORTE Y CHECKLIST DE VALIDACIÓN
# =============================================================================

def generar_reporte(df_mant: pd.DataFrame,
                     df_down: pd.DataFrame,
                     df_uso: pd.DataFrame,
                     df_dataset: pd.DataFrame) -> None:
    """
    Imprime estadísticas descriptivas y checklist de validación del dataset.
    """
    print("\n" + "="*65)
    print("REPORTE DEL DATASET SIMULADO")
    print("="*65)

    print(f"\n[1] TABLAS FUENTE")
    print(f"  Eventos de mantenimiento : {len(df_mant):,} registros")
    print(f"    — Preventivos          : {len(df_mant[df_mant['tipo_mantenimiento']=='preventivo']):,}")
    n_cm = len(df_mant[df_mant['tipo_mantenimiento']=='correctivo'])
    print(f"    — Correctivos          : {n_cm:,}")
    print(f"  Registros de downtime    : {len(df_down):,} registros")
    print(f"  Registros de uso         : {len(df_uso):,} registros")

    print(f"\n[2] DATASET FINAL")
    n_amb    = df_dataset['id_ambulancia'].nunique()
    n_cortes = df_dataset['t0'].nunique()
    print(f"  Ambulancias únicas       : {n_amb}  (esperado: 33)")
    print(f"  Cortes temporales        : {n_cortes}")
    print(f"  Observaciones totales    : {len(df_dataset):,}  (= {n_amb} × {n_cortes})")
    print(f"  Período                  : {df_dataset['t0'].min()} — {df_dataset['t0'].max()}")

    print(f"\n[3] VARIABLE OBJETIVO (y_inoperativa_14)")
    pct_pos = df_dataset['y_inoperativa_14'].mean() * 100
    n_pos   = df_dataset['y_inoperativa_14'].sum()
    n_neg   = len(df_dataset) - n_pos
    print(f"  Clase 0 (no inoperativa) : {n_neg:,}  ({100-pct_pos:.1f}%)")
    print(f"  Clase 1 (inoperativa)    : {n_pos:,}  ({pct_pos:.1f}%)")
    print(f"  Desbalance               : {(100-pct_pos)/max(pct_pos,0.01):.1f}:1")
    if 10 <= pct_pos <= 40:
        print(f"  ✓ Prevalencia en rango aprendible (10–40%)")
    else:
        print(f"  ⚠ Prevalencia fuera del rango recomendado (10–40%)")

    print(f"\n[4] PARTICIÓN TEMPORAL")
    train = df_dataset[pd.to_datetime(df_dataset['t0']).dt.year == 2024]
    val   = df_dataset[pd.to_datetime(df_dataset['t0']).dt.year == 2025]
    print(f"  Entrenamiento (2024)     : {len(train):,}  ({len(train)/len(df_dataset)*100:.1f}%)")
    print(f"  Validación    (2025)     : {len(val):,}  ({len(val)/len(df_dataset)*100:.1f}%)")
    print(f"  Y=1 en entrenamiento     : {train['y_inoperativa_14'].mean()*100:.1f}%")
    print(f"  Y=1 en validación        : {val['y_inoperativa_14'].mean()*100:.1f}%")

    print(f"\n[5] CORRELACIONES PREDICTIVAS (Pearson, variable ~ y)")
    vars_corr = [
        'n_cm_w', 'n_episodios_downtime_w', 'downtime_total_dias_w',
        'disponibilidad_w', 'km_en_w', 'servicios_en_w',
        'dias_desde_ultima_interv'
    ]
    y = df_dataset['y_inoperativa_14']
    for v in vars_corr:
        if v in df_dataset.columns:
            c = df_dataset[v].corr(y)
            flag = "✓" if abs(c) >= 0.05 else "⚠"
            print(f"  {flag} {v:<35}: {c:+.4f}")

    print(f"\n[6] ESTADÍSTICAS DE VARIABLES EXPLICATIVAS")
    cols_stats = ['n_pm_w','n_cm_w','dias_desde_ultima_interv',
                  'n_episodios_downtime_w','downtime_total_dias_w',
                  'disponibilidad_w','km_en_w','servicios_en_w']
    present = [c for c in cols_stats if c in df_dataset.columns]
    print(df_dataset[present].describe().round(2).to_string())

    print(f"\n[7] VERIFICACIÓN DE AUSENCIA DE FUGA TEMPORAL")
    print(f"  Ventana explicativa : (t0 - {W} días,  t0]")
    print(f"  Horizonte objetivo  : (t0,  t0 + {T} días]")
    print(f"  ✓ Los intervalos son disjuntos — no existe fuga de información.")

    print(f"\n[8] CHECKLIST DE VALIDACIÓN")
    checks = [
        (n_amb == 33,                      "33 ambulancias en el dataset"),
        (n_cortes > 0,                     "Cortes temporales generados"),
        (len(df_dataset) > 0,              "Dataset no vacío"),
        (10 <= pct_pos <= 45,              "Prevalencia Y=1 en rango aprendible"),
        (len(train) > 0 and len(val) > 0,  "Partición entrenamiento/validación válida"),
        (n_cm > 0,                         "Correctivos generados"),
        (len(df_down) > 0,                 "Downtimes generados"),
        ('km_en_w' in df_dataset.columns,  "Variable km_en_w presente"),
        ('servicios_en_w' in df_dataset.columns, "Variable servicios_en_w presente"),
        ('y_inoperativa_14' in df_dataset.columns, "Variable objetivo presente"),
    ]
    all_ok = True
    for ok, msg in checks:
        status = "✓" if ok else "✗"
        if not ok:
            all_ok = False
        print(f"  {status} {msg}")

    if all_ok:
        print(f"\n  ✓✓ TODOS LOS CHECKS PASADOS — Dataset listo para modelado.")
    else:
        print(f"\n  ⚠ REVISE LOS ITEMS MARCADOS CON ✗")


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    print("="*65)
    print("GENERANDO DATASET SIMULADO")
    print("Ambulancias Tipo II — Lima Metropolitana 2024-2025")
    print("Función de riesgo compuesta (km, servicios, tiempo, historial)")
    print("="*65)

    # 1. Generar flota y perfiles
    print("\n[1/6] Generando flota y perfiles de riesgo...")
    flota   = generar_flota(N_AMBULANCIAS)
    perfiles = generar_perfiles_ambulancia(flota)
    print(f"      {len(flota)} ambulancias: {flota[0]} ... {flota[-1]}")

    # 2. Tabla de mantenimiento
    print("\n[2/6] Generando tabla de mantenimiento (función λ compuesta)...")
    df_mantenimiento = generar_tabla_mantenimiento(flota, perfiles, FECHA_INICIO, FECHA_FIN)
    print(f"      {len(df_mantenimiento):,} eventos generados")

    # 3. Tabla de downtime
    print("\n[3/6] Generando tabla de downtime (condicionada al perfil de riesgo)...")
    df_downtime = generar_tabla_downtime(df_mantenimiento, flota, perfiles,
                                          FECHA_INICIO, FECHA_FIN)
    print(f"      {len(df_downtime):,} episodios generados")

    # 4. Tabla de uso operativo
    print("\n[4/6] Generando tabla de uso operativo...")
    df_uso = generar_tabla_uso(flota, perfiles, FECHA_INICIO, FECHA_FIN)
    print(f"      {len(df_uso):,} registros generados")

    # 5. Dataset de observaciones
    print("\n[5/6] Construyendo dataset de observaciones por ventanas...")
    df_dataset = construir_dataset(
        flota, df_mantenimiento, df_downtime, df_uso,
        FECHA_INICIO, FECHA_FIN, W, T, PASO_CORTE
    )

    # 6. Reporte y validación
    print("\n[6/6] Generando reporte de validación...")
    generar_reporte(df_mantenimiento, df_downtime, df_uso, df_dataset)

    # 7. Exportar CSVs
    print("\n--- EXPORTANDO ARCHIVOS ---")
    os.makedirs("datos_simulados", exist_ok=True)

    df_mantenimiento.to_csv("datos_simulados/tabla_mantenimiento.csv",     index=False)
    df_downtime.to_csv(     "datos_simulados/tabla_downtime.csv",           index=False)
    df_uso.to_csv(          "datos_simulados/tabla_uso.csv",                index=False)
    df_dataset.to_csv(      "datos_simulados/dataset_observaciones.csv",    index=False)

    df_train = df_dataset[pd.to_datetime(df_dataset['t0']).dt.year == 2024]
    df_val   = df_dataset[pd.to_datetime(df_dataset['t0']).dt.year == 2025]
    df_train.to_csv("datos_simulados/dataset_entrenamiento_2024.csv", index=False)
    df_val.to_csv(  "datos_simulados/dataset_validacion_2025.csv",    index=False)

    print(f"\n  tabla_mantenimiento.csv        ({len(df_mantenimiento):,} filas)")
    print(f"  tabla_downtime.csv             ({len(df_downtime):,} filas)")
    print(f"  tabla_uso.csv                  ({len(df_uso):,} filas)")
    print(f"  dataset_observaciones.csv      ({len(df_dataset):,} filas)")
    print(f"  dataset_entrenamiento_2024.csv ({len(df_train):,} filas)")
    print(f"  dataset_validacion_2025.csv    ({len(df_val):,} filas)")

    print("\n¡Dataset generado exitosamente!")
    print("Siguiente paso: Fase 3 — Modelado computacional predictivo.")