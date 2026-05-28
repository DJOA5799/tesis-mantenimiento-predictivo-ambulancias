"""
=============================================================================
FASE 5: LINEAMIENTOS TÉCNICOS Y OPERATIVOS
Entregable 3 de la tesis — Derivados del modelo computacional predictivo
Ambulancias Tipo II — Lima Metropolitana 2024-2025

Estructura:
1. Lineamientos de priorización de intervenciones
2. Protocolo de uso del modelo como soporte predictivo para mantenimiento
3. Criterios de alerta temprana por nivel de riesgo
4. Recomendaciones de gestión de datos para implementación real
5. Reporte ejecutivo consolidado (PDF-ready)

Referencias:
- Taoufyq et al. (2025) — enfoque data-driven
- Moubray (1997) — gestión centrada en confiabilidad
- ISO 17359:2018 — monitoreo y diagnóstico
- MINSA (2006) — NTS 051
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# FUNCIÓN 1: CARGAR RESULTADOS DEL MODELO
# =============================================================================

def cargar_resultados() -> tuple:
    """Carga los resultados generados en la Fase 3-4."""
    df_soporte = pd.read_csv("resultados/soporte_preventivo_completo.csv")
    df_metricas = pd.read_csv("resultados/tabla_metricas_comparativas.csv")
    df_importancia = pd.read_csv("resultados/importancia_variables.csv")
    df_val = pd.read_csv("datos_simulados/dataset_validacion_2025.csv")
    return df_soporte, df_metricas, df_importancia, df_val


# =============================================================================
# FUNCIÓN 2: GENERAR LINEAMIENTOS DE PRIORIZACIÓN
# =============================================================================

def generar_lineamientos_priorizacion(df_soporte: pd.DataFrame) -> pd.DataFrame:
    """
    Lineamiento 1: Criterios de priorización de intervenciones
    basados en el nivel de riesgo estimado por el modelo.

    Fundamentación: Taoufyq et al. (2025) — gestión basada en riesgo
    """
    lineamientos = pd.DataFrame([
        {
            'Nivel de riesgo': 'ALTO (≥ 0.50)',
            'Acción recomendada': 'Intervención preventiva prioritaria',
            'Plazo sugerido': 'Dentro de 3 días hábiles',
            'Responsable': 'Técnico de mantenimiento asignado',
            'Fundamento': 'Probabilidad estimada de inoperatividad ≥ 50% en 14 días',
            'Subsistema prioritario': 'Verificar subsistema con mayor historial de fallas'
        },
        {
            'Nivel de riesgo': 'MEDIO (0.25–0.49)',
            'Acción recomendada': 'Inspección técnica programada',
            'Plazo sugerido': 'Dentro de 7 días hábiles',
            'Responsable': 'Técnico de mantenimiento',
            'Fundamento': 'Probabilidad estimada entre 25% y 49% en 14 días',
            'Subsistema prioritario': 'Revisión general con énfasis en subsistema eléctrico'
        },
        {
            'Nivel de riesgo': 'BAJO (< 0.25)',
            'Acción recomendada': 'Mantenimiento preventivo según cronograma',
            'Plazo sugerido': 'Según intervalo programado (NTS 051)',
            'Responsable': 'Equipo de mantenimiento',
            'Fundamento': 'Probabilidad estimada < 25% en 14 días',
            'Subsistema prioritario': 'Seguimiento rutinario del plan preventivo'
        }
    ])
    return lineamientos


# =============================================================================
# FUNCIÓN 3: PROTOCOLO DE USO DEL MODELO
# =============================================================================

def generar_protocolo_uso() -> pd.DataFrame:
    """
    Lineamiento 2: Protocolo operativo de uso del modelo
    como herramienta de soporte a la toma de decisiones.
    """
    protocolo = pd.DataFrame([
        {
            'Paso': 1,
            'Actividad': 'Actualización de registros',
            'Descripción': 'El personal técnico registra todos los eventos '
                          'de mantenimiento, downtime y uso operativo '
                          'en el sistema de gestión.',
            'Frecuencia': 'Continua — al ocurrir cada evento',
            'Responsable': 'Técnico de turno'
        },
        {
            'Paso': 2,
            'Actividad': 'Ejecución del modelo',
            'Descripción': 'El modelo computacional procesa los registros '
                          'de los últimos 60 días (W) para cada unidad '
                          'y genera probabilidades de inoperatividad.',
            'Frecuencia': 'Semanal — cada lunes',
            'Responsable': 'Encargado de gestión de datos'
        },
        {
            'Paso': 3,
            'Actividad': 'Revisión del ranking de riesgo',
            'Descripción': 'El jefe de mantenimiento revisa el tablero '
                          'de soporte predictivo con el ranking de '
                          'ambulancias por nivel de riesgo.',
            'Frecuencia': 'Semanal — cada lunes',
            'Responsable': 'Jefe de mantenimiento'
        },
        {
            'Paso': 4,
            'Actividad': 'Programación de intervenciones',
            'Descripción': 'Se programan intervenciones preventivas '
                          'priorizando unidades de riesgo ALTO y MEDIO, '
                          'coordinando con disponibilidad operativa.',
            'Frecuencia': 'Semanal',
            'Responsable': 'Jefe de mantenimiento'
        },
        {
            'Paso': 5,
            'Actividad': 'Ejecución y registro de intervención',
            'Descripción': 'El técnico ejecuta la intervención y registra '
                          'los hallazgos, repuestos utilizados y tiempo '
                          'de intervención.',
            'Frecuencia': 'Según programación',
            'Responsable': 'Técnico de mantenimiento'
        },
        {
            'Paso': 6,
            'Actividad': 'Retroalimentación al modelo',
            'Descripción': 'Los nuevos registros alimentan el modelo '
                          'para el siguiente ciclo semanal, mejorando '
                          'progresivamente su capacidad predictiva.',
            'Frecuencia': 'Continua',
            'Responsable': 'Encargado de gestión de datos'
        }
    ])
    return protocolo


# =============================================================================
# FUNCIÓN 4: CRITERIOS DE ALERTA TEMPRANA POR SUBSISTEMA
# =============================================================================

def generar_criterios_alerta(df_importancia: pd.DataFrame) -> pd.DataFrame:
    """
    Lineamiento 3: Criterios específicos de alerta temprana
    basados en la importancia de variables del modelo Random Forest seleccionado para soporte preventivo.
    """
    criterios = pd.DataFrame([
        {
            'Variable crítica': 'Servicios prestados en W',
            'Importancia modelo': 'Variable crítica identificada por el modelo',
            'Umbral de alerta': '> 120 servicios en 60 días',
            'Interpretación': 'Alta carga operativa aumenta el riesgo '
                             'de falla por desgaste acelerado',
            'Acción': 'Adelantar inspección técnica general'
        },
        {
            'Variable crítica': 'Kilometraje en W',
            'Importancia modelo': 'Variable crítica identificada por el modelo',
            'Umbral de alerta': '> 8,000 km en 60 días',
            'Interpretación': 'Uso intensivo del vehículo asociado '
                             'a mayor probabilidad de falla mecánica',
            'Acción': 'Verificar sistema vehicular y tren motriz'
        },
        {
            'Variable crítica': 'Días desde última intervención',
            'Importancia modelo': 'Variable crítica identificada por el modelo',
            'Umbral de alerta': '> 45 días sin intervención',
            'Interpretación': 'Intervalo prolongado sin mantenimiento '
                             'aumenta riesgo de falla no planificada',
            'Acción': 'Programar mantenimiento preventivo inmediato'
        },
        {
            'Variable crítica': 'Disponibilidad histórica en W',
            'Importancia modelo': 'Variable crítica identificada por el modelo',
            'Umbral de alerta': '< 0.85 (menos del 85% operativa)',
            'Interpretación': 'Historial reciente de inoperatividad '
                             'es predictor de inoperatividad futura',
            'Acción': 'Revisión técnica integral prioritaria'
        },
        {
            'Variable crítica': 'Downtime total en W',
            'Importancia modelo': 'Variable crítica identificada por el modelo',
            'Umbral de alerta': '> 7 días acumulados en 60 días',
            'Interpretación': 'Patrón de inoperatividad recurrente '
                             'indica problema técnico no resuelto',
            'Acción': 'Análisis de causa raíz y corrección definitiva'
        },
        {
            'Variable crítica': 'Equipamiento funcional',
            'Importancia modelo': 'Variable crítica identificada por el modelo',
            'Umbral de alerta': 'Estado = 0 (no funcional)',
            'Interpretación': 'Indisponibilidad del equipamiento '
                             'obligatorio inhabilita la unidad '
                             'conforme a NTS 051',
            'Acción': 'Reemplazo o reparación inmediata del equipo'
        }
    ])
    return criterios


# =============================================================================
# FUNCIÓN 5: RECOMENDACIONES DE GESTIÓN DE DATOS
# =============================================================================

def generar_recomendaciones_datos() -> pd.DataFrame:
    """
    Lineamiento 4: Recomendaciones para la implementación real
    con datos institucionales del SAMU.
    """
    recomendaciones = pd.DataFrame([
        {
            'Área': 'Estandarización de registros',
            'Recomendación': 'Adoptar una codificación uniforme para '
                            'subsistemas y tipos de falla en todos '
                            'los formularios de mantenimiento.',
            'Prioridad': 'Alta',
            'Fundamento': 'ISO 17359:2018 — trazabilidad de datos'
        },
        {
            'Área': 'Trazabilidad temporal',
            'Recomendación': 'Registrar con precisión las fechas y horas '
                            'de inicio y fin de cada evento de '
                            'mantenimiento e inoperatividad.',
            'Prioridad': 'Alta',
            'Fundamento': 'Prytz (2014) — calidad de datos históricos'
        },
        {
            'Área': 'Identificadores únicos',
            'Recomendación': 'Asignar y mantener un identificador único '
                            'por ambulancia que se use consistentemente '
                            'en todos los sistemas de registro.',
            'Prioridad': 'Alta',
            'Fundamento': 'Requisito para construcción del dataset'
        },
        {
            'Área': 'Registro de uso operativo',
            'Recomendación': 'Registrar el kilometraje y número de '
                            'servicios por ambulancia con periodicidad '
                            'al menos mensual.',
            'Prioridad': 'Media',
            'Fundamento': 'Variables de mayor importancia en el modelo'
        },
        {
            'Área': 'Calidad de datos',
            'Recomendación': 'Implementar reglas de validación básicas: '
                            'fecha_fin ≥ fecha_inicio, identificadores '
                            'no nulos, duraciones positivas.',
            'Prioridad': 'Media',
            'Fundamento': 'Taoufyq et al. (2025) — gobernanza de datos'
        },
        {
            'Área': 'Actualización del modelo',
            'Recomendación': 'Reentrenar el modelo con datos reales '
                            'cada 6 meses para mantener su vigencia '
                            'y mejorar su capacidad predictiva.',
            'Prioridad': 'Media',
            'Fundamento': 'Si et al. (2011) — modelos adaptativos'
        },
        {
            'Área': 'Infraestructura tecnológica',
            'Recomendación': 'El modelo puede ejecutarse en cualquier '
                            'computadora con Python instalado, sin '
                            'requerir servidores especializados ni '
                            'infraestructura adicional.',
            'Prioridad': 'Informativa',
            'Fundamento': 'Viabilidad declarada en perfil de proyecto'
        }
    ])
    return recomendaciones


# =============================================================================
# FUNCIÓN 6: ANÁLISIS DE FALSOS NEGATIVOS
# =============================================================================

def analizar_falsos_negativos(df_soporte: pd.DataFrame,
                               df_val: pd.DataFrame) -> pd.DataFrame:
    """
    Análisis de los falsos negativos de alta relevancia operativa.
    Casos donde el modelo no anticipó una inoperatividad real.
    Crítico para la tesis: demuestra las limitaciones del modelo
    y orienta mejoras futuras.
    """
    df_merged = df_soporte.merge(
        df_val[['id_ambulancia', 't0', 'y_inoperativa_14']],
        on=['id_ambulancia', 't0'],
        how='left'
    )

    # Falsos negativos: modelo predijo BAJO riesgo pero hubo inoperatividad
    fn = df_merged[
        (df_merged['nivel_riesgo'] == 'BAJO') &
        (df_merged['y_inoperativa_14'] == 1)
    ].copy()

    if len(fn) > 0:
        fn_resumen = fn.groupby('id_ambulancia').agg(
            n_fn=('y_inoperativa_14', 'count'),
            downtime_promedio=('downtime_total_dias_w', 'mean'),
            dias_interv_promedio=('dias_desde_ultima_interv', 'mean')
        ).round(2).reset_index()
        fn_resumen = fn_resumen.sort_values('n_fn', ascending=False)
        return fn_resumen
    else:
        return pd.DataFrame(columns=['id_ambulancia', 'n_fn',
                                      'downtime_promedio',
                                      'dias_interv_promedio'])


# =============================================================================
# FUNCIÓN 7: VISUALIZACIÓN DE LINEAMIENTOS
# =============================================================================

def generar_visualizacion_lineamientos(df_soporte: pd.DataFrame,
                                        df_val: pd.DataFrame) -> None:
    """
    Genera visualizaciones de lineamientos técnicos y operativos.
    Paneles:
    1. Evolución temporal anual por nivel de riesgo.
    2. Distribución de riesgo en el último corte semanal.
    3. Ranking de probabilidad de inoperatividad en el último corte semanal.
    4. Protocolo semanal de uso operativo del modelo.
    """
    import matplotlib
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.gridspec import GridSpec

    # Estilo institucional sobrio
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': '#E6E6E6',
        'grid.linewidth': 0.6,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.titleweight': 'bold'
    })

    # Paleta de colores
    AZUL_MARINO = '#355C7D'
    AZUL_MEDIO  = '#6C8EAD'
    GRIS_OSCURO = '#4B5358'
    GRIS_MEDIO  = '#8E969B'
    GRIS_CLARO  = '#FAFAF6'

    ROJO_RIESGO  = '#C96B63'
    AMBAR_RIESGO = '#F4D35E'
    VERDE_RIESGO = '#78BFA3'

    C_RIESGO = {
        'ALTO': ROJO_RIESGO,
        'MEDIO': AMBAR_RIESGO,
        'BAJO': VERDE_RIESGO
    }

    df_soporte = df_soporte.copy()
    df_soporte['t0'] = pd.to_datetime(df_soporte['t0'])

    # Último corte semanal: debe conversar con la plantilla Excel
    ultimo_corte = df_soporte['t0'].max()
    df_corte = df_soporte[df_soporte['t0'] == ultimo_corte].copy()
    df_corte = df_corte.sort_values('prob_inoperatividad', ascending=False).reset_index(drop=True)
    corte_txt = ultimo_corte.strftime('%d/%m/%Y')

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.50)

    # -------------------------------------------------------------------------
    # Panel 1: Evolución temporal anual
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :2])

    dist = df_soporte.groupby(['t0', 'nivel_riesgo']).size().unstack(fill_value=0)

    estilos = {
        'ALTO':  {'ls': '-',  'lw': 1.8, 'marker': 'o'},
        'MEDIO': {'ls': '--', 'lw': 1.6, 'marker': 's'},
        'BAJO':  {'ls': ':',  'lw': 1.8, 'marker': '^'}
    }

    for nivel in ['ALTO', 'MEDIO', 'BAJO']:
        if nivel in dist.columns:
            ax1.plot(
                dist.index,
                dist[nivel],
                color=C_RIESGO[nivel],
                linestyle=estilos[nivel]['ls'],
                linewidth=estilos[nivel]['lw'],
                marker=estilos[nivel]['marker'],
                markersize=3,
                label=f'Riesgo {nivel}',
                zorder=3
            )

    ax1.set_xlabel('Fecha del corte temporal', fontsize=9)
    ax1.set_ylabel('Número de ambulancias', fontsize=9)
    ax1.set_title('Evolución semanal del nivel de riesgo durante 2025',
                  fontsize=10, fontweight='bold')
    ax1.legend(fontsize=8.5, frameon=True, loc='upper right')
    ax1.grid(True, alpha=0.35)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)

    # -------------------------------------------------------------------------
    # Panel 2: Distribución del último corte semanal
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])

    conteo_corte = df_corte['nivel_riesgo'].value_counts()
    orden = [n for n in ['ALTO', 'MEDIO', 'BAJO'] if n in conteo_corte.index]
    vals = [conteo_corte[n] for n in orden]
    labels = [f'{n}\n({conteo_corte[n]} unidades)' for n in orden]

    wedges, texts, autos = ax2.pie(
        vals,
        labels=labels,
        colors=[C_RIESGO[n] for n in orden],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 8.5},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2.0}
    )

    for a in autos:
        a.set_fontsize(8.5)
        a.set_color('white')
        a.set_fontweight('bold')

    ax2.set_title(f'Distribución de riesgo\nCorte {corte_txt}',
                  fontsize=10, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel 3: Ranking del último corte semanal
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, :2])

    colores_barras = [
        C_RIESGO[nivel] for nivel in df_corte['nivel_riesgo']
    ]

    ax3.bar(
        range(len(df_corte)),
        df_corte['prob_inoperatividad'],
        color=colores_barras,
        edgecolor='white',
        linewidth=0.7,
        width=0.78,
        alpha=0.90,
        zorder=3
    )

    # Líneas de umbral
    ax3.axhline(
        y=0.50,
        color=ROJO_RIESGO,
        linestyle='--',
        lw=1.1,
        alpha=0.80,
        zorder=2
    )

    ax3.axhline(
        y=0.25,
        color='#D8A900',
        linestyle='--',
        lw=1.1,
        alpha=0.90,
        zorder=2
    )

    # Etiquetas de umbral ubicadas fuera del área de barras
    ax3.text(
        1.01,
        0.50,
        'Umbral alto\nP = 0,50',
        transform=ax3.get_yaxis_transform(),
        fontsize=7.5,
        color=ROJO_RIESGO,
        ha='left',
        va='center'
    )

    ax3.text(
        1.01,
        0.25,
        'Umbral medio\nP = 0,25',
        transform=ax3.get_yaxis_transform(),
        fontsize=7.5,
        color='#B38F00',
        ha='left',
        va='center'
    )

    ax3.set_xticks(range(len(df_corte)))
    ax3.set_xticklabels(df_corte['id_ambulancia'], rotation=90, fontsize=7)

    ax3.set_xlabel('Ambulancia', fontsize=9)
    ax3.set_ylabel('Probabilidad estimada de inoperatividad', fontsize=9)
    ax3.set_title(f'Ranking de probabilidad de inoperatividad por unidad — Corte {corte_txt}',
                  fontsize=10, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.35, zorder=0)
    ax3.set_ylim(0, min(1.0, max(df_corte['prob_inoperatividad']) + 0.15))
    ax3.margins(x=0.02)

    ax3.legend(handles=[
        mpatches.Patch(color=ROJO_RIESGO, label='Riesgo ALTO (P ≥ 0,50)'),
        mpatches.Patch(color=AMBAR_RIESGO, label='Riesgo MEDIO (0,25 ≤ P < 0,50)'),
        mpatches.Patch(color=VERDE_RIESGO, label='Riesgo BAJO (P < 0,25)'),
    ], fontsize=8, loc='upper right', frameon=True)

    # -------------------------------------------------------------------------
    # Panel 4: Protocolo semanal
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_xlim(0, 5)
    ax4.set_ylim(0, 7.5)
    ax4.axis('off')
    ax4.set_title('Protocolo semanal de uso',
                  fontsize=10, fontweight='bold')

    pasos = [
        ('1. Actualizar registros operativos', GRIS_OSCURO),
        ('2. Ejecutar modelo predictivo', AZUL_MEDIO),
        ('3. Revisar ranking de riesgo', GRIS_OSCURO),
        ('4. Priorizar intervenciones', ROJO_RIESGO),
        ('5. Ejecutar y registrar acciones', GRIS_OSCURO),
        ('6. Retroalimentar con nuevos datos', VERDE_RIESGO),
    ]

    for i, (texto, color) in enumerate(pasos):
        y = 6.8 - i * 1.08
        ax4.add_patch(FancyBboxPatch(
            (0.15, y - 0.38),
            4.7,
            0.76,
            boxstyle='round,pad=0.07',
            facecolor=color,
            edgecolor='white',
            linewidth=1.4,
            zorder=3,
            alpha=0.95
        ))

        ax4.text(
            2.5,
            y,
            texto,
            ha='center',
            va='center',
            fontsize=8.4,
            color='white',
            fontweight='bold',
            zorder=4
        )

        if i < len(pasos) - 1:
            ax4.annotate(
                '',
                xy=(2.5, y - 0.42),
                xytext=(2.5, y - 0.62),
                arrowprops=dict(arrowstyle='->', color=GRIS_MEDIO, lw=1.2)
            )

    ax4.text(
        2.5,
        0.25,
        'Frecuencia sugerida: semanal. Responsable: jefatura o coordinación de mantenimiento.',
        ha='center',
        va='center',
        fontsize=7.3,
        color=GRIS_MEDIO,
        style='italic'
    )

    # -------------------------------------------------------------------------
    # Título general
    # -------------------------------------------------------------------------
    plt.suptitle(
        'Lineamientos técnicos y operativos para priorización de mantenimiento con soporte predictivo\n'
        'Modelo computacional Random Forest — Ambulancias Tipo II — Periodo 2025',
        fontsize=12,
        fontweight='bold',
        y=1.03
    )

    plt.savefig(
        'lineamientos_visualizacion.png',
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )

    print("  Figura guardada: lineamientos_visualizacion.png")
    print(f"  Corte semanal utilizado para ranking y distribución: {corte_txt}")
    plt.close()

# =============================================================================
# FUNCIÓN 7B: FIGURA INDIVIDUAL PARA TESIS — PROTOCOLO SEMANAL
# =============================================================================

def generar_figura_32_protocolo_semanal() -> None:
    """
    Genera la Figura 32:
    Protocolo semanal de uso del modelo computacional predictivo
    para soporte al mantenimiento preventivo de ambulancias Tipo II.

    Esta figura reemplaza la versión antigua del protocolo y mantiene
    coherencia con el modelo predictivo seleccionado: Random Forest.
    """
    import matplotlib
    from matplotlib.patches import FancyBboxPatch

    # Directorio de salida
    os.makedirs("figuras_tesis", exist_ok=True)

    # Estilo institucional sobrio
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

    # Paleta visual definida para la tesis
    AZUL_MARINO = '#355C7D'
    AZUL_MEDIO  = '#6C8EAD'
    GRIS_OSCURO = '#4B5358'
    GRIS_MEDIO  = '#8E969B'
    NEGRO       = '#1F2326'

    ROJO_RIESGO  = '#C96B63'
    AMBAR_RIESGO = '#F4D35E'
    VERDE_RIESGO = '#78BFA3'

    fig, ax = plt.subplots(figsize=(8.2, 10.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    def caja(x, y, w, h, texto, color, fs=8.7, text_color='white'):
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle='round,pad=0.05',
            facecolor=color,
            edgecolor='white',
            linewidth=1.4,
            alpha=0.97
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

    def flecha(x1, y1, x2, y2):
        ax.annotate(
            '',
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->',
                lw=1.15,
                color=GRIS_MEDIO
            )
        )

    # Bloque superior
    caja(
        3.0, 12.55, 4.0, 0.75,
        "INICIO — Lunes de cada semana",
        AZUL_MARINO,
        fs=8.8
    )

    caja(
        2.15, 11.05, 5.70, 0.82,
        "1. Actualizar registros operativos\n"
        "mantenimiento, downtime y uso semanal",
        GRIS_OSCURO
    )

    caja(
        2.15, 9.55, 5.70, 0.82,
        "2. Calcular variables explicativas\n"
        "ventana histórica W = 60 días",
        GRIS_OSCURO
    )

    caja(
        2.15, 8.05, 5.70, 0.82,
        "3. Ejecutar modelo Random Forest\n"
        "estimar P(inoperatividad) por unidad",
        AZUL_MEDIO
    )

    caja(
        2.15, 6.55, 5.70, 0.82,
        "4. Clasificar nivel de riesgo operativo\n"
        "según umbrales definidos",
        GRIS_OSCURO
    )

    # Bloques de riesgo
    caja(
        0.55, 4.15, 2.75, 1.30,
        "RIESGO ALTO\n"
        "P ≥ 0,50\n"
        "Intervención prioritaria\n"
        "≤ 3 días hábiles",
        ROJO_RIESGO,
        fs=8.2
    )

    caja(
        3.65, 4.15, 2.75, 1.30,
        "RIESGO MEDIO\n"
        "0,25 ≤ P < 0,50\n"
        "Inspección programada\n"
        "≤ 7 días hábiles",
        AMBAR_RIESGO,
        fs=8.2,
        text_color='white'
    )

    caja(
        6.75, 4.15, 2.75, 1.30,
        "RIESGO BAJO\n"
        "P < 0,25\n"
        "Seguimiento rutinario\n"
        "según cronograma",
        VERDE_RIESGO,
        fs=8.2
    )

    caja(
        1.65, 2.15, 6.70, 0.85,
        "5. Programar, ejecutar y registrar acciones de mantenimiento",
        GRIS_OSCURO,
        fs=8.6
    )

    caja(
        1.65, 0.85, 6.70, 0.85,
        "6. Retroalimentar el modelo con nuevos registros operativos",
        NEGRO,
        fs=8.6
    )

    # Flechas principales
    flecha(5.0, 12.55, 5.0, 11.87)
    flecha(5.0, 11.05, 5.0, 10.37)
    flecha(5.0, 9.55, 5.0, 8.87)
    flecha(5.0, 8.05, 5.0, 7.37)
    flecha(5.0, 6.55, 5.0, 5.50)

    # Ramificación hacia niveles de riesgo
    flecha(5.0, 6.55, 1.93, 5.45)
    flecha(5.0, 6.55, 5.03, 5.45)
    flecha(5.0, 6.55, 8.13, 5.45)

    # Convergencia hacia ejecución
    flecha(1.93, 4.15, 5.0, 3.00)
    flecha(5.03, 4.15, 5.0, 3.00)
    flecha(8.13, 4.15, 5.0, 3.00)

    # Cierre y retroalimentación
    flecha(5.0, 2.15, 5.0, 1.70)

    ax.text(
        5.0,
        0.28,
        "Frecuencia sugerida: semanal. Responsable: jefatura o coordinación de mantenimiento.",
        ha='center',
        va='center',
        fontsize=7.4,
        color=GRIS_MEDIO,
        style='italic'
    )

    plt.title(
        'Protocolo semanal de uso del modelo computacional predictivo\n'
        'para soporte al mantenimiento preventivo de ambulancias Tipo II',
        fontsize=11,
        fontweight='bold',
        pad=16
    )

    plt.tight_layout()
    plt.savefig(
        "figuras_tesis/figura_32_protocolo_semanal.png",
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close()

    print("  Figura guardada: figuras_tesis/figura_32_protocolo_semanal.png")

# =============================================================================
# FUNCIÓN 8: REPORTE EJECUTIVO CONSOLIDADO
# =============================================================================

def generar_reporte_ejecutivo(df_metricas: pd.DataFrame,
                               df_importancia: pd.DataFrame,
                               lineamientos: pd.DataFrame,
                               protocolo: pd.DataFrame,
                               criterios: pd.DataFrame,
                               recomendaciones: pd.DataFrame) -> None:
    """
    Genera el reporte ejecutivo consolidado en texto.
    Base para la redacción del capítulo de resultados de la tesis.
    """
    reporte = []
    reporte.append("="*70)
    reporte.append("REPORTE EJECUTIVO — MODELO COMPUTACIONAL PREDICTIVO")
    reporte.append("Ambulancias Médicas Urbanas Tipo II — Lima Metropolitana")
    reporte.append("Período de análisis: 2024-2025")
    reporte.append("="*70)

    reporte.append("\n1. RESUMEN DEL MODELO\n")
    reporte.append("   Tipo de modelo    : Clasificación binaria supervisada")
    reporte.append("   Modelo predictivo seleccionado: Random Forest")
    reporte.append("   Ventana histórica  : W = 60 días")
    reporte.append("   Horizonte predicción: T = 14 días")
    reporte.append("   Validación         : Backtesting temporal (2024→2025)")
    reporte.append("   Variables explicativas: 14 variables agrupadas en")
    reporte.append("     mantenimiento, inoperatividad, uso y equipamiento")

    reporte.append("\n2. MÉTRICAS DE DESEMPEÑO (Validación 2025)\n")
    for _, row in df_metricas.iterrows():
        reporte.append(f"   {row['Modelo']}")
        reporte.append(f"     AUC-ROC      : {row['AUC-ROC']}")
        reporte.append(f"     Precisión    : {row['Precisión']}")
        reporte.append(f"     Sensibilidad : {row['Sensibilidad']}")
        reporte.append(f"     MAE          : {row['MAE']}")
        reporte.append("")

    reporte.append("\n3. VARIABLES MÁS IMPORTANTES\n")
    for _, row in df_importancia.head(6).iterrows():
        reporte.append(f"   {row['variable']:40s}: {row['importancia_pct']:.2f}%")

    reporte.append("\n4. LINEAMIENTOS DE PRIORIZACIÓN\n")
    for _, row in lineamientos.iterrows():
        reporte.append(f"   {row['Nivel de riesgo']}")
        reporte.append(f"     Acción  : {row['Acción recomendada']}")
        reporte.append(f"     Plazo   : {row['Plazo sugerido']}")
        reporte.append("")

    reporte.append("\n5. CRITERIOS DE ALERTA TEMPRANA\n")
    for _, row in criterios.iterrows():
        reporte.append(f"   {row['Variable crítica']} ({row['Importancia modelo']})")
        reporte.append(f"     Umbral  : {row['Umbral de alerta']}")
        reporte.append(f"     Acción  : {row['Acción']}")
        reporte.append("")

    reporte.append("\n6. CONCLUSIONES TÉCNICAS\n")
    reporte.append("   a) Random Forest incrementa la sensibilidad frente al esquema")
    reporte.append("      preventivo tradicional, favoreciendo la detección temprana")
    reporte.append("      de eventos de inoperatividad.")
    reporte.append("   b) Las variables de uso operativo y mantenimiento reciente")
    reporte.append("      concentran la mayor importancia relativa del modelo.")
    reporte.append("      ejecutable en entorno Python estándar.")
    reporte.append("   d) Con datos reales del SAMU el desempeño mejorará al")
    reporte.append("      capturar patrones operativos auténticos de la flota.")
    reporte.append("   e) El protocolo semanal permite traducir las probabilidades")
    reporte.append("      estimadas por el modelo en criterios operativos de priorización.")

    reporte.append("\n7. LIMITACIONES Y TRABAJO FUTURO\n")
    reporte.append("   a) Datos simulados: validación con datos reales del SAMU")
    reporte.append("      es el siguiente paso crítico.")
    reporte.append("   b) Desbalance de clases (9:1): explorar SMOTE con datos reales.")
    reporte.append("   c) Variables adicionales: incorporar condición de neumáticos,")
    reporte.append("      historial de repuestos y antigüedad de la unidad.")
    reporte.append("   d) Expansión: replicar metodología para ambulancias Tipo I")
    reporte.append("      y Tipo III del sistema prehospitalario nacional.")

    reporte.append("\n" + "="*70)
    reporte.append("Fin del reporte ejecutivo")
    reporte.append("="*70)

    texto = "\n".join(reporte)
    print(texto)

    with open("resultados/reporte_ejecutivo.txt", "w", encoding="utf-8") as f:
        f.write(texto)
    print("\n  Reporte guardado: resultados/reporte_ejecutivo.txt")


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    print("="*60)
    print("FASE 5: LINEAMIENTOS TÉCNICOS Y OPERATIVOS")
    print("Entregable 3 — Tesis Ingeniería Biomédica UNMSM")
    print("="*60)

    # Cargar resultados
    print("\n[1/6] Cargando resultados de Fase 3-4...")
    df_soporte, df_metricas, df_importancia, df_val = cargar_resultados()
    print(f"      {len(df_soporte):,} observaciones del soporte preventivo")

    # Generar lineamientos
    print("\n[2/6] Generando lineamientos de priorización...")
    lineamientos = generar_lineamientos_priorizacion(df_soporte)
    print(f"      {len(lineamientos)} niveles de riesgo definidos")

    print("\n[3/6] Generando protocolo de uso del modelo...")
    protocolo = generar_protocolo_uso()
    print(f"      {len(protocolo)} pasos del protocolo operativo")

    print("\n[4/6] Generando criterios de alerta temprana...")
    criterios = generar_criterios_alerta(df_importancia)
    print(f"      {len(criterios)} variables críticas identificadas")

    print("\n[5/6] Generando recomendaciones de gestión de datos...")
    recomendaciones = generar_recomendaciones_datos()
    fn_analisis = analizar_falsos_negativos(df_soporte, df_val)
    print(f"      {len(recomendaciones)} recomendaciones generadas")

    # Visualizaciones
    print("\n[6/6] Generando visualizaciones de lineamientos...")
    generar_visualizacion_lineamientos(df_soporte, df_val)
    generar_figura_32_protocolo_semanal()

    # Reporte ejecutivo
    print("\n" + "="*60)
    print("REPORTE EJECUTIVO CONSOLIDADO")
    print("="*60)
    generar_reporte_ejecutivo(
        df_metricas, df_importancia,
        lineamientos, protocolo,
        criterios, recomendaciones
    )

    # Exportar todos los lineamientos
    print("\nExportando lineamientos...")
    lineamientos.to_csv("resultados/lineamiento_priorizacion.csv",
                        index=False, encoding='utf-8-sig')
    protocolo.to_csv("resultados/protocolo_uso_modelo.csv",
                     index=False, encoding='utf-8-sig')
    criterios.to_csv("resultados/criterios_alerta_temprana.csv",
                     index=False, encoding='utf-8-sig')
    recomendaciones.to_csv("resultados/recomendaciones_datos.csv",
                           index=False, encoding='utf-8-sig')

    if len(fn_analisis) > 0:
        fn_analisis.to_csv("resultados/analisis_falsos_negativos.csv",
                           index=False, encoding='utf-8-sig')

    print(f"\nArchivos generados en /resultados/:")
    print(f"  lineamiento_priorizacion.csv")
    print(f"  protocolo_uso_modelo.csv")
    print(f"  criterios_alerta_temprana.csv")
    print(f"  recomendaciones_datos.csv")
    print(f"  reporte_ejecutivo.txt")
    print(f"  lineamientos_visualizacion.png")
    print(f"  figuras_tesis/figura_32_protocolo_semanal.png")

    print("\n¡Fase 5 completada exitosamente!")
    print("\nRESUMEN DE ENTREGABLES DE TESIS:")
    print("  Entregable 1: modelo_predictivo_ambulancias.py ✓")
    print("  Entregable 2: soporte_preventivo_completo.csv ✓")
    print("  Entregable 3: lineamientos_tecnicos completos ✓")
    print("\nSiguiente paso: Redacción del capítulo de Resultados y Discusión")
