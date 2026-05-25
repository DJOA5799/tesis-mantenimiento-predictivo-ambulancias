"""
generar_plantilla_semanal.py
Genera la plantilla semanal de programación de mantenimiento con soporte predictivo.
Toma como entrada el archivo de salida del modelo (lineamiento_priorizacion.csv)
y produce un Excel editable listo para uso operativo.

Uso:
    python generar_plantilla_semanal.py

Entrada esperada (resultados/lineamiento_priorizacion.csv):
    id_ambulancia, t0, prob_inoperatividad, nivel_riesgo,
    dias_desde_ultima_interv, downtime_total_dias_w, n_cm_w,
    disponibilidad_w, equipamiento_funcional,
    km_en_w, servicios_en_w

Salida:
    resultados/plantilla_semanal_YYYYMMDD.xlsx
"""

import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import (
    PatternFill, Font, Alignment, Border, Side, GradientFill
)
from openpyxl.utils import get_column_letter
from openpyxl.styles.numbers import FORMAT_NUMBER_00
from datetime import datetime, date
import os

# ── PARÁMETROS CONFIGURABLES ──────────────────────────────────────────────────
INPUT_FILE   = "resultados/lineamiento_priorizacion.csv"
OUTPUT_DIR   = "resultados"
UMBRAL_ALTO  = 0.50
UMBRAL_MEDIO = 0.25
PLAZO_ALTO   = 3   # días hábiles
PLAZO_MEDIO  = 7   # días hábiles

# Umbrales de alerta (Anexo G)
UMBRAL_SERVICIOS   = 120
UMBRAL_KM          = 8000
UMBRAL_DIAS_SIN_PM = 45
UMBRAL_DISP        = 0.85
UMBRAL_DOWNTIME    = 7.0

# ── COLORES ───────────────────────────────────────────────────────────────────
C_HEADER_DARK  = "042C53"   # azul marino
C_HEADER_MID   = "185FA5"   # azul medio
C_HEADER_LIGHT = "E6F1FB"   # azul claro texto
C_ALTO_BG      = "FCEBEB"   # rojo claro
C_ALTO_DARK    = "A32D2D"   # rojo oscuro
C_MEDIO_BG     = "FAEEDA"   # ámbar claro
C_MEDIO_DARK   = "854F0B"   # ámbar oscuro
C_BAJO_BG      = "EAF3DE"   # verde claro
C_BAJO_DARK    = "3B6D11"   # verde oscuro
C_MEJORA_BG    = "E1F5EE"   # teal claro
C_MEJORA_DARK  = "085041"   # teal oscuro
C_SECTION_BG   = "F1EFE8"   # gris claro
C_WHITE        = "FFFFFF"
C_BORDER       = "B4B2A9"
C_ALERT_RED    = "F09595"
C_ALERT_AMB    = "FAC775"

def fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)

def font(bold=False, color="000000", size=10, italic=False):
    return Font(bold=bold, color=color, size=size, italic=italic,
                name="Calibri")

def border_thin():
    s = Side(style="thin", color=C_BORDER)
    return Border(left=s, right=s, top=s, bottom=s)

def border_medium():
    s = Side(style="medium", color="888780")
    return Border(left=s, right=s, top=s, bottom=s)

def align(h="left", v="center", wrap=False):
    return Alignment(horizontal=h, vertical=v, wrap_text=wrap)

# ── GENERAR DATOS DE EJEMPLO (si no existe el CSV) ────────────────────────────
def generar_datos_ejemplo():
    """Genera datos simulados del corte 12-dic-2025 para demo."""
    np.random.seed(42)
    n = 35
    ids = [f"AMB-T2-{i:03d}" for i in range(1, n+1)]
    t0 = "2025-12-12"

    probs = np.concatenate([
        np.random.uniform(0.50, 0.85, 13),
        np.random.uniform(0.25, 0.50, 20),
        np.random.uniform(0.05, 0.25,  2)
    ])
    np.random.shuffle(probs)

    def nivel(p):
        if p >= UMBRAL_ALTO:  return "ALTO"
        if p >= UMBRAL_MEDIO: return "MEDIO"
        return "BAJO"

    df = pd.DataFrame({
        "id_ambulancia":        ids,
        "t0":                   t0,
        "prob_inoperatividad":  probs.round(4),
        "nivel_riesgo":         [nivel(p) for p in probs],
        "dias_desde_ultima_interv": np.random.randint(5, 60, n),
        "downtime_total_dias_w":np.random.exponential(3, n).round(1),
        "n_cm_w":               np.random.poisson(0.8, n).astype(int),
        "disponibilidad_w":     np.random.uniform(0.80, 1.00, n).round(4),
        "equipamiento_funcional": np.random.choice([0,1], n, p=[0.05,0.95]),
        "km_en_w":              np.random.normal(2800, 600, n).round(0).astype(int),
        "servicios_en_w":       np.random.normal(45, 12, n).round(0).astype(int),
    })

    # Forzar valores extremos en alto riesgo para que activen alertas
    alto_idx = df[df["nivel_riesgo"]=="ALTO"].index[:5]
    df.loc[alto_idx[:2], "km_en_w"]  = [9200, 8700]
    df.loc[alto_idx[:2], "servicios_en_w"] = [135, 128]
    df.loc[alto_idx[2],  "downtime_total_dias_w"] = 9.4
    df.loc[alto_idx[2],  "disponibilidad_w"]      = 0.78
    df.loc[alto_idx[3],  "dias_desde_ultima_interv"] = 52

    df = df.sort_values("prob_inoperatividad", ascending=False).reset_index(drop=True)
    return df

# ── EVALUAR ALERTAS ────────────────────────────────────────────────────────────
def evaluar_alertas(row):
    alertas = []
    if row["servicios_en_w"] > UMBRAL_SERVICIOS:
        alertas.append(f"Servicios >120 ({int(row['servicios_en_w'])})")
    if row["km_en_w"] > UMBRAL_KM:
        alertas.append(f"Km >8 000 ({int(row['km_en_w'])} km)")
    if row["dias_desde_ultima_interv"] > UMBRAL_DIAS_SIN_PM:
        alertas.append(f">45 días sin PM ({int(row['dias_desde_ultima_interv'])}d)")
    if row["disponibilidad_w"] < UMBRAL_DISP:
        alertas.append(f"Disp. <85% ({row['disponibilidad_w']*100:.0f}%)")
    if row["downtime_total_dias_w"] > UMBRAL_DOWNTIME:
        alertas.append(f"Downtime >{UMBRAL_DOWNTIME}d ({row['downtime_total_dias_w']}d)")
    if row["equipamiento_funcional"] == 0:
        alertas.append("Equipo biomédico no funcional")
    return alertas

def instruccion(row, alertas):
    nivel = row["nivel_riesgo"]
    if nivel == "ALTO":
        base = "Inspección técnica general"
        if any("Km" in a for a in alertas):
            base += " + revisión tren motriz"
        if any("Downtime" in a or "Disp" in a for a in alertas):
            base += " + análisis causa raíz inoperatividad"
        if any(">45 días" in a for a in alertas):
            base += " + PM inmediato (intervalo vencido)"
        if any("biomédico" in a for a in alertas):
            base += " + reemplazo/reparación equipo biomédico"
        return base
    elif nivel == "MEDIO":
        base = "Inspección dirigida"
        if alertas:
            subsistemas = []
            if any("Km" in a or "Servicios" in a for a in alertas):
                subsistemas.append("sistema vehicular")
            if any("Downtime" in a or "Disp" in a for a in alertas):
                subsistemas.append("historial inoperatividad")
            if any(">45 días" in a for a in alertas):
                subsistemas.append("PM próximo a vencer")
            base += ": " + " + ".join(subsistemas) if subsistemas else ""
        return base
    else:
        return "Continuar operación normal. PM según cronograma NTS 051."

# ── CONSTRUIR EL EXCEL ────────────────────────────────────────────────────────
def construir_excel(df):
    wb = Workbook()

    # ── HOJA 1: PLANTILLA SEMANAL ─────────────────────────────────────────────
    ws = wb.active
    ws.title = "Programación semanal"
    ws.sheet_view.showGridLines = False

    corte = df["t0"].iloc[0] if "t0" in df.columns else str(date.today())
    try:
        corte_dt = datetime.strptime(str(corte), "%Y-%m-%d")
        semana   = corte_dt.isocalendar()[1]
        anio     = corte_dt.year
        corte_str = corte_dt.strftime("%d/%m/%Y")
    except:
        corte_str = str(corte); semana = "—"; anio = 2025

    n_alto  = len(df[df["nivel_riesgo"]=="ALTO"])
    n_medio = len(df[df["nivel_riesgo"]=="MEDIO"])
    n_bajo  = len(df[df["nivel_riesgo"]=="BAJO"])
    total   = len(df)

    # Estimación de unidades que el preventivo no hubiera programado:
    # Unidades de riesgo ALTO/MEDIO cuyo dias_desde_ultima_interv < 37 días
    df["alertas"]     = df.apply(lambda r: evaluar_alertas(r), axis=1)
    df["instruccion"] = df.apply(lambda r: instruccion(r, r["alertas"]), axis=1)
    df_no_prev = df[
        (df["nivel_riesgo"].isin(["ALTO","MEDIO"])) &
        (df["dias_desde_ultima_interv"] < 37)
    ]
    n_mejora = len(df_no_prev)

    def set_row_height(ws, row, height):
        ws.row_dimensions[row].height = height

    def merge_and_write(ws, r, c1, c2, value, fill_c, font_c, bold=True,
                        size=10, h="left", wrap=False, italic=False):
        cell = ws.cell(row=r, column=c1, value=value)
        if c1 != c2:
            ws.merge_cells(start_row=r, start_column=c1,
                           end_row=r, end_column=c2)
        cell.fill     = fill(fill_c)
        cell.font     = font(bold=bold, color=font_c, size=size, italic=italic)
        cell.alignment = align(h, "center", wrap)
        return cell

    # Column widths: A=24, B=8, C=10, D=10, E=40, F=35, G=18, H=18
    col_w = [0, 24, 8, 10, 10, 40, 35, 18, 18]
    for i, w in enumerate(col_w[1:], 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    r = 1
    # ── CABECERA ──────────────────────────────────────────────────────────────
    set_row_height(ws, r, 28)
    merge_and_write(ws, r, 1, 6,
        "PROGRAMACIÓN SEMANAL DE MANTENIMIENTO — SOPORTE PREDICTIVO",
        C_HEADER_DARK, C_HEADER_LIGHT, size=12, bold=True, h="center")
    merge_and_write(ws, r, 7, 8,
        f"Semana {semana} / {anio}  |  Corte: {corte_str}",
        C_HEADER_DARK, C_HEADER_LIGHT, size=10, bold=False, h="center")

    r += 1
    set_row_height(ws, r, 16)
    merge_and_write(ws, r, 1, 8,
        "Ambulancias médicas urbanas Tipo II  ·  SAMU Lima Metropolitana  ·  "
        f"Flota total: {total} unidades  ·  Horizonte predicción: 14 días  ·  Umbral: τ = 0,30",
        C_HEADER_MID, C_HEADER_LIGHT, size=9, bold=False, h="center")

    # ── KPI ROW ───────────────────────────────────────────────────────────────
    r += 2
    set_row_height(ws, r, 14)
    kpis = [
        (1, 2, f"{n_alto}  RIESGO ALTO",   C_ALTO_BG,   C_ALTO_DARK),
        (3, 4, f"{n_medio}  RIESGO MEDIO", C_MEDIO_BG,  C_MEDIO_DARK),
        (5, 6, f"{n_bajo}  RIESGO BAJO",   C_BAJO_BG,   C_BAJO_DARK),
        (7, 8, f"+{n_mejora}  NO DETECTADAS POR PREVENTIVO", C_MEJORA_BG, C_MEJORA_DARK),
    ]
    for c1, c2, txt, bg, fg in kpis:
        merge_and_write(ws, r, c1, c2, txt, bg, fg,
                        bold=True, size=11, h="center")

    r += 1
    set_row_height(ws, r, 26)
    subtexts = [
        (1,2, f"P ≥ 0,50 · Intervenir en ≤ {PLAZO_ALTO} días hábiles"),
        (3,4, f"0,25 ≤ P < 0,50 · Intervenir en ≤ {PLAZO_MEDIO} días hábiles"),
        (5,6, "P < 0,25 · Solo NTS 051 cuando corresponda"),
        (7,8, "Unidades con riesgo alto/medio detectadas antes de vencer intervalo preventivo"),
    ]
    fills_sub = [C_ALTO_BG, C_MEDIO_BG, C_BAJO_BG, C_MEJORA_BG]
    fgcols    = [C_ALTO_DARK, C_MEDIO_DARK, C_BAJO_DARK, C_MEJORA_DARK]
    for (c1,c2,txt), bg, fg in zip(subtexts, fills_sub, fgcols):
        merge_and_write(ws, r, c1, c2, txt, bg, fg,
                        bold=False, size=8, h="center", wrap=True, italic=True)

    # ── BANNER MEJORA ─────────────────────────────────────────────────────────
    r += 2
    set_row_height(ws, r, 38)
    merge_and_write(ws, r, 1, 8,
        f"MEJORA RESPECTO AL PREVENTIVO NTS 051: El modelo detecta {n_mejora} unidades con riesgo alto o medio "
        f"cuyo intervalo preventivo NO vence esta semana. Sin soporte predictivo, estas unidades habrían "
        f"continuado en operación con probabilidad de inoperatividad > 0,50 en los próximos 14 días.",
        C_MEJORA_BG, C_MEJORA_DARK, bold=False, size=9, h="left", wrap=True)

    # ── TABLA DETALLE ─────────────────────────────────────────────────────────
    def section_header(ws, r, texto, bg, fg):
        set_row_height(ws, r, 14)
        merge_and_write(ws, r, 1, 8, texto, bg, fg,
                        bold=True, size=9, h="left")

    def col_headers(ws, r):
        set_row_height(ws, r, 28)
        headers = [
            (1, "Unidad"), (2, "P(inop.)"), (3, "Riesgo"),
            (4, "N.° alertas"), (5, "Alertas activas"),
            (6, "Instrucción al taller"), (7, "Plazo máximo"), (8, "Observaciones")
        ]
        for c, h in headers:
            cell = ws.cell(row=r, column=c, value=h)
            cell.fill      = fill(C_HEADER_DARK)
            cell.font      = font(bold=True, color=C_HEADER_LIGHT, size=9)
            cell.alignment = align("center", "center", True)
            cell.border    = border_thin()

    for nivel, bg_sec, fg_sec, plazo_txt in [
        ("ALTO",  C_ALTO_BG,  C_ALTO_DARK,
         f"Solicitar ingreso al taller ANTES del *** (≤ {PLAZO_ALTO} días hábiles desde corte)"),
        ("MEDIO", C_MEDIO_BG, C_MEDIO_DARK,
         f"Programar inspección dirigida ANTES del *** (≤ {PLAZO_MEDIO} días hábiles desde corte)"),
        ("BAJO",  C_BAJO_BG,  C_BAJO_DARK,
         "Sin intervención anticipada · Mantener cronograma preventivo NTS 051"),
    ]:
        r += 2
        section_header(ws, r, f"  PRIORIDAD {['ALTO','MEDIO','BAJO'].index(nivel)+1} — RIESGO {nivel}  ·  {plazo_txt}", bg_sec, fg_sec)
        r += 1
        col_headers(ws, r)

        sub = df[df["nivel_riesgo"]==nivel].reset_index(drop=True)
        for _, row_data in sub.iterrows():
            r += 1
            set_row_height(ws, r, 38)
            alertas   = row_data["alertas"]
            n_alertas = len(alertas)
            alerta_str = "\n".join(alertas) if alertas else "—"
            inst_str  = row_data["instruccion"]

            if nivel == "ALTO":
                row_bg = C_ALTO_BG if n_alertas >= 2 else "FFF5F5"
            elif nivel == "MEDIO":
                row_bg = C_MEDIO_BG if n_alertas >= 1 else "FFFDF7"
            else:
                row_bg = C_BAJO_BG

            vals = [
                row_data["id_ambulancia"],
                round(float(row_data["prob_inoperatividad"]), 4),
                nivel,
                n_alertas,
                alerta_str,
                inst_str,
                f"≤ {PLAZO_ALTO}d hábiles" if nivel=="ALTO" else
                (f"≤ {PLAZO_MEDIO}d hábiles" if nivel=="MEDIO" else "Según km NTS 051"),
                ""
            ]
            for c, v in enumerate(vals, 1):
                cell = ws.cell(row=r, column=c, value=v)
                cell.fill      = fill(row_bg)
                cell.border    = border_thin()
                cell.alignment = align("center" if c in [2,3,4,7] else "left",
                                       "center", wrap=True)
                cell.font      = font(
                    bold=(c==1),
                    color=(C_ALTO_DARK if nivel=="ALTO" else
                           C_MEDIO_DARK if nivel=="MEDIO" else C_BAJO_DARK)
                           if c==3 else "000000",
                    size=9
                )
                if c == 2:
                    cell.number_format = "0.0000"

    # ── RESUMEN TRAMITACIÓN ────────────────────────────────────────────────────
    r += 3
    set_row_height(ws, r, 14)
    merge_and_write(ws, r, 1, 8,
        "  RESUMEN PARA TRAMITACIÓN CON EL TALLER",
        C_HEADER_DARK, C_HEADER_LIGHT, bold=True, size=10)

    r += 1
    set_row_height(ws, r, 22)
    res_hdrs = ["Tipo de solicitud","N.° unidades","Plazo máximo ingreso",
                "Tipo de servicio a solicitar","","","",""]
    for c, h in enumerate(res_hdrs[:4], 1):
        cell = ws.cell(row=r, column=c, value=h)
        cell.fill = fill(C_HEADER_MID)
        cell.font = font(bold=True, color=C_HEADER_LIGHT, size=9)
        cell.alignment = align("center","center",True)
        cell.border = border_thin()
    ws.merge_cells(start_row=r,start_column=4,end_row=r,end_column=8)

    res_data = [
        ("URGENTE — Riesgo ALTO",    str(n_alto),
         f"≤ {PLAZO_ALTO} días hábiles",
         "Inspección técnica general + PM anticipado (si km corresponde) + revisión por alertas activas",
         C_ALTO_BG, C_ALTO_DARK),
        ("PROGRAMADO — Riesgo MEDIO",str(n_medio),
         f"≤ {PLAZO_MEDIO} días hábiles",
         "Inspección dirigida por alertas activas + PM estándar si intervalo vence esta semana",
         C_MEDIO_BG, C_MEDIO_DARK),
        ("DIFERIDO — Riesgo BAJO",   str(n_bajo),
         "Según cronograma NTS 051",
         "Solo PM estándar cuando corresponda por kilometraje acumulado",
         C_BAJO_BG,  C_BAJO_DARK),
        (f"ADICIONAL — Detectadas solo por modelo (+{n_mejora})", str(n_mejora),
         "Esta semana",
         "Unidades sin intervalo vencido pero con riesgo alto/medio. No programadas por solo preventivo.",
         C_MEJORA_BG, C_MEJORA_DARK),
    ]
    for tipo, nu, plazo, serv, bg, fg in res_data:
        r += 1
        set_row_height(ws, r, 30)
        ws.merge_cells(start_row=r,start_column=4,end_row=r,end_column=8)
        for c, v in enumerate([tipo, nu, plazo, serv], 1):
            cell = ws.cell(row=r, column=c, value=v)
            cell.fill      = fill(bg)
            cell.font      = font(bold=(c==1), color=fg, size=9)
            cell.alignment = align("center" if c in [2,3] else "left",
                                   "center", wrap=True)
            cell.border    = border_thin()

    # ── FIRMAS ────────────────────────────────────────────────────────────────
    r += 3
    set_row_height(ws, r, 14)
    merge_and_write(ws, r, 1, 4,
        "Elaborado por — Encargado de gestión de datos:",
        C_SECTION_BG, "444441", bold=False, size=9)
    merge_and_write(ws, r, 5, 8,
        "Aprobado por — Jefe de mantenimiento:",
        C_SECTION_BG, "444441", bold=False, size=9)

    r += 1
    set_row_height(ws, r, 32)
    merge_and_write(ws, r, 1, 4, f"Nombre y firma:                     Fecha: {corte_str}",
                    C_WHITE, "000000", bold=False, size=9)
    merge_and_write(ws, r, 5, 8,  "Nombre y firma:                     Fecha: ___/___/____",
                    C_WHITE, "000000", bold=False, size=9)

    # ── NOTA PIE ─────────────────────────────────────────────────────────────
    r += 2
    set_row_height(ws, r, 38)
    merge_and_write(ws, r, 1, 8,
        "Fuente: Elaboración propia. Plantilla generada a partir del modelo computacional Gradient Boosting "
        "(Entregable 2) y lineamientos técnicos (Entregable 3). Alertas evaluadas conforme Anexo G (NTS N.º "
        "051-MINSA/OGDN-V.01, MINSA 2006). Esta plantilla es una propuesta metodológica basada en datos "
        f"simulados. Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        C_SECTION_BG, "5F5E5A", bold=False, size=8, h="left", wrap=True, italic=True)

    # ── HOJA 2: DATOS COMPLETOS ───────────────────────────────────────────────
    ws2 = wb.create_sheet("Datos completos")
    ws2.sheet_view.showGridLines = True

    export_cols = [
        "id_ambulancia","t0","prob_inoperatividad","nivel_riesgo",
        "dias_desde_ultima_interv","downtime_total_dias_w","n_cm_w",
        "disponibilidad_w","equipamiento_funcional","km_en_w","servicios_en_w"
    ]
    df_export = df[export_cols].copy()
    df_export["alertas_activas"] = df["alertas"].apply(lambda x: " | ".join(x) if x else "—")
    df_export["instruccion_taller"] = df["instruccion"]

    headers2 = list(df_export.columns)
    for c, h in enumerate(headers2, 1):
        cell = ws2.cell(row=1, column=c, value=h)
        cell.fill = fill(C_HEADER_DARK)
        cell.font = font(bold=True, color=C_HEADER_LIGHT, size=9)
        cell.alignment = align("center","center",True)

    for ri, row_data in df_export.iterrows():
        nivel = row_data["nivel_riesgo"]
        row_bg = (C_ALTO_BG if nivel=="ALTO" else
                  C_MEDIO_BG if nivel=="MEDIO" else C_BAJO_BG)
        for c, v in enumerate(row_data, 1):
            cell = ws2.cell(row=ri+2, column=c, value=v)
            cell.fill = fill(row_bg)
            cell.font = font(size=9)
            cell.alignment = align("center","center",True)

    for c in range(1, len(headers2)+1):
        ws2.column_dimensions[get_column_letter(c)].width = 18
    ws2.freeze_panes = "A2"

    # ── GUARDAR ───────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        corte_fmt = datetime.strptime(str(corte), "%Y-%m-%d").strftime("%Y%m%d")
    except:
        corte_fmt = datetime.now().strftime("%Y%m%d")

    out_path = os.path.join(OUTPUT_DIR,
                            f"plantilla_semanal_{corte_fmt}.xlsx")
    wb.save(out_path)
    return out_path

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Cargando datos...")
    if os.path.exists(INPUT_FILE):
        df = pd.read_csv(INPUT_FILE)
        print(f"  Archivo cargado: {INPUT_FILE} ({len(df)} filas)")
    else:
        print(f"  Archivo '{INPUT_FILE}' no encontrado.")
        print("  Generando datos de ejemplo para demostración...")
        df = generar_datos_ejemplo()

    print("Generando Excel...")
    path = construir_excel(df)
    print(f"\n✓ Plantilla generada: {path}")
    print(f"  Unidades ALTO:  {len(df[df['nivel_riesgo']=='ALTO'])}")
    print(f"  Unidades MEDIO: {len(df[df['nivel_riesgo']=='MEDIO'])}")
    print(f"  Unidades BAJO:  {len(df[df['nivel_riesgo']=='BAJO'])}")
