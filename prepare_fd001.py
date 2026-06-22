import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# PREPARACIÓN DATASET NASA C-MAPSS FD001
# Proyecto: Dashboard GAF - Mantenimiento Predictivo
# Autor: Sebastián Carrillo
# Objetivo:
#   Transformar train_FD001.txt en tablas limpias para Power BI:
#   - fact_engine_cycles.csv
#   - dim_engine.csv
#   - dim_engine_failure_state.csv
#   - dim_risk_level.csv
# =====================================================


# =========================
# 1. Rutas
# =========================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "train_FD001.txt"
OUTPUT_DIR = BASE_DIR / "processed"

OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# 2. Definir columnas
# =========================
# Estructura estándar C-MAPSS FD001:
# engine_id, cycle, 3 operational settings, 21 sensors

columns = (
    ["engine_id", "cycle"]
    + ["setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# =========================
# 3. Cargar datos
# =========================

df = pd.read_csv(
    INPUT_FILE,
    sep=r"\s+",
    header=None,
    names=columns
)


# =========================
# 4. Calcular ciclo de falla y RUL
# =========================
# En train_FD001 cada motor tiene datos hasta su ciclo final.
# Por eso, el ciclo máximo por engine_id se considera su ciclo de falla.

failure_cycles = (
    df.groupby("engine_id")["cycle"]
    .max()
    .reset_index()
    .rename(columns={"cycle": "failure_cycle"})
)

df = df.merge(failure_cycles, on="engine_id", how="left")

# Remaining Useful Life
df["rul"] = df["failure_cycle"] - df["cycle"]

# Porcentaje de vida consumida y vida remanente
df["life_consumed_pct"] = df["cycle"] / df["failure_cycle"]
df["rul_pct"] = df["rul"] / df["failure_cycle"]


# =========================
# 5. Health Index
# =========================
# Índice simple y defendible:
# 100 = activo sano al inicio
# 0 = activo llegando a falla

df["health_index"] = (df["rul_pct"] * 100).clip(lower=0, upper=100)


# =========================
# 6. Risk Score
# =========================
# Riesgo aumenta cuando baja el Health Index.
# Luego se puede mejorar usando sensores y modelo predictivo.

df["risk_score"] = 100 - df["health_index"]


# =========================
# 7. Clasificación de riesgo
# =========================
# Regla basada en RUL.
# Estos umbrales se pueden convertir después en parámetro What-if en Power BI.

def classify_risk(rul):
    if rul <= 20:
        return "Crítico"
    elif rul <= 50:
        return "Alto"
    elif rul <= 90:
        return "Medio"
    else:
        return "Bajo"


def recommend_action(risk_level):
    if risk_level == "Crítico":
        return "Intervención prioritaria"
    elif risk_level == "Alto":
        return "Programar mantención"
    elif risk_level == "Medio":
        return "Seguimiento preventivo"
    else:
        return "Monitoreo normal"


df["risk_level"] = df["rul"].apply(classify_risk)
df["recommended_action"] = df["risk_level"].apply(recommend_action)


# =========================
# 8. Contexto industrial simulado
# =========================
# Esto transforma el dataset NASA en un caso GAF más defendible.
# No cambia los datos técnicos; agrega contexto de gestión.

np.random.seed(42)

engine_ids = sorted(df["engine_id"].unique())

site_options = [
    "Planta Norte",
    "Planta Sur",
    "Planta Concentradora"
]

area_options = [
    "Línea 1",
    "Línea 2",
    "Línea 3",
    "Servicios Auxiliares"
]

asset_family_options = [
    "Motor crítico",
    "Equipo rotativo",
    "Turbina industrial"
]

engine_context = pd.DataFrame({
    "engine_id": engine_ids,
    "site": np.random.choice(site_options, size=len(engine_ids)),
    "area": np.random.choice(area_options, size=len(engine_ids)),
    "asset_family": np.random.choice(asset_family_options, size=len(engine_ids)),
    "criticality_class": np.random.choice(
        ["A - Alta", "B - Media", "C - Baja"],
        size=len(engine_ids),
        p=[0.35, 0.45, 0.20]
    )
})

df = df.merge(engine_context, on="engine_id", how="left")


# =========================
# 9A. Crear dimensión de activos en estado final de falla
# =========================
# Esta tabla representa el último ciclo real de cada motor.
# En train_FD001 todos llegan a falla, por eso current_rul será 0.
# Se deja como respaldo técnico, no como tabla principal del dashboard ejecutivo.

last_cycle_idx = df.groupby("engine_id")["cycle"].idxmax()

dim_engine_failure_state = df.loc[last_cycle_idx, [
    "engine_id",
    "site",
    "area",
    "asset_family",
    "criticality_class",
    "failure_cycle",
    "cycle",
    "rul",
    "health_index",
    "risk_score",
    "risk_level",
    "recommended_action"
]].copy()

dim_engine_failure_state = dim_engine_failure_state.rename(columns={
    "cycle": "current_cycle",
    "rul": "current_rul",
    "health_index": "current_health_index",
    "risk_score": "current_risk_score",
    "risk_level": "current_risk_level",
    "recommended_action": "current_recommended_action"
})


# =========================
# 9B. Crear snapshot operacional variado
# =========================
# En lugar de tomar el último ciclo antes de falla para todos los activos,
# simulamos un "ciclo actual" distinto para cada motor.
# Esto representa una flota en operación con activos en distintos estados:
# sanos, medios, altos y críticos.

np.random.seed(123)

snapshot_rows = []

for engine_id in engine_ids:
    engine_data = df[df["engine_id"] == engine_id].copy()
    failure_cycle = engine_data["failure_cycle"].max()

    # Simulamos que el activo está entre 35% y 98% de su vida consumida.
    # Así tendremos una distribución razonable de estados de salud.
    life_position = np.random.uniform(0.35, 0.98)
    selected_cycle = int(round(failure_cycle * life_position))

    # Evitar seleccionar fuera de rango.
    # Dejamos como máximo failure_cycle - 1 para evitar que todos terminen en falla.
    selected_cycle = max(1, min(selected_cycle, failure_cycle - 1))

    selected_row = engine_data[engine_data["cycle"] == selected_cycle]

    # Si por redondeo no encuentra el ciclo exacto,
    # toma el ciclo más cercano inferior.
    if selected_row.empty:
        selected_row = engine_data[engine_data["cycle"] <= selected_cycle].tail(1)

    snapshot_rows.append(selected_row)

dim_engine = pd.concat(snapshot_rows, ignore_index=True)

dim_engine = dim_engine[[
    "engine_id",
    "site",
    "area",
    "asset_family",
    "criticality_class",
    "failure_cycle",
    "cycle",
    "rul",
    "health_index",
    "risk_score",
    "risk_level",
    "recommended_action"
]].copy()

dim_engine = dim_engine.rename(columns={
    "cycle": "current_cycle",
    "rul": "current_rul",
    "health_index": "current_health_index",
    "risk_score": "current_risk_score",
    "risk_level": "current_risk_level",
    "recommended_action": "current_recommended_action"
})


# =========================
# 10. Dimensión de riesgo
# =========================
# Tabla manual para ordenar niveles de riesgo y asociar acciones recomendadas.

dim_risk_level = pd.DataFrame({
    "risk_level": ["Bajo", "Medio", "Alto", "Crítico"],
    "risk_order": [1, 2, 3, 4],
    "risk_description": [
        "Activo con vida remanente suficiente y sin acción inmediata.",
        "Activo requiere seguimiento preventivo.",
        "Activo requiere programación de mantenimiento.",
        "Activo requiere intervención prioritaria."
    ],
    "recommended_action": [
        "Monitoreo normal",
        "Seguimiento preventivo",
        "Programar mantención",
        "Intervención prioritaria"
    ]
})


# =========================
# 11. Exportar CSV
# =========================

df.to_csv(
    OUTPUT_DIR / "fact_engine_cycles.csv",
    index=False,
    encoding="utf-8-sig"
)

dim_engine.to_csv(
    OUTPUT_DIR / "dim_engine.csv",
    index=False,
    encoding="utf-8-sig"
)

dim_engine_failure_state.to_csv(
    OUTPUT_DIR / "dim_engine_failure_state.csv",
    index=False,
    encoding="utf-8-sig"
)

dim_risk_level.to_csv(
    OUTPUT_DIR / "dim_risk_level.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================
# 12. Resumen de control
# =========================

print("Archivos generados correctamente:")
print(f"- {OUTPUT_DIR / 'fact_engine_cycles.csv'}")
print(f"- {OUTPUT_DIR / 'dim_engine.csv'}")
print(f"- {OUTPUT_DIR / 'dim_engine_failure_state.csv'}")
print(f"- {OUTPUT_DIR / 'dim_risk_level.csv'}")
print()

print("Resumen fact_engine_cycles:")
print("Filas:", len(df))
print("Motores únicos:", df["engine_id"].nunique())
print("Columnas:", len(df.columns))
print()

print("Resumen dim_engine snapshot operacional:")
print("Filas:", len(dim_engine))
print("RUL promedio:", round(dim_engine["current_rul"].mean(), 2))
print("Health Index promedio:", round(dim_engine["current_health_index"].mean(), 2))
print("Risk Score promedio:", round(dim_engine["current_risk_score"].mean(), 2))
print()

print("Distribución de riesgo en dim_engine:")
print(dim_engine["current_risk_level"].value_counts())
print()

print("Vista previa dim_engine:")
print(dim_engine.head())