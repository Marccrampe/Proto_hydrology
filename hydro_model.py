import numpy as np
import pandas as pd


def get_event_intensity_mm_hr(event_row: pd.Series) -> float:
    intensity = event_row.get("intensity_mm_hr", None)
    if intensity is not None and pd.notna(intensity):
        return float(intensity)
    return float(event_row["rainfall_mm"]) / max(float(event_row["duration_hr"]), 1.0)


def rainfall_volume_m3(rainfall_mm: float, area_km2: float) -> float:
    """
    Rainfall volume over watershed.
    rainfall_mm: event rainfall depth in mm
    area_km2: watershed area in km²
    """
    rainfall_m = rainfall_mm / 1000.0
    area_m2 = area_km2 * 1_000_000.0
    return rainfall_m * area_m2


def synthetic_event_hyetograph(
    rainfall_mm: float,
    duration_hr: float,
    n_steps: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a simple synthetic event rainfall hyetograph (mm/hr).
    """
    t = np.linspace(0, duration_hr, n_steps)
    center = duration_hr * 0.45
    width = max(duration_hr / 6.0, 0.75)

    raw = np.exp(-((t - center) ** 2) / (2 * width ** 2))
    raw = raw / raw.sum()

    dt_hr = t[1] - t[0]
    intensity = raw * rainfall_mm / dt_hr
    return t, intensity


def synthetic_unit_response(
    duration_hr: float,
    lag_hr: float,
    n_steps: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple synthetic watershed transfer shape.
    """
    t = np.linspace(0, duration_hr * 2.2, n_steps)

    center = max(lag_hr, 0.5)
    width = max(duration_hr / 7.0, 0.8)

    u = np.exp(-((t - center) ** 2) / (2 * width ** 2))
    recession = np.exp(-0.035 * np.maximum(t - center, 0))
    u = u * recession
    u = u / np.trapezoid(u, t)
    return t, u


def build_baseline_hydrograph_from_event(
    event_row: pd.Series,
    watershed_row: pd.Series,
    n_steps: int = 320,
) -> dict:
    """
    Build a synthetic but hydrologically-structured baseline event response.
    """
    rainfall_mm = float(event_row["rainfall_mm"])
    duration_hr = float(event_row["duration_hr"])
    intensity_mm_hr = get_event_intensity_mm_hr(event_row)

    area_km2 = float(watershed_row["area_km2"])
    impervious_pct = float(watershed_row["impervious_pct"])
    initial_abstraction_mm = float(watershed_row.get("initial_abstraction_mm", 8))

    # Effective rainfall
    Pe_mm = max(rainfall_mm - initial_abstraction_mm, 0.0)

    # Synthetic effective runoff coefficient
    # Higher for more impervious watersheds and more intense events
    c_eff = 0.18 + 0.45 * (impervious_pct / 100.0) + 0.12 * min(intensity_mm_hr / 10.0, 1.0)
    c_eff = min(max(c_eff, 0.05), 0.90)

    runoff_depth_mm = Pe_mm * c_eff
    V_rain = rainfall_volume_m3(rainfall_mm, area_km2)
    V_runoff = rainfall_volume_m3(runoff_depth_mm, area_km2)

    # Basin lag proxy
    lag_hr = 1.0 + 4.0 * (1.0 - impervious_pct / 100.0) + 0.01 * duration_hr
    lag_hr = max(0.8, lag_hr)

    t_rain, hyeto = synthetic_event_hyetograph(rainfall_mm=runoff_depth_mm, duration_hr=duration_hr, n_steps=n_steps)
    t_uh, uh = synthetic_unit_response(duration_hr=duration_hr, lag_hr=lag_hr, n_steps=n_steps)

    dt_hr = t_rain[1] - t_rain[0]
    dt_sec = dt_hr * 3600.0

    # Convert excess rainfall over watershed to flow input
    # runoff depth per step (mm/hr) -> m³/s equivalent before routing
    area_m2 = area_km2 * 1_000_000.0
    excess_m_per_hr = hyeto / 1000.0
    inflow_m3_per_s = excess_m_per_hr * area_m2 / 3600.0

    q = np.convolve(inflow_m3_per_s, uh, mode="full")[: len(t_rain)] * dt_hr
    t = t_rain

    # Scale to target runoff volume
    q_volume = np.trapezoid(q, t * 3600.0)
    if q_volume > 0:
        q = q * (V_runoff / q_volume)

    peak_flow = float(np.max(q))
    t_peak = float(t[np.argmax(q)])

    return {
        "time_hr": t,
        "hyetograph_mm_hr": hyeto,
        "baseline_q_m3s": q,
        "rainfall_volume_m3": V_rain,
        "runoff_volume_m3": V_runoff,
        "effective_runoff_coeff": V_runoff / V_rain if V_rain > 0 else 0.0,
        "peak_flow_m3s": peak_flow,
        "t_peak_hr": t_peak,
        "lag_hr": lag_hr,
        "runoff_depth_mm": runoff_depth_mm,
    }


def apply_nbs_to_hydrograph(
    baseline: dict,
    selected_df: pd.DataFrame,
    event_row: pd.Series,
) -> dict:
    """
    Apply literature-based NBS adjustments to an event hydrograph.
    """
    rainfall_mm = float(event_row["rainfall_mm"])
    q_base = baseline["baseline_q_m3s"].copy()
    t = baseline["time_hr"].copy()

    total_runoff_red = 0.0
    total_peak_red = 0.0
    total_lag_hr = 0.0
    rows = []

    for _, row in selected_df.iterrows():
        coverage = float(row["coverage_pct"]) / 100.0
        runoff_red = float(row["max_effect_runoff"]) * coverage
        peak_red = float(row["max_effect_peak"]) * coverage
        lag_add = float(row["max_effect_lag_hr"]) * coverage

        if rainfall_mm >= 250 and row["family"] == "GI":
            runoff_red *= 0.72
            peak_red *= 0.75
            lag_add *= 0.85

        if rainfall_mm >= 250 and row["family"] == "Storage":
            runoff_red *= 0.90
            peak_red *= 0.95
            lag_add *= 0.95

        total_runoff_red += runoff_red
        total_peak_red += peak_red
        total_lag_hr += lag_add

        rows.append({
            "solution": row["name"],
            "family": row["family"],
            "coverage_pct": row["coverage_pct"],
            "runoff_reduction_pct": 100 * runoff_red,
            "peak_reduction_pct": 100 * peak_red,
            "lag_add_hr": lag_add,
        })

    total_runoff_red = min(total_runoff_red, 0.58)
    total_peak_red = min(total_peak_red, 0.72)
    total_lag_hr = min(total_lag_hr, 2.8)

    q_mod = q_base * (1 - total_runoff_red) * (1 - total_peak_red)
    t_mod = t + total_lag_hr

    runoff_base = np.trapezoid(q_base, t * 3600.0)
    runoff_mod = np.trapezoid(q_mod, t_mod * 3600.0)

    peak_base = float(np.max(q_base))
    peak_mod = float(np.max(q_mod))
    t_peak_base = float(t[np.argmax(q_base)])
    t_peak_mod = float(t_mod[np.argmax(q_mod)])

    c_eff_base = baseline["effective_runoff_coeff"]
    c_eff_mod = c_eff_base * (1 - total_runoff_red)

    return {
        "time_hr": t,
        "time_mod_hr": t_mod,
        "q_base_m3s": q_base,
        "q_mod_m3s": q_mod,
        "runoff_base_m3": runoff_base,
        "runoff_mod_m3": runoff_mod,
        "peak_base_m3s": peak_base,
        "peak_mod_m3s": peak_mod,
        "t_peak_base_hr": t_peak_base,
        "t_peak_mod_hr": t_peak_mod,
        "peak_reduction_pct": 100 * (1 - peak_mod / peak_base) if peak_base > 0 else 0.0,
        "runoff_reduction_pct": 100 * (1 - runoff_mod / runoff_base) if runoff_base > 0 else 0.0,
        "lag_increase_hr": t_peak_mod - t_peak_base,
        "effective_runoff_coeff_base": c_eff_base,
        "effective_runoff_coeff_mod": c_eff_mod,
        "details_df": pd.DataFrame(rows),
    }