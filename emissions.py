"""
emissions.py
=============
Module for scientifically documented carbon emission factor calculations.
Uses India-specific emission factors wherever reliable data is available (CEA, ARAI, CPCB, ICAR).
Includes explicit uncertainty bounds (min, max) to present confidence ranges.
"""

import os
from typing import Dict, Any, Tuple
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FACTORS_FILE = os.path.join(DATA_DIR, "emission_factors.csv")


_FACTORS_CACHE = None


def load_emission_factors() -> pd.DataFrame:
    """Load the emission factors database into a pandas DataFrame (cached in-memory)."""
    global _FACTORS_CACHE
    if _FACTORS_CACHE is not None:
        return _FACTORS_CACHE
    if not os.path.exists(FACTORS_FILE):
        raise FileNotFoundError(f"Emission factors file not found at: {FACTORS_FILE}")
    _FACTORS_CACHE = pd.read_csv(FACTORS_FILE)
    return _FACTORS_CACHE


def get_factor_info(sub_activity: str) -> Dict[str, Any]:
    """Retrieve details and bounds for a specific sub-activity."""
    df = load_emission_factors()
    match = df[df["sub_activity"] == sub_activity]
    if match.empty:
        raise ValueError(f"Unknown sub-activity factor: {sub_activity}")
    row = match.iloc[0]
    return {
        "sub_activity": row["sub_activity"],
        "category": row["category"],
        "factor_value": float(row["factor_value"]),
        "unit": row["unit"],
        "geographical_scope": row["geographical_scope"],
        "source": row["source"],
        "year": int(row["year"]),
        "uncertainty_min": float(row["uncertainty_min"]),
        "uncertainty_max": float(row["uncertainty_max"]),
        "notes": row["notes"],
    }


def calculate_transport_emissions(
    mode: str,
    distance_km: float,
    occupancy: int = 1,
) -> Dict[str, float]:
    """
    Calculate daily transportation emissions based on transport mode and distance.
    Occupancy adjusts vehicle emissions for carpooling (solo occupancy = 1).
    """
    occupancy = max(1, occupancy)
    distance_km = max(0.0, float(distance_km))

    mode_factor_map = {
        "petrol_two_wheeler": ("petrol_two_wheeler", False),
        "petrol_car": ("petrol_car_solo", True),
        "diesel_car": ("diesel_car_solo", True),
        "cng_car_auto": ("cng_car_auto", True),
        "ev_two_wheeler": ("ev_two_wheeler", False),
        "ev_car": ("ev_car_solo", True),
        "public_bus": ("public_bus", False),
        "metro_rail": ("metro_rail", False),
        "cycling_walking": ("cycling_walking", False),
    }

    if mode not in mode_factor_map:
        raise ValueError(f"Unsupported transport mode: {mode}")

    sub_act, is_vehicle_based = mode_factor_map[mode]
    info = get_factor_info(sub_act)

    divisor = occupancy if is_vehicle_based else 1.0

    val = (info["factor_value"] * distance_km) / divisor
    min_val = (info["uncertainty_min"] * distance_km) / divisor
    max_val = (info["uncertainty_max"] * distance_km) / divisor

    return {
        "emissions_kg": round(val, 4),
        "uncertainty_min": round(min_val, 4),
        "uncertainty_max": round(max_val, 4),
        "distance_km": distance_km,
        "mode": mode,
        "occupancy": occupancy,
        "factor_used": round(info["factor_value"] / divisor, 4),
    }


def calculate_electricity_emissions(
    electricity_kwh: float,
    solar_percentage: float = 0.0,
    household_size: int = 1,
    include_cooking_lpg: bool = True,
) -> Dict[str, float]:
    """
    Calculate electricity and cooking fuel emissions apportioned per person.
    Accounts for rooftop solar generation offset and household sharing.
    """
    household_size = max(1, household_size)
    electricity_kwh = max(0.0, float(electricity_kwh))
    solar_pct = min(100.0, max(0.0, float(solar_percentage))) / 100.0

    per_capita_kwh = electricity_kwh / household_size
    grid_kwh = per_capita_kwh * (1.0 - solar_pct)
    solar_kwh = per_capita_kwh * solar_pct

    grid_info = get_factor_info("grid_electricity_in")
    solar_info = get_factor_info("rooftop_solar")
    lpg_info = get_factor_info("lpg_cooking_per_capita") if include_cooking_lpg else None

    # Central estimate
    elec_val = (grid_kwh * grid_info["factor_value"]) + (solar_kwh * solar_info["factor_value"])
    lpg_val = lpg_info["factor_value"] if lpg_info else 0.0
    val = elec_val + lpg_val

    # Min bound
    elec_min = (grid_kwh * grid_info["uncertainty_min"]) + (solar_kwh * solar_info["uncertainty_min"])
    lpg_min = lpg_info["uncertainty_min"] if lpg_info else 0.0
    min_val = elec_min + lpg_min

    # Max bound
    elec_max = (grid_kwh * grid_info["uncertainty_max"]) + (solar_kwh * solar_info["uncertainty_max"])
    lpg_max = lpg_info["uncertainty_max"] if lpg_info else 0.0
    max_val = elec_max + lpg_max

    return {
        "emissions_kg": round(val, 4),
        "uncertainty_min": round(min_val, 4),
        "uncertainty_max": round(max_val, 4),
        "per_capita_kwh": round(per_capita_kwh, 2),
        "grid_kwh": round(grid_kwh, 2),
        "solar_kwh": round(solar_kwh, 2),
        "lpg_emissions_kg": round(lpg_val, 4),
    }


def calculate_food_emissions(
    diet_type: str,
    extra_dairy_servings: int = 0,
) -> Dict[str, float]:
    """
    Calculate daily dietary greenhouse gas emissions based on diet pattern
    and supplemental dairy servings (milk, paneer, yogurt).
    """
    diet_map = {
        "vegan": "diet_vegan",
        "vegetarian": "diet_vegetarian",
        "eggetarian": "diet_eggetarian",
        "nonveg_poultry": "diet_nonveg_poultry",
        "nonveg_heavy": "diet_nonveg_heavy",
    }
    if diet_type not in diet_map:
        raise ValueError(f"Unknown diet type: {diet_type}")

    diet_info = get_factor_info(diet_map[diet_type])
    dairy_info = get_factor_info("extra_dairy_serving")

    servings = max(0, int(extra_dairy_servings))

    val = diet_info["factor_value"] + (servings * dairy_info["factor_value"])
    min_val = diet_info["uncertainty_min"] + (servings * dairy_info["uncertainty_min"])
    max_val = diet_info["uncertainty_max"] + (servings * dairy_info["uncertainty_max"])

    return {
        "emissions_kg": round(val, 4),
        "uncertainty_min": round(min_val, 4),
        "uncertainty_max": round(max_val, 4),
        "diet_type": diet_type,
        "extra_dairy_servings": servings,
    }


def calculate_waste_emissions(
    waste_kg: float,
    compost_percentage: float = 0.0,
    recycle_percentage: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate daily waste emissions based on total waste generated (kg/day)
    and diversion rates (aerobic composting & dry recyclables sorting).
    """
    waste_kg = max(0.0, float(waste_kg))
    compost_pct = max(0.0, float(compost_percentage))
    recycle_pct = max(0.0, float(recycle_percentage))

    # Cap diversion at 100%
    total_diverted_pct = compost_pct + recycle_pct
    if total_diverted_pct > 100.0:
        compost_pct = (compost_pct / total_diverted_pct) * 100.0
        recycle_pct = (recycle_pct / total_diverted_pct) * 100.0

    landfill_pct = max(0.0, 100.0 - compost_pct - recycle_pct)

    landfill_kg = waste_kg * (landfill_pct / 100.0)
    compost_kg = waste_kg * (compost_pct / 100.0)
    recycle_kg = waste_kg * (recycle_pct / 100.0)

    landfill_info = get_factor_info("waste_unsegregated_landfill")
    compost_info = get_factor_info("waste_composted")
    recycle_info = get_factor_info("waste_recycled")

    val = (
        (landfill_kg * landfill_info["factor_value"])
        + (compost_kg * compost_info["factor_value"])
        + (recycle_kg * recycle_info["factor_value"])
    )

    min_val = (
        (landfill_kg * landfill_info["uncertainty_min"])
        + (compost_kg * compost_info["uncertainty_min"])
        + (recycle_kg * recycle_info["uncertainty_min"])
    )

    max_val = (
        (landfill_kg * landfill_info["uncertainty_max"])
        + (compost_kg * compost_info["uncertainty_max"])
        + (recycle_kg * recycle_info["uncertainty_max"])
    )

    return {
        "emissions_kg": round(val, 4),
        "uncertainty_min": round(min_val, 4),
        "uncertainty_max": round(max_val, 4),
        "landfill_kg": round(landfill_kg, 3),
        "compost_kg": round(compost_kg, 3),
        "recycle_kg": round(recycle_kg, 3),
    }


def calculate_total_emissions(
    transport_mode: str,
    distance_km: float,
    occupancy: int,
    electricity_kwh: float,
    solar_percentage: float,
    household_size: int,
    diet_type: str,
    extra_dairy_servings: int,
    waste_kg: float,
    compost_percentage: float,
    recycle_percentage: float,
    include_cooking_lpg: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive multi-activity daily carbon emission calculation.
    Returns total emissions, individual activity components, uncertainty ranges,
    and comparative benchmark metrics.
    """
    t_res = calculate_transport_emissions(transport_mode, distance_km, occupancy)
    e_res = calculate_electricity_emissions(electricity_kwh, solar_percentage, household_size, include_cooking_lpg)
    f_res = calculate_food_emissions(diet_type, extra_dairy_servings)
    w_res = calculate_waste_emissions(waste_kg, compost_percentage, recycle_percentage)

    total_kg = t_res["emissions_kg"] + e_res["emissions_kg"] + f_res["emissions_kg"] + w_res["emissions_kg"]
    min_total = t_res["uncertainty_min"] + e_res["uncertainty_min"] + f_res["uncertainty_min"] + w_res["uncertainty_min"]
    max_total = t_res["uncertainty_max"] + e_res["uncertainty_max"] + f_res["uncertainty_max"] + w_res["uncertainty_max"]

    # Activity percentages
    pct_t = round((t_res["emissions_kg"] / total_kg) * 100, 1) if total_kg > 0 else 0.0
    pct_e = round((e_res["emissions_kg"] / total_kg) * 100, 1) if total_kg > 0 else 0.0
    pct_f = round((f_res["emissions_kg"] / total_kg) * 100, 1) if total_kg > 0 else 0.0
    pct_w = round((w_res["emissions_kg"] / total_kg) * 100, 1) if total_kg > 0 else 0.0

    # India daily average per capita ~ 5.6 kg CO2e (~2.05 tonnes/year)
    # Paris Agreement target ~ 5.0 kg CO2e/day
    benchmark_india_kg = 5.60
    benchmark_paris_target_kg = 5.00

    annual_tonnes = (total_kg * 365.25) / 1000.0

    # Eco Rating
    if total_kg < 3.5:
        eco_rating = "Low Carbon (Eco-Champion)"
        eco_badge_color = "green"
    elif total_kg <= 6.5:
        eco_rating = "Moderate (Near National Average)"
        eco_badge_color = "orange"
    elif total_kg <= 10.0:
        eco_rating = "High Footprint"
        eco_badge_color = "darkorange"
    else:
        eco_rating = "Very High Footprint"
        eco_badge_color = "red"

    return {
        "total_kg": round(total_kg, 2),
        "uncertainty_min": round(min_total, 2),
        "uncertainty_max": round(max_total, 2),
        "annual_tonnes": round(annual_tonnes, 2),
        "breakdown": {
            "Transportation": round(t_res["emissions_kg"], 2),
            "Electricity & Cooking": round(e_res["emissions_kg"], 2),
            "Food & Diet": round(f_res["emissions_kg"], 2),
            "Waste Management": round(w_res["emissions_kg"], 2),
        },
        "percentages": {
            "Transportation": pct_t,
            "Electricity & Cooking": pct_e,
            "Food & Diet": pct_f,
            "Waste Management": pct_w,
        },
        "details": {
            "transport": t_res,
            "electricity": e_res,
            "food": f_res,
            "waste": w_res,
        },
        "benchmarks": {
            "india_average_daily_kg": benchmark_india_kg,
            "paris_target_daily_kg": benchmark_paris_target_kg,
            "diff_from_india_pct": round(((total_kg - benchmark_india_kg) / benchmark_india_kg) * 100, 1),
            "eco_rating": eco_rating,
            "eco_badge_color": eco_badge_color,
        },
    }
