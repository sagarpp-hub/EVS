"""
recommendations.py
==================
Module for generating personalized carbon reduction recommendations
and computing What-If scenario simulations.
"""

from typing import Dict, Any, List
from emissions import calculate_total_emissions


def generate_recommendations(profile_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate prioritized, personalized recommendations based on the user's highest
    emission categories and behavioral profile.
    """
    recs = []
    details = profile_result["details"]
    breakdown = profile_result["breakdown"]
    total_kg = profile_result["total_kg"]

    # Sort activities by emission contribution
    sorted_activities = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)

    # 1. Transportation Recommendations
    t_info = details["transport"]
    mode = t_info["mode"]
    dist = t_info["distance_km"]
    occupancy = t_info["occupancy"]
    t_kg = t_info["emissions_kg"]

    if dist > 0:
        if mode in ["petrol_car", "diesel_car"]:
            if occupancy == 1:
                # Carpooling suggestion
                carpool_daily_save = t_kg * 0.5  # Sharing with 1 extra person (occupancy 2)
                recs.append({
                    "category": "Transportation",
                    "priority": "High" if breakdown["Transportation"] > 3.0 else "Medium",
                    "title": "Adopt Carpooling or Ridesharing (2+ occupants)",
                    "description": f"Sharing your {dist} km daily drive reduces per-person fuel consumption by 50%.",
                    "daily_co2_saving_kg": round(carpool_daily_save, 2),
                    "annual_co2_saving_kg": round(carpool_daily_save * 300, 1), # 300 working days
                    "estimated_annual_cost_saving_inr": round(dist * 6.5 * 300 * 0.5, 0), # ~₹6.5/km fuel saving
                    "action_type": "carpool",
                })

            # Metro / Bus transit suggestion
            metro_kg = round(dist * 0.022, 2)
            metro_daily_save = max(0.0, t_kg - metro_kg)
            if dist >= 8:
                recs.append({
                    "category": "Transportation",
                    "priority": "High",
                    "title": "Switch Commute to Metro or Electric Urban Bus",
                    "description": f"Taking the Metro for your {dist} km commute instead of driving solo cuts transport emissions by ~85%.",
                    "daily_co2_saving_kg": round(metro_daily_save, 2),
                    "annual_co2_saving_kg": round(metro_daily_save * 300, 1),
                    "estimated_annual_cost_saving_inr": round(dist * 7.0 * 300 * 0.7, 0),
                    "action_type": "switch_public_transit",
                })

            # EV adoption suggestion
            ev_kg = round((dist * 0.107) / occupancy, 2)
            ev_daily_save = max(0.0, t_kg - ev_kg)
            recs.append({
                "category": "Transportation",
                "priority": "Medium",
                "title": "Transition to an Electric Vehicle (EV)",
                "description": f"An electric car charging on the Indian grid emits ~0.107 kg CO2e/km vs ~0.165-0.185 kg for ICE vehicles, saving operational costs.",
                "daily_co2_saving_kg": round(ev_daily_save, 2),
                "annual_co2_saving_kg": round(ev_daily_save * 365, 1),
                "estimated_annual_cost_saving_inr": round(dist * 5.0 * 365, 0),
                "action_type": "transition_ev",
            })

        elif mode == "petrol_two_wheeler" and dist >= 10:
            ev_2w_kg = dist * 0.021
            save_2w = max(0.0, t_kg - ev_2w_kg)
            recs.append({
                "category": "Transportation",
                "priority": "Medium",
                "title": "Switch to an Electric Scooter / 2-Wheeler",
                "description": f"Switching your {dist} km daily ride from petrol scooter (45 km/l) to an EV 2-wheeler cuts emissions by over 50% and lowers running cost to ~₹0.30/km.",
                "daily_co2_saving_kg": round(save_2w, 2),
                "annual_co2_saving_kg": round(save_2w * 365, 1),
                "estimated_annual_cost_saving_inr": round(dist * 1.8 * 365, 0),
                "action_type": "switch_ev_2w",
            })

        if dist <= 5 and mode not in ["cycling_walking"]:
            recs.append({
                "category": "Transportation",
                "priority": "Medium",
                "title": "Adopt Active Mobility (Cycling / Walking for short trips)",
                "description": f"Your trip is {dist} km. Replacing motorized trips under 5 km with walking or cycling produces zero emissions and improves cardiovascular health.",
                "daily_co2_saving_kg": round(t_kg, 2),
                "annual_co2_saving_kg": round(t_kg * 300, 1),
                "estimated_annual_cost_saving_inr": round(dist * 4.0 * 300, 0),
                "action_type": "active_mobility",
            })

    # 2. Electricity & Energy Recommendations
    e_info = details["electricity"]
    per_capita_kwh = e_info["per_capita_kwh"]
    solar_kwh = e_info["solar_kwh"]
    grid_kwh = e_info["grid_kwh"]

    if per_capita_kwh > 2.5:
        # Thermostat & appliance efficiency
        ac_opt_save_kwh = round(per_capita_kwh * 0.20, 2)  # ~20% savings by 24°C setting & star rated appliances
        ac_opt_save_co2 = round(ac_opt_save_kwh * 0.82, 2)
        recs.append({
            "category": "Electricity & Energy",
            "priority": "High" if per_capita_kwh > 5.0 else "Medium",
            "title": "Optimize Air Conditioning (Set to 24°C - 26°C) & Star Ratings",
            "description": f"Increasing AC setpoint from 18°C/20°C to 24°C reduces cooling power consumption by 6% per degree (saving ~{ac_opt_save_kwh} kWh/day).",
            "daily_co2_saving_kg": ac_opt_save_co2,
            "annual_co2_saving_kg": round(ac_opt_save_co2 * 365, 1),
            "estimated_annual_cost_saving_inr": round(ac_opt_save_kwh * 8.5 * 365, 0), # ~₹8.5/kWh tariff
            "action_type": "ac_thermostat",
        })

    if solar_kwh == 0 and grid_kwh > 1.5:
        # Rooftop solar
        potential_solar_pct = 0.60
        solar_gen_kwh = round(grid_kwh * potential_solar_pct, 2)
        solar_save_co2 = round(solar_gen_kwh * (0.82 - 0.04), 2)
        recs.append({
            "category": "Electricity & Energy",
            "priority": "Medium",
            "title": "Install Rooftop Solar Photovoltaic (PM Surya Ghar Scheme)",
            "description": f"Offsetting 60% of your daily grid electricity with rooftop solar cuts coal-based grid reliance under India's subsidized rooftop solar mission.",
            "daily_co2_saving_kg": solar_save_co2,
            "annual_co2_saving_kg": round(solar_save_co2 * 365, 1),
            "estimated_annual_cost_saving_inr": round(solar_gen_kwh * 7.5 * 365, 0),
            "action_type": "install_solar",
        })

    # 3. Dietary Recommendations
    f_info = details["food"]
    diet = f_info["diet_type"]
    dairy_servings = f_info["extra_dairy_servings"]

    if diet == "nonveg_heavy":
        # Switch to flexitarian / poultry
        diet_save_kg = round(6.70 - 4.80, 2)
        recs.append({
            "category": "Food & Diet",
            "priority": "High",
            "title": "Shift Towards Flexitarian / Poultry-Focused Diet",
            "description": "Ruminant meat (mutton/beef) has 4x higher carbon intensity than poultry and pulses. Replacing red meat with plant-rich meals significantly reduces enteric methane.",
            "daily_co2_saving_kg": diet_save_kg,
            "annual_co2_saving_kg": round(diet_save_kg * 365, 1),
            "estimated_annual_cost_saving_inr": round(250 * 52, 0), # saving on expensive meat
            "action_type": "flexitarian_diet",
        })

    if diet in ["nonveg_poultry", "nonveg_heavy"]:
        # Meatless days
        meatless_save_kg = round(1.50, 2)
        recs.append({
            "category": "Food & Diet",
            "priority": "Medium",
            "title": "Adopt 'Meatless Days' (2 Plant-Based Days per Week)",
            "description": "Enjoying traditional Indian dal, paneer, and vegetable dishes 2 days per week saves ~1.5 kg CO2e per plant-based day.",
            "daily_co2_saving_kg": round(meatless_save_kg * (2 / 7), 2),
            "annual_co2_saving_kg": round(meatless_save_kg * 104, 1),
            "estimated_annual_cost_saving_inr": round(150 * 104, 0),
            "action_type": "meatless_days",
        })

    if dairy_servings >= 3:
        dairy_save_kg = round(0.45 * (dairy_servings - 1), 2)
        recs.append({
            "category": "Food & Diet",
            "priority": "Low",
            "title": "Moderate High Dairy Intake / Substitute with Plant-Based Milks",
            "description": "Reducing high excess dairy portions (milk, cheese, butter) lowers livestock methane footprint while maintaining optimal protein via pulses and nuts.",
            "daily_co2_saving_kg": dairy_save_kg,
            "annual_co2_saving_kg": round(dairy_save_kg * 365, 1),
            "estimated_annual_cost_saving_inr": round(25 * 365, 0),
            "action_type": "moderate_dairy",
        })

    # 4. Waste Management Recommendations
    w_info = details["waste"]
    landfill_kg = w_info["landfill_kg"]

    if landfill_kg > 0.2:
        # Home Composting
        compost_save_kg = round(landfill_kg * 0.70 * (0.95 - 0.12), 2)
        recs.append({
            "category": "Waste Management",
            "priority": "Medium" if landfill_kg > 0.4 else "Low",
            "title": "Segregate Wet Waste & Initiate Home/Community Composting",
            "description": f"Diverting 70% of kitchen organic wet waste from landfill avoids anaerobic methane generation, yielding nutrient-rich organic compost.",
            "daily_co2_saving_kg": compost_save_kg,
            "annual_co2_saving_kg": round(compost_save_kg * 365, 1),
            "estimated_annual_cost_saving_inr": 800.0, # saved fertilizer / compost cost
            "action_type": "home_composting",
        })

    if w_info["recycle_kg"] < 0.1 and landfill_kg > 0.1:
        # Recycling dry waste
        recycle_save_kg = round(landfill_kg * 0.25 * (0.95 - 0.05), 2)
        recs.append({
            "category": "Waste Management",
            "priority": "Low",
            "title": "Segregate Dry Recyclables for Local Recycling Channels",
            "description": "Rinsing and segregating plastics, cardboard, and metal containers for doorstep collection prevents open burning and reduces landfill burden.",
            "daily_co2_saving_kg": recycle_save_kg,
            "annual_co2_saving_kg": round(recycle_save_kg * 365, 1),
            "estimated_annual_cost_saving_inr": 1200.0, # scrap sales
            "action_type": "dry_recycling",
        })

    # Sort recommendations by daily CO2 savings descending
    recs.sort(key=lambda r: r["daily_co2_saving_kg"], reverse=True)
    return recs


def simulate_what_if(
    baseline_inputs: Dict[str, Any],
    modifications: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute 'What-If' simulation by comparing baseline emissions against a modified scenario.
    """
    # Baseline
    base_res = calculate_total_emissions(**baseline_inputs)

    # Modified scenario
    scenario_inputs = dict(baseline_inputs)
    scenario_inputs.update(modifications)
    scenario_res = calculate_total_emissions(**scenario_inputs)

    base_total = base_res["total_kg"]
    new_total = scenario_res["total_kg"]
    daily_delta_kg = round(base_total - new_total, 2)
    pct_reduction = round((daily_delta_kg / base_total) * 100, 1) if base_total > 0 else 0.0
    annual_saved_tonnes = round((daily_delta_kg * 365.25) / 1000.0, 2)

    # Equivalent trees planted: An average mature urban tree absorbs ~22 kg CO2 per year
    trees_equivalent = round((daily_delta_kg * 365.25) / 22.0, 1)

    return {
        "baseline_total_kg": base_total,
        "scenario_total_kg": new_total,
        "daily_saved_kg": daily_delta_kg,
        "pct_reduction": pct_reduction,
        "annual_saved_tonnes": annual_saved_tonnes,
        "trees_equivalent": max(0.0, trees_equivalent),
        "baseline_breakdown": base_res["breakdown"],
        "scenario_breakdown": scenario_res["breakdown"],
        "scenario_result": scenario_res,
    }
