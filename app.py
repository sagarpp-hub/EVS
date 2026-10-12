"""
app.py
======
Machine Learning-Based Personalized Carbon Emission Estimation
and Reduction Recommendation System for Daily Activities.

Enhanced, Modernized Streamlit Web Application Interface.
"""

import os
from datetime import date
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import emissions
import recommendations
import model

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CarbonPulse | Personalized Carbon Footprint AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Modern CSS Design System
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%);
        border-radius: 20px;
        padding: 30px 35px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -5px rgba(6, 78, 59, 0.25);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-top: 8px;
        margin-bottom: 15px;
        max-width: 850px;
        line-height: 1.5;
    }
    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .hero-tag {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 4px 12px;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* Section Cards */
    .input-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .input-card:hover {
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.07);
    }
    .card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid #f1f5f9;
    }
    .card-step-badge {
        background: #ecfdf5;
        color: #047857;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }

    /* KPI Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin-bottom: 25px;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #10b981, #06b6d4);
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Recommendation Cards */
    .recom-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        border-left: 5px solid #10b981;
    }
    .recom-priority-high {
        border-left-color: #ef4444;
    }
    .recom-priority-medium {
        border-left-color: #f59e0b;
    }
    .recom-priority-low {
        border-left-color: #10b981;
    }
    .badge-prio {
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-prio-high {
        background: #fee2e2;
        color: #b91c1c;
    }
    .badge-prio-medium {
        background: #fef3c7;
        color: #b45309;
    }
    .badge-prio-low {
        background: #ecfdf5;
        color: #047857;
    }

    /* Pill Badges */
    .pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .pill-green { background: #dcfce7; color: #15803d; }
    .pill-amber { background: #fef3c7; color: #b45309; }
    .pill-red { background: #fee2e2; color: #b91c1c; }
    .pill-blue { background: #e0f2fe; color: #0369a1; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization for User History
# ---------------------------------------------------------
HISTORY_FILE = os.path.join(emissions.DATA_DIR, "user_history.csv")
SAMPLE_HISTORY_FILE = os.path.join(emissions.DATA_DIR, "sample_user_history.csv")

if "history_df" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        st.session_state.history_df = pd.read_csv(HISTORY_FILE)
    elif os.path.exists(SAMPLE_HISTORY_FILE):
        st.session_state.history_df = pd.read_csv(SAMPLE_HISTORY_FILE)
    else:
        st.session_state.history_df = pd.DataFrame()


def save_history(df: pd.DataFrame):
    """Save history dataframe to CSV and update session state."""
    st.session_state.history_df = df
    df.to_csv(HISTORY_FILE, index=False)


# ---------------------------------------------------------
# Sidebar Controls & Lifestyle Presets
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Profile & Settings")
    st.caption("Select a lifestyle archetype or calibrate custom values.")
    
    preset_choice = st.selectbox(
        "⚡ Quick Lifestyle Archetype",
        [
            "Custom Inputs",
            "Urban Car Commuter (High Footprint)",
            "Student / Metro Commuter (Low Footprint)",
            "Typical Indian Household Member",
            "Eco-Conscious Citizen (Solar & Composting)",
        ],
        index=0,
    )

    # Preset definitions
    presets = {
        "Urban Car Commuter (High Footprint)": {
            "transport_mode": "petrol_car", "distance_km": 28.0, "occupancy": 1,
            "electricity_kwh": 14.0, "solar_percentage": 0.0, "household_size": 3,
            "diet_type": "nonveg_poultry", "extra_dairy_servings": 2,
            "waste_kg": 0.70, "compost_percentage": 0.0, "recycle_percentage": 10.0,
        },
        "Student / Metro Commuter (Low Footprint)": {
            "transport_mode": "metro_rail", "distance_km": 16.0, "occupancy": 1,
            "electricity_kwh": 5.0, "solar_percentage": 0.0, "household_size": 4,
            "diet_type": "vegetarian", "extra_dairy_servings": 1,
            "waste_kg": 0.35, "compost_percentage": 40.0, "recycle_percentage": 30.0,
        },
        "Typical Indian Household Member": {
            "transport_mode": "petrol_two_wheeler", "distance_km": 15.0, "occupancy": 1,
            "electricity_kwh": 9.0, "solar_percentage": 0.0, "household_size": 4,
            "diet_type": "vegetarian", "extra_dairy_servings": 1,
            "waste_kg": 0.50, "compost_percentage": 10.0, "recycle_percentage": 20.0,
        },
        "Eco-Conscious Citizen (Solar & Composting)": {
            "transport_mode": "cycling_walking", "distance_km": 4.0, "occupancy": 1,
            "electricity_kwh": 6.0, "solar_percentage": 65.0, "household_size": 3,
            "diet_type": "vegan", "extra_dairy_servings": 0,
            "waste_kg": 0.30, "compost_percentage": 75.0, "recycle_percentage": 25.0,
        },
    }
    selected_preset = presets.get(preset_choice, None)

    st.markdown("---")
    st.markdown("#### 🔬 Indian Scientific Baselines")
    st.markdown("""
    - **Grid Factor**: `0.82 kg CO₂e/kWh`  
      *(CEA Ver 19, with ~18% T&D loss)*
    - **Mobility**: `ARAI / MoRTH Standards`  
      *(Petrol 2W: 0.043 kg/km, Car: 0.165 kg/km)*
    - **Waste**: `CPCB MSW Model`  
      *(Landfill CH₄: 0.95 kg/kg)*
    - **Diet**: `ICAR LCA Dietary Studies`  
      *(Vegetarian: 3.2 kg/day, Poultry: 4.8 kg)*
    """)

    st.markdown("---")
    st.caption("CarbonPulse AI v2.0 | Built with Python, Scikit-learn & Streamlit")

# ---------------------------------------------------------
# Main Page Hero Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">🌱 CarbonPulse AI</h1>
    <p class="hero-subtitle">
        Personalized Carbon Emission Estimation and Reduction Intelligence for Daily Activities.
        Bridging scientific Life Cycle Assessment with predictive Machine Learning calibrated for India.
    </p>
    <div class="hero-tags">
        <span class="hero-tag">🇮🇳 CEA Grid Factor 0.82 kg/kWh</span>
        <span class="hero-tag">🤖 ML Ensemble (GBR R² = 0.9455)</span>
        <span class="hero-tag">🎯 95% Confidence Bounds</span>
        <span class="hero-tag">💡 What-If Scenario Engine</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Mode mappings
transport_labels = {
    "petrol_two_wheeler": "🛵 Petrol 2-Wheeler (Motorcycle / Scooter)",
    "petrol_car": "🚗 Petrol 4-Wheeler (Car / Cab)",
    "diesel_car": "🚙 Diesel 4-Wheeler (Car / SUV)",
    "cng_car_auto": "🛺 CNG Auto Rickshaw / Car",
    "ev_two_wheeler": "⚡ Electric 2-Wheeler (EV Scooter)",
    "ev_car": "🔋 Electric 4-Wheeler (EV Car)",
    "public_bus": "🚌 Public City Bus",
    "metro_rail": "🚇 Metro Rail / Suburban Train",
    "cycling_walking": "🚶 Active Mobility (Walking / Cycling)",
}
inv_transport_labels = {v: k for k, v in transport_labels.items()}

diet_labels = {
    "vegan": "🥗 Vegan (Plant-based, 0% animal products)",
    "vegetarian": "🍛 Lacto-Vegetarian (Dairy, pulses, vegetables)",
    "eggetarian": "🍳 Eggetarian (Vegetarian + eggs)",
    "nonveg_poultry": "🍗 Non-Vegetarian (Poultry/Fish 2-4x/week)",
    "nonveg_heavy": "🥩 Heavy Non-Vegetarian (Frequent mutton/red meat)",
}
inv_diet_labels = {v: k for k, v in diet_labels.items()}

# ---------------------------------------------------------
# Top Main Navigation Tabs
# ---------------------------------------------------------
tab_calc, tab_recs, tab_hist, tab_ml, tab_report = st.tabs([
    "📊 Daily Carbon Estimator",
    "💡 Reduction Intelligence & What-If",
    "📈 Longitudinal Tracking",
    "🔬 ML Models & Methodology",
    "📄 Export Audit Report",
])


# ==============================================================================
# TAB 1: DAILY CARBON ESTIMATOR
# ==============================================================================
with tab_calc:
    st.markdown("### 1. Daily Lifestyle Activity Inputs")
    st.caption("Fill in your activities for today or choose an archetype preset from the sidebar.")

    c_left, c_right = st.columns(2)

    with c_left:
        # Card 1: Transportation
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <span class="card-step-badge">Step 1</span>
                <h3 class="card-title">🚗 Mobility & Commute</h3>
            </div>
        """, unsafe_allow_html=True)
        
        default_mode_key = selected_preset["transport_mode"] if selected_preset else "petrol_car"
        mode_label = st.selectbox(
            "Primary Transport Mode",
            list(transport_labels.values()),
            index=list(transport_labels.keys()).index(default_mode_key),
            key="calc_mode"
        )
        inp_transport_mode = inv_transport_labels[mode_label]

        default_dist = float(selected_preset["distance_km"]) if selected_preset else 20.0
        inp_distance_km = st.number_input(
            "Daily Distance Travelled (km)",
            min_value=0.0, max_value=250.0, value=default_dist, step=1.0,
            key="calc_dist"
        )

        if inp_transport_mode in ["petrol_car", "diesel_car", "cng_car_auto", "ev_car"]:
            default_occ = int(selected_preset["occupancy"]) if selected_preset else 1
            inp_occupancy = st.slider(
                "Vehicle Occupancy (Car sharing/carpool)",
                min_value=1, max_value=5, value=default_occ,
                help="Solo drive = 1. Carpooling with 2+ members divides vehicle emissions.",
                key="calc_occ"
            )
        else:
            inp_occupancy = 1

        st.markdown("</div>", unsafe_allow_html=True)

        # Card 2: Electricity & Energy
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <span class="card-step-badge">Step 2</span>
                <h3 class="card-title">⚡ Electricity & Energy</h3>
            </div>
        """, unsafe_allow_html=True)

        default_elec = float(selected_preset["electricity_kwh"]) if selected_preset else 10.0
        inp_electricity_kwh = st.number_input(
            "Household Electricity Consumed (kWh/day)",
            min_value=0.0, max_value=80.0, value=default_elec, step=0.5,
            help="Divide your monthly electricity bill units (kWh) by 30.",
            key="calc_elec"
        )

        c_h, c_s = st.columns(2)
        with c_h:
            default_hh = int(selected_preset["household_size"]) if selected_preset else 3
            inp_household_size = st.number_input(
                "Household Members", min_value=1, max_value=12, value=default_hh,
                help="Apportions shared household electricity & cooking per capita.",
                key="calc_hh"
            )
        with c_s:
            default_solar = float(selected_preset["solar_percentage"]) if selected_preset else 0.0
            inp_solar_percentage = st.slider(
                "Rooftop Solar Share (%)", min_value=0.0, max_value=100.0, value=default_solar, step=5.0,
                key="calc_solar"
            )

        inp_lpg = st.checkbox("Include Cooking LPG Gas (~0.55 kg CO₂e/person-day)", value=True, key="calc_lpg")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        # Card 3: Food & Diet
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <span class="card-step-badge">Step 3</span>
                <h3 class="card-title">🥗 Food & Nutrition</h3>
            </div>
        """, unsafe_allow_html=True)

        default_diet_key = selected_preset["diet_type"] if selected_preset else "vegetarian"
        diet_label = st.selectbox(
            "Dietary Habit",
            list(diet_labels.values()),
            index=list(diet_labels.keys()).index(default_diet_key),
            key="calc_diet"
        )
        inp_diet_type = inv_diet_labels[diet_label]

        default_dairy = int(selected_preset["extra_dairy_servings"]) if selected_preset else 1
        inp_extra_dairy = st.slider(
            "Extra Dairy Servings (glasses of milk / curd / 50g paneer)",
            min_value=0, max_value=6, value=default_dairy,
            help="Indian dairy lifecycle factor: 0.45 kg CO₂e per serving.",
            key="calc_dairy"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Card 4: Solid Waste Management
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <span class="card-step-badge">Step 4</span>
                <h3 class="card-title">🗑️ Waste & Circularity</h3>
            </div>
        """, unsafe_allow_html=True)

        default_waste = float(selected_preset["waste_kg"]) if selected_preset else 0.50
        inp_waste_kg = st.number_input(
            "Total Daily Waste Generated (kg/day)",
            min_value=0.0, max_value=5.0, value=default_waste, step=0.05,
            help="Typical Indian urban average: 0.35 - 0.65 kg per capita per day.",
            key="calc_waste"
        )

        c_w1, c_w2 = st.columns(2)
        with c_w1:
            default_comp = float(selected_preset["compost_percentage"]) if selected_preset else 0.0
            inp_compost_percentage = st.slider("Composted Wet Waste (%)", min_value=0.0, max_value=100.0, value=default_comp, step=5.0, key="calc_comp")
        with c_w2:
            default_rec = float(selected_preset["recycle_percentage"]) if selected_preset else 10.0
            inp_recycle_percentage = st.slider("Recycled Dry Waste (%)", min_value=0.0, max_value=100.0, value=default_rec, step=5.0, key="calc_rec")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Execute Dual Estimation Engine
    # ---------------------------------------------------------
    user_inputs = {
        "transport_mode": inp_transport_mode,
        "distance_km": inp_distance_km,
        "occupancy": inp_occupancy,
        "electricity_kwh": inp_electricity_kwh,
        "solar_percentage": inp_solar_percentage,
        "household_size": inp_household_size,
        "diet_type": inp_diet_type,
        "extra_dairy_servings": inp_extra_dairy,
        "waste_kg": inp_waste_kg,
        "compost_percentage": inp_compost_percentage,
        "recycle_percentage": inp_recycle_percentage,
        "include_cooking_lpg": inp_lpg,
    }

    results = emissions.calculate_total_emissions(**user_inputs)
    ml_results = model.predict_emissions(user_inputs)

    st.markdown("---")
    st.markdown("### 2. Carbon Footprint Intelligence Dashboard")

    # KPI Row
    diff_india = results['benchmarks']['diff_from_india_pct']
    diff_sign = "+" if diff_india > 0 else ""
    diff_color = "pill-red" if diff_india > 20 else ("pill-amber" if diff_india > 0 else "pill-green")

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Daily Footprint</div>
            <div class="kpi-value">{results['total_kg']} <span style="font-size: 1.1rem; color: #64748b;">kg CO₂e</span></div>
            <div class="kpi-subtext">
                <span class="pill pill-blue">Range: [{results['uncertainty_min']} – {results['uncertainty_max']}] kg</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Annual Projection</div>
            <div class="kpi-value">{results['annual_tonnes']} <span style="font-size: 1.1rem; color: #64748b;">Tonnes</span></div>
            <div class="kpi-subtext">
                <span class="pill pill-green">Status: {results['benchmarks']['eco_rating']}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">vs. India Average (5.6 kg)</div>
            <div class="kpi-value">{diff_sign}{diff_india:.1f}%</div>
            <div class="kpi-subtext">
                <span class="pill {diff_color}">Benchmark: 5.60 kg CO₂e/day</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">ML Ensemble Prediction</div>
            <div class="kpi-value">{ml_results['primary_prediction_kg']} <span style="font-size: 1.1rem; color: #64748b;">kg</span></div>
            <div class="kpi-subtext">
                <span class="pill pill-blue">95% Conf: [{ml_results['ml_interval_lower_kg']} – {ml_results['ml_interval_upper_kg']}] kg</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Visualizations Row: Gauge Meter + Donut + Bar with Error Bars
    col_gauge, col_donut, col_bar = st.columns([1, 1, 1.2])

    with col_gauge:
        # Plotly Speedometer Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=results['total_kg'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "<b>National Benchmark Dial</b>", 'font': {'size': 16, 'color': '#0f172a'}},
            gauge={
                'axis': {'range': [0, 16], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': "#0f172a", 'thickness': 0.25},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#cbd5e1",
                'steps': [
                    {'range': [0, 4.0], 'color': '#dcfce7'},    # Green / Eco
                    {'range': [4.0, 6.5], 'color': '#fef3c7'},  # Yellow / India Avg
                    {'range': [6.5, 16.0], 'color': '#fee2e2'}, # Red / High
                ],
                'threshold': {
                    'line': {'color': "#dc2626", 'width': 3},
                    'thickness': 0.8,
                    'value': 5.60,
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_donut:
        # Donut Chart
        breakdown_df = pd.DataFrame([
            {"Activity": k, "Emissions (kg CO₂e)": v, "Percentage": results["percentages"][k]}
            for k, v in results["breakdown"].items()
        ])
        fig_donut = px.pie(
            breakdown_df,
            values="Emissions (kg CO₂e)",
            names="Activity",
            hole=0.55,
            color="Activity",
            color_discrete_map={
                "Transportation": "#ef4444",
                "Electricity & Cooking": "#f59e0b",
                "Food & Diet": "#10b981",
                "Waste Management": "#06b6d4",
            },
            title="<b>Activity Distribution</b>",
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent')
        fig_donut.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), showlegend=True, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_bar:
        # Bar Chart with Uncertainty Whiskers
        categories = list(results["breakdown"].keys())
        cat_values = [results["breakdown"][c] for c in categories]
        det = results["details"]
        min_errs = [
            cat_values[0] - det["transport"]["uncertainty_min"],
            cat_values[1] - det["electricity"]["uncertainty_min"],
            cat_values[2] - det["food"]["uncertainty_min"],
            cat_values[3] - det["waste"]["uncertainty_min"],
        ]
        max_errs = [
            det["transport"]["uncertainty_max"] - cat_values[0],
            det["electricity"]["uncertainty_max"] - cat_values[1],
            det["food"]["uncertainty_max"] - cat_values[2],
            det["waste"]["uncertainty_max"] - cat_values[3],
        ]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=categories,
            x=cat_values,
            orientation='h',
            marker=dict(color=['#ef4444', '#f59e0b', '#10b981', '#06b6d4']),
            error_x=dict(
                type='data', symmetric=False,
                array=max_errs, arrayminus=min_errs,
                color='#334155', thickness=1.5, width=4
            ),
            text=[f"{v:.2f} kg" for v in cat_values],
            textposition='auto',
        ))
        fig_bar.update_layout(
            title="<b>Category Emissions with ± Bounds</b>",
            xaxis_title="kg CO₂e / day",
            height=280,
            margin=dict(l=10, r=10, t=40, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Top Emission Source Banner
    top_source = max(results["breakdown"].items(), key=lambda x: x[1])
    st.info(
        f"🎯 **Primary Emission Driver:** **{top_source[0]}** contributes **{top_source[1]} kg CO₂e/day** "
        f"({results['percentages'][top_source[0]]}% of total). Check the **'Reduction Intelligence'** tab for high-impact interventions."
    )

    # One-Click Log to History
    st.markdown("---")
    c_log1, c_log2 = st.columns([1, 2])
    with c_log1:
        log_date = st.date_input("Audit Record Date", value=date.today(), key="calc_log_date")
        if st.button("💾 Save to Personal History Log", type="primary", use_container_width=True):
            new_entry = {
                "date": str(log_date),
                "transport_mode": inp_transport_mode,
                "distance_km": inp_distance_km,
                "occupancy": inp_occupancy,
                "electricity_kwh": inp_electricity_kwh,
                "solar_percentage": inp_solar_percentage,
                "household_size": inp_household_size,
                "diet_type": inp_diet_type,
                "extra_dairy_servings": inp_extra_dairy,
                "waste_kg": inp_waste_kg,
                "compost_percentage": inp_compost_percentage,
                "recycle_percentage": inp_recycle_percentage,
                "total_emissions_kg": results["total_kg"],
                "transport_kg": results["breakdown"]["Transportation"],
                "electricity_kg": results["breakdown"]["Electricity & Cooking"],
                "food_kg": results["breakdown"]["Food & Diet"],
                "waste_emissions_kg": results["breakdown"]["Waste Management"],
                "ml_predicted_kg": ml_results["primary_prediction_kg"],
            }
            updated_df = pd.concat([st.session_state.history_df, pd.DataFrame([new_entry])], ignore_index=True)
            save_history(updated_df)
            st.success(f"✅ Saved record for {log_date} ({results['total_kg']} kg CO₂e)!")
            st.rerun()
    with c_log2:
        st.markdown(f"**Tracked History:** Currently storing **{len(st.session_state.history_df)} days** in your personal log.")
        if not st.session_state.history_df.empty:
            avg_hist = st.session_state.history_df["total_emissions_kg"].mean()
            st.caption(f"Historical 14-day average: **{avg_hist:.2f} kg CO₂e/day**")


# ==============================================================================
# TAB 2: REDUCTION INTELLIGENCE & WHAT-IF SIMULATOR
# ==============================================================================
with tab_recs:
    st.markdown("### 💡 Prioritized Carbon Reduction Intelligence")
    st.caption("Quantified action recommendations based on your highest emission categories, ranked by carbon and financial savings.")

    # Base profile for recs
    base_inputs_recs = user_inputs
    base_res = emissions.calculate_total_emissions(**base_inputs_recs)
    recs = recommendations.generate_recommendations(base_res)

    for idx, r in enumerate(recs, 1):
        prio = r["priority"].lower()
        st.markdown(f"""
        <div class="recom-card recom-priority-{prio}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <h4 style="margin: 0; font-size: 1.1rem; color: #0f172a;">{idx}. {r['title']}</h4>
                <span class="badge-prio badge-prio-{prio}">{r['priority']} Priority</span>
            </div>
            <p style="color: #475569; font-size: 0.92rem; margin-bottom: 12px;">{r['description']}</p>
            <div style="display: flex; flex-wrap: wrap; gap: 12px;">
                <span class="pill pill-green">📉 Daily Saving: -{r['daily_co2_saving_kg']} kg CO₂e</span>
                <span class="pill pill-blue">🌿 Annual Saving: -{r['annual_co2_saving_kg']} kg CO₂e</span>
                <span class="pill pill-amber">💰 Est. Financial Savings: ₹{r['estimated_annual_cost_saving_inr']:,.0f}/yr</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Interactive 'What-If' Scenario Simulator")
    st.caption("Simulate lifestyle interventions and see instantaneous carbon reductions and tree equivalencies.")

    s_col1, s_col2 = st.columns([1, 1.1])

    with s_col1:
        st.markdown("##### Adjust Hypothetical Interventions")
        sim_mode_label = st.selectbox(
            "Shift Transport Mode:",
            list(transport_labels.values()),
            index=list(transport_labels.keys()).index("metro_rail" if user_inputs["transport_mode"] != "metro_rail" else "cycling_walking"),
            key="sim_mode"
        )
        sim_mode = inv_transport_labels[sim_mode_label]

        sim_occupancy = st.slider(
            "Increase Carpool Occupancy:",
            min_value=1, max_value=5, value=2 if user_inputs["occupancy"] == 1 else user_inputs["occupancy"],
            key="sim_occ"
        )

        sim_solar = st.slider(
            "Install Rooftop Solar (% Share):",
            min_value=0.0, max_value=100.0, value=60.0 if user_inputs["solar_percentage"] == 0 else user_inputs["solar_percentage"],
            key="sim_solar"
        )

        sim_elec_cut = st.slider(
            "Cut Daily Electricity by (kWh via 24°C AC / LED):",
            min_value=0.0, max_value=10.0, value=2.0, step=0.5,
            key="sim_elec_cut"
        )

        sim_diet_label = st.selectbox(
            "Adopt Dietary Shift:",
            list(diet_labels.values()),
            index=list(diet_labels.keys()).index("vegetarian" if user_inputs["diet_type"] in ["nonveg_heavy", "nonveg_poultry"] else "vegan"),
            key="sim_diet"
        )
        sim_diet = inv_diet_labels[sim_diet_label]

        sim_compost = st.slider(
            "Compost Kitchen Wet Waste (%):",
            min_value=0.0, max_value=100.0, value=75.0,
            key="sim_compost"
        )

    # Compute What-If
    modifications = {
        "transport_mode": sim_mode,
        "occupancy": sim_occupancy,
        "solar_percentage": sim_solar,
        "electricity_kwh": max(0.5, user_inputs["electricity_kwh"] - sim_elec_cut),
        "diet_type": sim_diet,
        "compost_percentage": sim_compost,
    }
    what_if_res = recommendations.simulate_what_if(user_inputs, modifications)

    with s_col2:
        st.markdown("##### Simulated Impact & Equivalencies")

        m_w1, m_w2 = st.columns(2)
        with m_w1:
            st.metric(
                label="Simulated Daily Footprint",
                value=f"{what_if_res['scenario_total_kg']} kg CO₂e",
                delta=f"-{what_if_res['daily_saved_kg']} kg ({what_if_res['pct_reduction']}%)",
                delta_color="inverse"
            )
        with m_w2:
            st.metric(
                label="Avoided Carbon / Year",
                value=f"{what_if_res['annual_saved_tonnes']} Tonnes",
                delta=f"🌳 {what_if_res['trees_equivalent']} Trees Equiv",
                delta_color="normal"
            )

        st.success(
            f"🌳 **Environmental Equivalency:** Adopting these changes prevents **{what_if_res['annual_saved_tonnes']} tonnes** "
            f"of GHG emissions annually, equivalent to the annual carbon absorption of **{what_if_res['trees_equivalent']} mature urban trees**!"
        )

        # Before & After Grouped Bar
        comp_df = pd.DataFrame([
            {"Activity": k, "Scenario": "Current Baseline", "Emissions (kg)": what_if_res["baseline_breakdown"][k]}
            for k in what_if_res["baseline_breakdown"]
        ] + [
            {"Activity": k, "Scenario": "What-If Simulated", "Emissions (kg)": what_if_res["scenario_breakdown"][k]}
            for k in what_if_res["scenario_breakdown"]
        ])
        fig_whatif = px.bar(
            comp_df,
            x="Activity",
            y="Emissions (kg)",
            color="Scenario",
            barmode="group",
            color_discrete_map={"Current Baseline": "#ef4444", "What-If Simulated": "#10b981"},
            title="<b>Baseline vs. What-If Scenario Category Emissions</b>",
        )
        fig_whatif.update_layout(height=290, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_whatif, use_container_width=True)


# ==============================================================================
# TAB 3: LONGITUDINAL TRACKING
# ==============================================================================
with tab_hist:
    st.markdown("### 📈 Longitudinal Carbon Audit Tracking")
    df_hist = st.session_state.history_df

    if df_hist.empty:
        st.info("No historical records logged yet. Use the 'Save to Personal History Log' button in the Daily Estimator tab.")
    else:
        df_hist = df_hist.copy()
        df_hist["date"] = pd.to_datetime(df_hist["date"], format="mixed", errors="coerce")
        df_hist = df_hist.dropna(subset=["date"])
        df_hist = df_hist.sort_values(by="date").reset_index(drop=True)

        # Metrics row
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.metric("Total Days Tracked", f"{len(df_hist)} Days")
        with h2:
            st.metric("All-Time Average", f"{df_hist['total_emissions_kg'].mean():.2f} kg CO₂e")
        with h3:
            st.metric("Best Daily Record", f"{df_hist['total_emissions_kg'].min():.2f} kg CO₂e")
        with h4:
            st.metric("7-Day Rolling Avg", f"{df_hist.tail(7)['total_emissions_kg'].mean():.2f} kg CO₂e")

        # Time series chart
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=df_hist["date"], y=df_hist["total_emissions_kg"],
            mode="lines+markers", name="Daily Footprint (LCA)",
            line=dict(color="#10b981", width=3), marker=dict(size=8, color="#065f46")
        ))
        if "ml_predicted_kg" in df_hist.columns:
            fig_time.add_trace(go.Scatter(
                x=df_hist["date"], y=df_hist["ml_predicted_kg"],
                mode="lines", name="ML Model Predicted",
                line=dict(color="#f59e0b", width=2, dash="dash")
            ))
        fig_time.add_hline(
            y=5.60, line_dash="dot", line_color="#ef4444",
            annotation_text="India Average (5.60 kg)", annotation_position="bottom right"
        )
        fig_time.update_layout(
            title="<b>Daily Carbon Footprint Over Time (kg CO₂e/day)</b>",
            xaxis_title="Date", yaxis_title="Emissions (kg CO₂e)",
            hovermode="x unified", height=350, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_time, use_container_width=True)

        # Stacked Activity evolution
        st.markdown("##### Activity Evolution Over Time")
        stack_cols = ["transport_kg", "electricity_kg", "food_kg", "waste_emissions_kg"]
        if all(c in df_hist.columns for c in stack_cols):
            fig_stack = go.Figure()
            fig_stack.add_trace(go.Bar(x=df_hist["date"], y=df_hist["transport_kg"], name="Transport", marker_color="#ef4444"))
            fig_stack.add_trace(go.Bar(x=df_hist["date"], y=df_hist["electricity_kg"], name="Electricity", marker_color="#f59e0b"))
            fig_stack.add_trace(go.Bar(x=df_hist["date"], y=df_hist["food_kg"], name="Food", marker_color="#10b981"))
            fig_stack.add_trace(go.Bar(x=df_hist["date"], y=df_hist["waste_emissions_kg"], name="Waste", marker_color="#06b6d4"))
            fig_stack.update_layout(
                barmode='stack', title="<b>Daily Activity Stack Breakdown</b>",
                xaxis_title="Date", yaxis_title="kg CO₂e", height=320, margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_stack, use_container_width=True)

        # Table
        with st.expander("📄 View Full Audit Log Table"):
            st.dataframe(df_hist.style.format({"total_emissions_kg": "{:.2f}", "ml_predicted_kg": "{:.2f}"}), use_container_width=True)


# ==============================================================================
# TAB 4: ML MODELS & METHODOLOGY
# ==============================================================================
with tab_ml:
    st.markdown("### 🔬 Machine Learning Evaluation & Scientific Methodology")
    st.markdown("""
    This platform rigorously separates **Deterministic Scientific LCA Factors** (CEA, ARAI, CPCB, ICAR)
    from **Empirical Machine Learning Regressors** trained on 2,500 lifestyle patterns to model non-linear behaviors,
    occupancy thresholds, and seasonal variance without data leakage.
    """)

    # 1. Model Comparison Table
    st.markdown("##### 1. Model Benchmark Comparison on Held-Out Test Data (N = 500)")
    comp_df = model.get_model_comparison_table()
    st.dataframe(
        comp_df.style.highlight_min(subset=["Test MAE (kg CO₂e)", "Test RMSE (kg CO₂e)"], color="#dcfce7")
               .highlight_max(subset=["Test R² Score"], color="#dcfce7")
               .format({"Test MAE (kg CO₂e)": "{:.4f}", "Test RMSE (kg CO₂e)": "{:.4f}", "Test R² Score": "{:.4f}", "Train R² Score": "{:.4f}"}),
        use_container_width=True
    )

    # 2. Parity Plot and Residuals
    st.markdown("---")
    st.markdown("##### 2. Parity Plots: Predicted vs Actual on Held-Out Test Data")
    parity_data = model.get_parity_plot_data()
    if parity_data:
        y_true = parity_data["y_true"]
        preds_dict = parity_data["preds"]
        gb_preds = preds_dict["Gradient Boosting Regressor"]

        p_c1, p_c2 = st.columns(2)
        with p_c1:
            fig_par = go.Figure()
            fig_par.add_trace(go.Scatter(
                x=y_true, y=gb_preds, mode="markers", name="Gradient Boosting",
                marker=dict(color="#10b981", size=8, opacity=0.8)
            ))
            min_v = min(min(y_true), min(gb_preds))
            max_v = max(max(y_true), max(gb_preds))
            fig_par.add_trace(go.Scatter(
                x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                name="Ideal 1:1 Line", line=dict(color="#ef4444", dash="dash")
            ))
            fig_par.update_layout(
                title="<b>Gradient Boosting Parity Plot (R² = 0.9455)</b>",
                xaxis_title="Actual Emissions (kg CO₂e)", yaxis_title="Predicted Emissions (kg CO₂e)",
                height=320, margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_par, use_container_width=True)

        with p_c2:
            residuals = np.array(y_true) - np.array(gb_preds)
            fig_res = px.histogram(
                residuals, nbins=25,
                title="<b>Residual Error Distribution (Actual - Predicted)</b>",
                color_discrete_sequence=["#06b6d4"],
                labels={"value": "Residual Error (kg CO₂e)"}
            )
            fig_res.add_vline(x=0, line_dash="dash", line_color="#ef4444")
            fig_res.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_res, use_container_width=True)

    # 3. Feature Importances
    st.markdown("---")
    st.markdown("##### 3. Top Feature Importances (Tree Ensemble)")
    fi_df = model.get_feature_importances()
    if not fi_df.empty:
        fig_fi = px.bar(
            fi_df.head(10), x="importance", y="feature_clean",
            orientation="h", color="importance", color_continuous_scale="Teal",
            title="<b>Top 10 Emission Drivers Identified by Machine Learning</b>"
        )
        fig_fi.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Feature Weight", height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_fi, use_container_width=True)

    # 4. Verified Factor Database
    st.markdown("---")
    st.markdown("##### 4. Official India-Specific Emission Factors Database")
    st.dataframe(emissions.load_emission_factors(), use_container_width=True)


# ==============================================================================
# TAB 5: EXPORT AUDIT REPORT
# ==============================================================================
with tab_report:
    st.markdown("### 📄 Export Carbon Footprint Audit Report")
    curr_res = emissions.calculate_total_emissions(**user_inputs)
    curr_recs = recommendations.generate_recommendations(curr_res)

    st.markdown("##### Current Audit Summary")
    st.write(f"- **Assessment Date:** {date.today()}")
    st.write(f"- **Calculated Daily Total:** **{curr_res['total_kg']} kg CO₂e** (Range: {curr_res['uncertainty_min']} – {curr_res['uncertainty_max']} kg)")
    st.write(f"- **Annualized Footprint:** **{curr_res['annual_tonnes']} Tonnes CO₂e / year**")
    st.write(f"- **Eco-Status:** **{curr_res['benchmarks']['eco_rating']}**")

    # CSV Generation
    audit_data = [
        {"Category": "Total Footprint", "Metric": "Daily Emissions", "Value": curr_res['total_kg'], "Unit": "kg CO₂e/day"},
        {"Category": "Uncertainty Lower", "Metric": "Min Bound", "Value": curr_res['uncertainty_min'], "Unit": "kg CO₂e/day"},
        {"Category": "Uncertainty Upper", "Metric": "Max Bound", "Value": curr_res['uncertainty_max'], "Unit": "kg CO₂e/day"},
        {"Category": "Annualized", "Metric": "Yearly Extrapolation", "Value": curr_res['annual_tonnes'], "Unit": "Tonnes/year"},
        {"Category": "Transportation", "Metric": "Tailpipe & Transit", "Value": curr_res['breakdown']['Transportation'], "Unit": "kg CO₂e/day"},
        {"Category": "Electricity & Cooking", "Metric": "Grid & LPG", "Value": curr_res['breakdown']['Electricity & Cooking'], "Unit": "kg CO₂e/day"},
        {"Category": "Food & Diet", "Metric": "Dietary GHG", "Value": curr_res['breakdown']['Food & Diet'], "Unit": "kg CO₂e/day"},
        {"Category": "Waste Management", "Metric": "Methane & Landfill", "Value": curr_res['breakdown']['Waste Management'], "Unit": "kg CO₂e/day"},
    ]
    for r in curr_recs:
        audit_data.append({
            "Category": f"Recommendation: {r['category']}",
            "Metric": r['title'],
            "Value": r['daily_co2_saving_kg'],
            "Unit": "kg CO₂e saved/day",
        })

    csv_audit = pd.DataFrame(audit_data).to_csv(index=False).encode('utf-8')

    c_down1, c_down2 = st.columns(2)
    with c_down1:
        st.download_button(
            label="📥 Download Daily Carbon Audit Report (CSV)",
            data=csv_audit,
            file_name=f"carbon_audit_report_{date.today()}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
    with c_down2:
        if not st.session_state.history_df.empty:
            csv_hist = st.session_state.history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Tracking History Dataset (CSV)",
                data=csv_hist,
                file_name=f"carbon_tracking_history_{date.today()}.csv",
                mime="text/csv",
                use_container_width=True
            )
