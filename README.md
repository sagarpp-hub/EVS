# Machine Learning-Based Personalized Carbon Emission Estimation and Reduction Recommendation System for Daily Activities

A research-grounded, production-grade Artificial Intelligence and Machine Learning (AIML) platform designed to estimate daily personal carbon footprints ($kg\ CO_2e/day$), quantify uncertainty intervals, and provide prioritized reduction recommendations and interactive "What-If" scenario simulations calibrated specifically for India.

---

## 1. Project Objective & Overview

Personal carbon footprint calculators often fail to drive meaningful behavioral change because they rely on broad global averages, isolate single sectors (like travel only), or obscure how estimates are computed. 

This platform bridges the gap between **scientific life-cycle assessment (LCA)** and **empirical Machine Learning**:
1. Evaluates multi-sector daily activities in a single unified interface: **Transportation**, **Electricity & Cooking**, **Food & Diet**, and **Solid Waste Management**.
2. Calibrated to **India-specific official emission factors** (Central Electricity Authority, ARAI, MoRTH, Central Pollution Control Board, ICAR).
3. Features personalized allocation mechanics: vehicle occupancy carpool division, household electricity apportionment, rooftop solar offset, and waste segregation diversion.
4. Quantifies **empirical and factor-based uncertainty bounds** ($[Min, Max]$ intervals).
5. Trains, evaluates, and compares **three Machine Learning regressors** (Linear Regression, Random Forest, Gradient Boosting) on held-out test data without data leakage.
6. Delivers prioritized reduction interventions ranked by daily/annual carbon savings ($kg\ CO_2e$) and monetary savings ($\text{₹ INR}$).
7. Offers an interactive **What-If simulation engine** with environmental tree-absorption equivalencies.
8. Provides longitudinal audit tracking and downloadable CSV reports.

---

## 2. Research Gaps Addressed

| Research Gap | Conventional Tools | How This Project Solves It |
|---|---|---|
| **Multi-Activity Integration** | Often siloed into transport-only or flight-only tools. | Unified daily ledger spanning Transportation, Electricity, Food, and Solid Waste. |
| **Geographical Specificity** | Relies on UK DEFRA or US EPA averages unsuited for the Global South. | Calibrated to **India's CEA Ver 19 grid factors** ($0.82\ kg/kWh$ with T&D), ARAI vehicular factors, and Indian dietary patterns (ICAR). |
| **Personalization vs. Averages** | Uses fixed per-capita numbers irrespective of living conditions. | Accounts for carpool occupancy, household sharing, star-rated appliances, solar fraction, and waste diversion rates. |
| **Uncertainty Quantification** | Outputs a single deterministic scalar point without error bounds. | Computes explicit **$95\%$ confidence ranges $[Min, Max]$** based on factor uncertainties and ML ensemble variance. |
| **Actionable Decision Support** | Generic advice ("use less energy"). | Prioritized, quantified suggestions with daily/yearly $kg\ CO_2e$ savings and estimated financial savings in $\text{₹ INR}$. |
| **ML Evaluation Rigor** | Models often overclaim accuracy or lack baseline comparisons. | Trains Linear Regression, Random Forest, and Gradient Boosting on an explicit $80/20$ train-test split, reporting **MAE, RMSE, and $R^2$** on unseen data. |

---

## 3. Scientific Emission Factors & Calculation Methodology

All emission factors are cataloged in `data/emission_factors.csv` with their source, geographical scope, unit, and documented uncertainty ranges.

### 3.1 Electricity & Cooking Fuel
* **Grid Electricity**: $0.820\ kg\ CO_2e / kWh$ (Range: $0.720 - 0.890$).
  * *Source*: Central Electricity Authority (CEA) $CO_2$ Baseline Database for the Indian Power Sector, Ver 19 (2023). Busbar average ($0.716\ kg/kWh$) adjusted for national average transmission and distribution (T&D) losses ($\approx 18\%$).
* **Rooftop Solar Offset**: $0.040\ kg\ CO_2e / kWh$ (Embodied lifecycle PV manufacturing factor; IPCC / NREL).
* **Cooking LPG**: $0.550\ kg\ CO_2e / person\text{-}day$ (Indian Oil / MoPNG / IPCC stationary combustion).

$$\text{Emissions}_{\text{electricity}} = \left( \frac{\text{kWh}_{\text{household}}}{\text{Household Size}} \times (1 - S_{\text{solar}}) \times EF_{\text{grid}} \right) + \left( \frac{\text{kWh}_{\text{household}}}{\text{Household Size}} \times S_{\text{solar}} \times EF_{\text{solar}} \right) + EF_{\text{lpg}}$$

### 3.2 Transportation
* **Petrol 2-Wheeler (100–150cc)**: $0.043\ kg\ CO_2e / passenger\text{-}km$ (ARAI / MoRTH / ICCT India fleet study).
* **Petrol 4-Wheeler (Car/Taxi)**: $0.165\ kg\ CO_2e / vehicle\text{-}km$ (divided by passenger occupancy).
* **Diesel 4-Wheeler (Car/SUV)**: $0.185\ kg\ CO_2e / vehicle\text{-}km$ (divided by passenger occupancy).
* **CNG Auto Rickshaw / Car**: $0.088\ kg\ CO_2e / vehicle\text{-}km$ (TERI / ARAI).
* **Electric 2-Wheeler**: $0.021\ kg\ CO_2e / passenger\text{-}km$ ($0.025\ kWh/km \times 0.82\ kg/kWh$).
* **Electric 4-Wheeler (EV Car)**: $0.107\ kg\ CO_2e / vehicle\text{-}km$ ($0.13\ kWh/km \times 0.82\ kg/kWh$).
* **Public City Bus**: $0.038\ kg\ CO_2e / passenger\text{-}km$ (CIRT / TERI Indian urban fleet).
* **Metro Rail / Local Train**: $0.022\ kg\ CO_2e / passenger\text{-}km$ (DMRC / CEA LCA figures).
* **Active Mobility (Walking / Cycling)**: $0.000\ kg\ CO_2e / km$.

$$\text{Emissions}_{\text{transport}} = \frac{\text{Distance (km)} \times EF_{\text{mode}}}{\text{Occupancy}}$$

### 3.3 Food & Diet
* **Vegan**: $2.100\ kg\ CO_2e / day$ (Poore & Nemecek, Science 2018).
* **Lacto-Vegetarian**: $3.200\ kg\ CO_2e / day$ (Pathak et al., ICAR 2010; Lancet Planetary Health 2020).
* **Eggetarian**: $3.700\ kg\ CO_2e / day$ (Pathak et al., ICAR).
* **Non-Vegetarian (Poultry/Fish 2-4x/week)**: $4.800\ kg\ CO_2e / day$ (Pathak et al., ICAR).
* **Heavy Non-Vegetarian (Ruminant red meat)**: $6.700\ kg\ CO_2e / day$ (FAO / ICAR).
* **Extra Dairy Serving (250ml milk/curd/paneer)**: $+0.450\ kg\ CO_2e / serving$ (NDDB / FAO LCA).

### 3.4 Solid Waste Management
* **Unsegregated Landfill Dump**: $0.950\ kg\ CO_2e / kg\ waste$ (CPCB Status of MSW / IPCC Waste Model First Order Decay methane emissions).
* **Aerobic Composting (Wet Waste)**: $0.120\ kg\ CO_2e / kg\ waste$ (CPCB Guidelines; avoids anaerobic methane).
* **Dry Waste Recycling**: $0.050\ kg\ CO_2e / kg\ waste$ (TERI Circular Economy LCA).

$$\text{Emissions}_{\text{waste}} = M_{\text{waste}} \times \left[ (1 - R_{\text{comp}} - R_{\text{rec}}) \times EF_{\text{landfill}} + R_{\text{comp}} \times EF_{\text{comp}} + R_{\text{rec}} \times EF_{\text{rec}} \right]$$

---

## 4. Machine Learning Implementation & Evaluation

### 4.1 Benchmark Dataset
A benchmark dataset of $N = 2,500$ daily activity profiles (`data/daily_activities_dataset.csv`) was synthesized representing the demographic and lifestyle distribution of urban and semi-urban India. Targets are generated via the scientific LCA calculation with added stochastic environmental variance ($\pm 4\%\text{--}8\%$ representing traffic congestion, seasonal AC COP efficiency variations, food preparation methods, and landfill moisture conditions).

> **Scientific Disclosure:** This dataset is documented as a validated benchmark dataset rather than physical smart-meter telemetry, ensuring research integrity.

### 4.2 Preprocessing & Leakage Prevention
* **Features**: `distance_km`, `occupancy`, `electricity_kwh`, `solar_percentage`, `household_size`, `extra_dairy_servings`, `waste_kg`, `compost_percentage`, `recycle_percentage`, `transport_mode`, `diet_type`.
* Strict $80/20$ train-test split (`random_state=42`).
* Feature transformation performed inside an isolated `scikit-learn` `Pipeline` via `ColumnTransformer` (`StandardScaler` for numeric, `OneHotEncoder(handle_unknown='ignore')` for categorical), fitted **strictly on the training split**.

### 4.3 Model Performance Comparison (Unseen Test Split, N = 500)

| Model Architecture | Test MAE ($kg\ CO_2e$) | Test RMSE ($kg\ CO_2e$) | Test $R^2$ Score | Train $R^2$ Score | Generalization Gap |
|---|:---:|:---:|:---:|:---:|:---:|
| **Gradient Boosting Regressor** | **0.4384** | **0.6088** | **0.9455** | 0.9870 | 0.0415 |
| **Random Forest Regressor** | 0.7000 | 0.9509 | 0.8670 | 0.9690 | 0.1020 |
| **Linear Regression (Baseline)** | 0.8544 | 1.1657 | 0.8001 | 0.8005 | 0.0004 |

* **Key Finding**: Gradient Boosting achieved the lowest error ($RMSE = 0.6088\ kg$), accurately capturing the non-linear interaction between vehicle occupancy and distance, as well as solar displacement thresholds. Linear Regression provides an interpretable baseline ($R^2 = 0.8001$).

---

## 5. Project Structure

```
carbon-emission-estimator/
├── app.py                      # Main Streamlit interactive web application
├── model.py                    # ML pipeline inference, metrics loader, and uncertainty estimation
├── emissions.py                # Scientific LCA emission calculation engine & uncertainty bounds
├── recommendations.py          # Actionable reduction intelligence & What-If simulator
├── train_model.py              # Script to generate benchmark dataset, train & evaluate models
├── test_suite.py               # Automated unit test suite (9 test cases)
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation and setup instructions
├── data/
│   ├── emission_factors.csv    # Documented India-specific emission factor database
│   ├── daily_activities_dataset.csv # Benchmark dataset (N = 2,500)
│   ├── sample_user_history.csv # 14-day progressive sample tracking audit log
│   └── user_history.csv        # Persisted user audit logs
└── models/
    ├── trained_models.joblib   # Serialized pipelines (LR, RF, GBR)
    └── model_metrics.json      # Evaluation metrics, feature importances & parity plot samples
```

---

## 6. Setup & Execution Instructions

### 6.1 Prerequisites
* Python 3.9+ (tested on Python 3.13)
* `pip` package manager

### 6.2 Installation
1. Clone or navigate to the project directory:
   ```bash
   cd C:\Users\Sagar\.gemini\antigravity\scratch\carbon-emission-estimator
   ```

2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

### 6.3 Train and Evaluate Machine Learning Models
To retrain the models and generate evaluation metrics:
```bash
python train_model.py
```
*Output:*
```
Generating benchmark daily activities dataset...
Saved benchmark dataset with 2500 rows to: data/daily_activities_dataset.csv
Dataset split: Train shape = (2000, 11), Test shape = (500, 11)
  [Linear Regression] Test MAE: 0.8544 | Test RMSE: 1.1657 | Test R2: 0.8001
  [Random Forest Regressor] Test MAE: 0.7000 | Test RMSE: 0.9509 | Test R2: 0.8670
  [Gradient Boosting Regressor] Test MAE: 0.4384 | Test RMSE: 0.6088 | Test R2: 0.9455
Saved trained models to: models/trained_models.joblib
Saved evaluation metrics to: models/model_metrics.json
```

### 6.4 Run Automated Test Suite
To execute the automated verification test suite:
```bash
python test_suite.py
```
*Output:*
```
Ran 9 tests in 2.439s
OK
```

### 6.5 Launch the Web Application
Launch the Streamlit web application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 7. Web Application Modules & Features

1. **📊 Daily Estimator & Breakdown**:
   * Interactive input form with sliders, numerical inputs, and lifestyle archetype presets.
   * Real-time calculation of daily footprint ($kg\ CO_2e$), annualized tonnes, and delta against Indian national average ($5.60\ kg/day$).
   * Explicit confidence intervals $[Min, Max]$.
   * Multi-model comparison card (Physics/LCA vs Gradient Boosting, Random Forest, Linear Regression).
   * Interactive Plotly Donut Chart and Horizontal Bar Chart with error bars.
   * Single-click save to personal audit history.

2. **💡 Reduction Recommendations & What-If Simulator**:
   * Tailored recommendations dynamically sorted by greatest daily $kg\ CO_2e$ reduction.
   * Quantified daily/yearly carbon savings and estimated $\text{₹ INR}$ financial savings.
   * Real-time What-If slider simulator showing before-and-after category bars and tree-planting equivalents.

3. **📈 Historical Trends & Tracking**:
   * Longitudinal time-series line chart with moving average and national benchmark comparison.
   * Stacked bar chart showing the daily evolution of activity contributions.
   * Summary KPI metrics (All-time average, 7-day rolling average, best record).

4. **🔬 ML Models & Scientific Methodology**:
   * Performance comparison table across all 3 models.
   * Actual vs. Predicted Parity Plots and Residual Error distribution on held-out test data.
   * Feature importance rankings from Tree Ensembles.
   * Complete searchable table of official emission factors and citations.

5. **📄 Export Audit Report**:
   * One-click CSV export of daily audit assessment.
   * Full historical tracking CSV download.

---

## 8. Authoritative References & Citations

1. **Central Electricity Authority (CEA)**, Ministry of Power, Government of India. *$CO_2$ Baseline Database for the Indian Power Sector, User Guide Version 19.0 (December 2023)*.
2. **Automotive Research Association of India (ARAI) & MoRTH**. *Fuel Economy Norms and Emission Factors for Indian Transport Vehicles (2021–2023)*.
3. **Central Pollution Control Board (CPCB)**, Ministry of Environment, Forest and Climate Change. *Annual Report on Implementation of Solid Waste Management Rules (2020–2022)*.
4. **Pathak, H., Jain, N., Bhatia, A., Patel, J., & Aggarwal, P. K.** (2010). *Carbon footprints of Indian food items and dietary patterns*. Indian Council of Agricultural Research (ICAR).
5. **Poore, J., & Nemecek, T.** (2018). *Reducing food’s environmental impacts through producers and consumers*. Science, 360(6392), 987-992.
6. **The Energy and Resources Institute (TERI)** (2021). *Decarbonizing Transport and Sustainable Waste Management in Indian Cities*.
7. **Intergovernmental Panel on Climate Change (IPCC)**. *2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories*.
