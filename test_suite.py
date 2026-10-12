"""
test_suite.py
=============
Automated verification tests for:
- emission calculations & uncertainty bounds
- personalized recommendations & What-If simulation
- ML inference & model performance metrics
"""

import unittest
import numpy as np
import emissions
import recommendations
import model


class TestEmissionsModule(unittest.TestCase):
    def test_factors_loaded(self):
        df = emissions.load_emission_factors()
        self.assertFalse(df.empty)
        self.assertIn("factor_value", df.columns)
        self.assertIn("uncertainty_min", df.columns)
        self.assertIn("uncertainty_max", df.columns)

    def test_transport_calculation(self):
        # Petrol car solo 20 km
        res = emissions.calculate_transport_emissions("petrol_car", 20.0, occupancy=1)
        self.assertGreater(res["emissions_kg"], 0.0)
        self.assertLessEqual(res["uncertainty_min"], res["emissions_kg"])
        self.assertGreaterEqual(res["uncertainty_max"], res["emissions_kg"])

        # Carpooling 2 people cuts per-person emissions in half
        res_carpool = emissions.calculate_transport_emissions("petrol_car", 20.0, occupancy=2)
        self.assertAlmostEqual(res_carpool["emissions_kg"], res["emissions_kg"] / 2.0, places=3)

        # Cycling is zero tailpipe
        res_cycle = emissions.calculate_transport_emissions("cycling_walking", 10.0)
        self.assertEqual(res_cycle["emissions_kg"], 0.0)

    def test_electricity_calculation(self):
        # 10 kWh split across 2 people = 5 kWh/capita
        res = emissions.calculate_electricity_emissions(10.0, solar_percentage=0.0, household_size=2)
        self.assertEqual(res["per_capita_kwh"], 5.0)
        self.assertGreater(res["emissions_kg"], 0.0)
        self.assertLessEqual(res["uncertainty_min"], res["emissions_kg"])

        # 100% solar drastically reduces emissions
        res_solar = emissions.calculate_electricity_emissions(10.0, solar_percentage=100.0, household_size=2)
        self.assertLess(res_solar["emissions_kg"], res["emissions_kg"])

    def test_food_calculation(self):
        res_vegan = emissions.calculate_food_emissions("vegan", 0)
        res_veg = emissions.calculate_food_emissions("vegetarian", 0)
        res_nonveg = emissions.calculate_food_emissions("nonveg_heavy", 0)
        self.assertLess(res_vegan["emissions_kg"], res_veg["emissions_kg"])
        self.assertLess(res_veg["emissions_kg"], res_nonveg["emissions_kg"])

    def test_waste_calculation(self):
        res_landfill = emissions.calculate_waste_emissions(1.0, compost_percentage=0, recycle_percentage=0)
        res_diverted = emissions.calculate_waste_emissions(1.0, compost_percentage=60, recycle_percentage=30)
        self.assertLess(res_diverted["emissions_kg"], res_landfill["emissions_kg"])


class TestRecommendationsModule(unittest.TestCase):
    def test_recommendations_generation(self):
        profile = emissions.calculate_total_emissions(
            transport_mode="petrol_car",
            distance_km=30.0,
            occupancy=1,
            electricity_kwh=15.0,
            solar_percentage=0.0,
            household_size=2,
            diet_type="nonveg_heavy",
            extra_dairy_servings=3,
            waste_kg=0.8,
            compost_percentage=0.0,
            recycle_percentage=0.0,
        )
        recs = recommendations.generate_recommendations(profile)
        self.assertGreater(len(recs), 0)
        # Check that top recommendation has saving > 0
        self.assertGreater(recs[0]["daily_co2_saving_kg"], 0.0)

    def test_what_if_simulation(self):
        baseline = {
            "transport_mode": "petrol_car",
            "distance_km": 25.0,
            "occupancy": 1,
            "electricity_kwh": 10.0,
            "solar_percentage": 0.0,
            "household_size": 3,
            "diet_type": "vegetarian",
            "extra_dairy_servings": 1,
            "waste_kg": 0.5,
            "compost_percentage": 0.0,
            "recycle_percentage": 10.0,
            "include_cooking_lpg": True,
        }
        modifications = {
            "transport_mode": "metro_rail",
            "solar_percentage": 50.0,
        }
        whatif = recommendations.simulate_what_if(baseline, modifications)
        self.assertGreater(whatif["daily_saved_kg"], 0.0)
        self.assertGreater(whatif["pct_reduction"], 0.0)
        self.assertGreater(whatif["trees_equivalent"], 0.0)


class TestModelModule(unittest.TestCase):
    def test_ml_inference(self):
        inputs = {
            "transport_mode": "petrol_car",
            "distance_km": 20.0,
            "occupancy": 1,
            "electricity_kwh": 8.0,
            "solar_percentage": 0.0,
            "household_size": 3,
            "diet_type": "vegetarian",
            "extra_dairy_servings": 1,
            "waste_kg": 0.5,
            "compost_percentage": 10.0,
            "recycle_percentage": 20.0,
        }
        res = model.predict_emissions(inputs)
        self.assertIn("predictions_by_model", res)
        self.assertIn("Linear Regression", res["predictions_by_model"])
        self.assertIn("Random Forest Regressor", res["predictions_by_model"])
        self.assertIn("Gradient Boosting Regressor", res["predictions_by_model"])
        self.assertGreater(res["primary_prediction_kg"], 0.0)

    def test_comparison_table(self):
        comp = model.get_model_comparison_table()
        self.assertEqual(len(comp), 3)
        self.assertIn("Test R² Score", comp.columns)
        self.assertIn("Test MAE (kg CO₂e)", comp.columns)
        self.assertIn("Test RMSE (kg CO₂e)", comp.columns)


if __name__ == "__main__":
    unittest.main()
