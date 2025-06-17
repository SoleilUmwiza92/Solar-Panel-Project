# ☀️ Solar Panel Energy Yield Simulation

This repository contains a Python-based solar panel yield model developed to simulate and estimate energy production based on KNMI weather data, geographical parameters, and PV system specifications. The model is currently being adapted into an AnyLogic environment for advanced simulation and validation.

---

## 🔍 Project Overview

The goal of this project is to simulate solar panel energy production using real weather data and to validate the simulation against available production records. This helps assess performance, optimize setup, and support future decision-making on solar energy infrastructure.

---

## 🧠 Key Features

- Utilizes **KNMI hourly weather data** (radiation, temperature, wind, etc.)
- Computes **solar position** using latitude and longitude
- Models **global irradiance on tilted surfaces**
- Estimates **energy yield (kWh)** for custom panel setups
- Produces **3D visualizations** of solar panel output over time
- Designed for **integration into AnyLogic** for agent-based simulation

---

## 🧪 Technologies Used

- **Python** (main calculation engine)
- `pvlib`, `pandas`, `matplotlib`, `numpy`, `requests`
- **KNMI data** for hourly Dutch weather conditions
- **AnyLogic** (for simulation modeling and visualization)

---



