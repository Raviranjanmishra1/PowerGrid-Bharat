PowerGrid-Bharat

https://dbc-ff5e1c9e-f295.cloud.databricks.com/browse/folders/1131707218103137?o=7474644341547936


https://dbc-ff5e1c9e-f295.cloud.databricks.com/editor/files/1107782230867992?o=7474644341547936

# ⚡ PowerGrid-Bharat — India Power Grid Intelligence Platform

> A Graph Neural Network-powered system for forecasting, monitoring, and optimizing India's national power grid using real generation, renewable energy, and demand-supply data.

---

## 📌 Overview

PowerGuard AI is an end-to-end ML pipeline that ingests raw power generation data from India's Central Electricity Authority, cleans and engineers time-series features, builds a state-level geographic graph of the national grid, and trains a Graph Neural Network (GNN) to model inter-state power flows and forecast generation deficits.

The system models India's power grid as a **graph where each state is a node** (connected by shared geographic borders), enabling spatially-aware forecasting — not just predicting state-level demand and supply in isolation, but accounting for how surplus from one state can flow to neighboring deficit states up to 3 hops away.

---

## 🗂 Repository Structure

```
PowerGuard-AI/
│
├── 01_cleaning_timeseries_forecast.ipynb   # Data cleaning, EDA & Chronos forecasting
├── 02_graph_construction.ipynb             # Graph building, GNN export, viz
│
├── data/
│   ├── daily-power-generation.csv              # Raw station-level daily generation (2017–2024)
│   ├── daily-renewable-energy-generation.csv   # State-level wind/solar/other RE (2020–)
│   ├── energy-requirement-and-availabililty.csv # Monthly demand vs. availability by state
│   │
│   ├── processed_daily_india.csv           # All-India daily aggregation
│   ├── processed_daily_state.csv           # State-level daily generation + RE
│   ├── processed_demand_supply.csv         # Monthly demand, supply, deficit by state
│   ├── processed_station_summary.csv       # Station-level lifetime summary stats
│   │
│   ├── feature_cols.csv                    # Feature column list for GNN node features
│   ├── node_index.csv                      # State → node index mapping (30 nodes)
│   └── graph_data_for_gnn.npz              # Final graph export for PyTorch Geometric
│
└── README.md
```

---

## 🔄 Pipeline

```
Raw CSVs  →  Cleaning & Null Handling  →  Time-Series Aggregation
    →  Chronos Forecasting  →  Graph Construction  →  GNN Export
```

### Notebook 1 — Data Cleaning, Time Series & Forecasting

- **Load** three raw datasets: station-level generation, renewable generation, demand/supply
- **Clean** nulls, standardize date types, tag Bhutan imports separately
- **Aggregate** to daily All-India and state-level time series
- **Visualize** trends, volatility, day-of-week patterns, heatmaps, station-type breakdowns
- **Forecast** 30-day generation using [Amazon Chronos-T5-Small](https://github.com/amazon-science/chronos-forecasting) (~46M params) with backtesting
- **Save** processed outputs for the graph notebook

### Notebook 2 — Graph Construction for GNN

- **Canonicalize** state names across all datasets
- **Build adjacency** from real Indian state geographic borders (30 mainland states, ~70 edges)
- **Construct node features** from aggregated station stats and Chronos forecasts
- **Visualize** the grid graph and overlay surplus/deficit signals
- **Demo** 3-hop neighborhood reachability (which states can receive power from a surplus state)
- **Export** to `.npz` format for PyTorch Geometric

---

## 📊 Data

| File | Description | Key Columns |
|------|-------------|-------------|
| `daily-power-generation.csv` | Raw station-level generation, 2017–2024 | `date`, `state_name`, `station_type`, `sector`, `monitored_capacity`, `todays_gen_act` |
| `daily-renewable-energy-generation.csv` | State wind/solar/RE totals | `date`, `state_name`, `wind_energy`, `solar_energy`, `total_renewable_energy` |
| `energy-requirement-and-availabililty.csv` | Monthly demand vs. availability | `month`, `state_name`, `energy_requirement`, `energy_availability` |
| `processed_demand_supply.csv` | Cleaned demand/supply with deficit calc | `demand_mu`, `supply_mu`, `deficit_mu`, `deficit_pct`, `grid_region` |
| `graph_data_for_gnn.npz` | Final graph: node features, edge index, edge weights | PyG-ready format |

**Coverage:** 30 Indian states/UTs · 2017–2024 · Daily resolution for generation, Monthly for demand/supply

---

## 🧠 Model

| Component | Choice |
|-----------|--------|
| Time-series forecasting | Amazon Chronos-T5-Small (zero-shot) |
| Graph framework | NetworkX → PyTorch Geometric |
| Node count | 30 (mainland states) |
| Edge definition | Shared geographic borders |
| Edge weights | Grid region proximity |
| Node features | `avg_daily_gen`, `std_daily_gen`, Chronos forecast, RE share, deficit % |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn
pip install chronos-forecasting torch
pip install networkx torch-geometric
pip install scikit-learn
```

### Run the Pipeline

```bash
# Step 1: Data cleaning and forecasting
jupyter notebook 01_cleaning_timeseries_forecast.ipynb

# Step 2: Graph construction and GNN export
jupyter notebook 02_graph_construction.ipynb
```

Make sure the raw CSV files are in the same directory as the notebooks (or update the `DATA_DIR` path at the top of Notebook 1).

---

## 📈 Key Insights Surfaced

- **Thermal dominates** generation capacity; RE share is growing but uneven across states
- **Chronos zero-shot forecasting** achieves strong backtesting accuracy on India's generation time series without any fine-tuning
- **Deficit hotspots** are geographically clustered — the graph structure reveals which neighboring states have surplus capacity to cover them
- **3-hop reachability** maps reveal realistic power-flow corridors across the national grid

---

## 📍 Data Sources

- [Central Electricity Authority (CEA)](https://cea.nic.in/) — Daily generation monitoring
- [Ministry of Power, India](https://powermin.gov.in/) — Demand/supply reports
- Indian state geographic adjacency — derived from official administrative boundaries

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

[MIT](LICENSE)
