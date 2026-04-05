#!/usr/bin/env python
# coding: utf-8

# # PowerGrid Bharat — Data Cleaning, Time Series & Forecasting
# **Local Jupyter version** — pandas + matplotlib. Same logic as Databricks notebook.
# 
# Ensure these files are in the same directory (or update paths):
# - `daily-power-generation.csv` (276 MB)
# - `daily-renewable-energy-generation.csv`
# - `energy-requirement-and-availabililty.csv`
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 40)


# ## 1. Load Raw Data

# In[2]:


# Update paths if your files are elsewhere
DATA_DIR = "./Dataset/archive_Indiaelectricity"
#pg = pd.read_csv("daily-power-generation.csv", dtype={"state_code": str})
#re = pd.read_csv("daily-renewable-energy-generation.csv", dtype={"state_code": str})
#ds = pd.read_csv("energy-requirement-and-availabililty.csv", dtype={"state_code": str})
pg = pd.read_csv(f"{DATA_DIR}/daily-power-generation.csv", dtype={"state_code": str})
re = pd.read_csv(f"{DATA_DIR}/daily-renewable-energy-generation.csv", dtype={"state_code": str})
ds = pd.read_csv(f"{DATA_DIR}/energy-requirement-and-availabililty.csv", dtype={"state_code": str})

print(f"Power Gen:     {pg.shape}")
print(f"Renewable:     {re.shape}")
print(f"Demand/Supply: {ds.shape}")


# ## 2. Cleaning — Report & Fix Nulls

# In[3]:


# ========================================
# 2a. POWER GENERATION — Report nulls
# ========================================
print("=" * 50)
print("POWER GENERATION — NULLS BEFORE CLEANING")
print("=" * 50)
null_report = pg.isnull().sum()
print(null_report[null_report > 0])

# What are the null rows?
pg_nulls = pg[pg[["sector", "station_type"]].isnull().any(axis=1)]
print(f"\nNull sector/station_type: {len(pg_nulls)} rows")
print("These are Bhutan import rows:")
print(pg_nulls[["date", "region", "state_name", "power_station"]].drop_duplicates("power_station").to_string(index=False))

# What stations have null capacity?
mc_nulls = pg[pg["monitored_capacity"].isnull()]
print(f"\nNull monitored_capacity: {len(mc_nulls)} rows across {mc_nulls.power_station.nunique()} stations:")
print(mc_nulls[["date", "state_name", "power_station", "station_type"]].drop_duplicates("power_station").to_string(index=False))


# In[4]:


# ========================================
# 2a. POWER GENERATION — Fix nulls
# ========================================
pg["date"] = pd.to_datetime(pg["date"])

# Tag Bhutan imports
pg["sector"] = pg["sector"].fillna("Import")
pg["station_type"] = pg["station_type"].fillna("Import")
pg["state_code"] = pg["state_code"].fillna("IMP")

# Fill capacity with station median, then 0
pg["monitored_capacity"] = pg.groupby("power_station")["monitored_capacity"].transform(
    lambda x: x.fillna(x.median())
)
pg["monitored_capacity"] = pg["monitored_capacity"].fillna(0)

# Fill gen nulls (only 3 each)
pg["todays_gen_prgm"] = pg["todays_gen_prgm"].fillna(0)
pg["todays_gen_act"] = pg["todays_gen_act"].fillna(0)

# Derived columns
pg["gen_deviation"] = pg["todays_gen_act"] - pg["todays_gen_prgm"]
pg["year"] = pg["date"].dt.year
pg["month"] = pg["date"].dt.month
pg["day_of_week"] = pg["date"].dt.day_name()
pg["week_of_year"] = pg["date"].dt.isocalendar().week.astype(int)

print(f"✅ Power Gen cleaned: {pg.shape}, remaining nulls: {pg.isnull().sum().sum()}")


# In[5]:


# ========================================
# 2b. RENEWABLE — Report & Fix
# ========================================
print("=" * 50)
print("RENEWABLE — NULLS BEFORE CLEANING")
print("=" * 50)
null_report = re.isnull().sum()
print(null_report[null_report > 0])

re_null_region = re[re["region"].isnull()]
print(f"\nNull region rows are all: {re_null_region.state_name.unique()}")

# Fix
re["date"] = pd.to_datetime(re["date"])
region_map = re.dropna(subset=["region"]).drop_duplicates("state_name").set_index("state_name")["region"].to_dict()
re["region"] = re["state_name"].map(region_map)
re.loc[re["state_name"] == "All India", "region"] = "All India"
re["solar_energy"] = re["solar_energy"].fillna(0)
re["other_renewable_energy"] = re["other_renewable_energy"].fillna(0)
re["region_short"] = re["region"].str.replace(" Region", "", regex=False)

print(f"\n✅ Renewable cleaned: {re.shape}, remaining nulls: {re.isnull().sum().sum()}")


# In[6]:


# ========================================
# 2c. DEMAND/SUPPLY — Report & Fix
# ========================================
print("=" * 50)
print("DEMAND/SUPPLY — NULLS BEFORE CLEANING")
print("=" * 50)
null_report = ds.isnull().sum()
print(null_report[null_report > 0])
print(f"Null state_code states: {ds[ds['state_code'].isnull()].state_name.unique()}")

ds["month"] = pd.to_datetime(ds["month"])
ds["state_code"] = ds["state_code"].fillna("NA")
ds = ds.rename(columns={"energy_requirement": "demand_mu", "energy_availability": "supply_mu"})
ds["deficit_mu"] = (ds["demand_mu"] - ds["supply_mu"]).round(3)
ds["deficit_pct"] = ((ds["demand_mu"] - ds["supply_mu"]) / ds["demand_mu"] * 100).round(2)

# Grid region mapping
grid_region_map = {
    "Chandigarh": "Northern", "Delhi": "Northern", "Haryana": "Northern",
    "Himachal Pradesh": "Northern", "UTs of J&K and Ladakh": "Northern",
    "Punjab": "Northern", "Rajasthan": "Northern", "Uttar Pradesh": "Northern",
    "Uttarakhand": "Northern",
    "Chhattisgarh": "Western", "Gujarat": "Western", "Madhya Pradesh": "Western",
    "Maharashtra": "Western", "Dadra & Nagar Haveli and Daman & Diu": "Western",
    "Goa": "Western",
    "Andhra Pradesh": "Southern", "Telangana": "Southern", "Karnataka": "Southern",
    "Kerala": "Southern", "Tamil Nadu": "Southern", "Puducherry": "Southern",
    "Lakshadweep": "Southern",
    "Bihar": "Eastern", "DVC": "Eastern", "Jharkhand": "Eastern",
    "Odisha": "Eastern", "West Bengal": "Eastern", "Sikkim": "Eastern",
    "Andaman- Nicobar": "Eastern",
    "Arunachal Pradesh": "North-Eastern", "Assam": "North-Eastern",
    "Manipur": "North-Eastern", "Meghalaya": "North-Eastern",
    "Mizoram": "North-Eastern", "Nagaland": "North-Eastern",
    "Tripura": "North-Eastern",
}
ds["grid_region"] = ds["state_name"].map(grid_region_map)

print(f"\n✅ Demand/Supply cleaned: {ds.shape}, remaining nulls: {ds.isnull().sum().sum()}")


# ## 3. Time Series Aggregations

# In[9]:


# ========================================
# 3a. Daily All India (conventional gen)
# ========================================
pg_india = pg[pg["region"] != "Bhutan Imp."]

daily_india = (
    pg_india.groupby("date")
    .agg(
        total_gen_actual=("todays_gen_act", "sum"),
        total_gen_programmed=("todays_gen_prgm", "sum"),
        total_capacity=("monitored_capacity", "sum"),
        active_stations=("power_station", "nunique"),
    )
    .reset_index()
)

# Merge All India renewable
re_india = re[re["state_name"] == "All India"][["date", "wind_energy", "solar_energy", "other_renewable_energy", "total_renewable_energy"]]
daily_india = daily_india.merge(re_india, on="date", how="left")

# Derived
daily_india["total_generation"] = daily_india["total_gen_actual"] + daily_india["total_renewable_energy"].fillna(0)
daily_india["renewable_share_pct"] = (
    daily_india["total_renewable_energy"].fillna(0) / daily_india["total_generation"] * 100
).round(2)
daily_india["gen_deviation_pct"] = (
    (daily_india["total_gen_actual"] - daily_india["total_gen_programmed"])
    / daily_india["total_gen_programmed"].replace(0, np.nan) * 100
).round(2)

print(f"✅ Daily India: {daily_india.shape}")
daily_india.tail()


# In[8]:


# ========================================
# 3b. Daily State level + renewable merge
# ========================================
daily_state = (
    pg_india.groupby(["date", "region", "state_name"])
    .agg(
        gen_actual=("todays_gen_act", "sum"),
        gen_programmed=("todays_gen_prgm", "sum"),
        capacity=("monitored_capacity", "sum"),
    )
    .reset_index()
)

re_states = re[re["state_name"] != "All India"][["date", "state_name", "wind_energy", "solar_energy", "other_renewable_energy", "total_renewable_energy"]]
daily_state = daily_state.merge(re_states, on=["date", "state_name"], how="left")
daily_state["total_gen_with_re"] = daily_state["gen_actual"] + daily_state["total_renewable_energy"].fillna(0)

print(f"✅ Daily State: {daily_state.shape}")
print(f"Rows with renewable data: {daily_state['total_renewable_energy'].notna().sum()} ({daily_state['total_renewable_energy'].notna().mean()*100:.1f}%)")


# In[14]:


# ========================================
# 3c. Weekly aggregation
# ========================================
daily_india["year"] = daily_india["date"].dt.year
daily_india["week"] = daily_india["date"].dt.isocalendar().week.astype(int)

weekly = (
    daily_india.groupby(["year", "week"])
    .agg(
        week_start=("date", "min"),
        avg_daily_gen=("total_gen_actual", "mean"),
        max_daily_gen=("total_gen_actual", "max"),
        min_daily_gen=("total_gen_actual", "min"),
        volatility=("total_gen_actual", "std"),
        avg_renewable=("total_renewable_energy", "mean"),
        avg_renewable_pct=("renewable_share_pct", "mean"),
    )
    .reset_index()
    .sort_values(["year", "week"])
)

print(f"✅ Weekly: {weekly.shape}")
weekly.tail()


# In[10]:


# ========================================
# 3d. Monthly aggregation
# ========================================
daily_india["month_start"] = daily_india["date"].dt.to_period("M").dt.to_timestamp()

monthly = (
    daily_india.groupby("month_start")
    .agg(
        avg_daily_gen=("total_gen_actual", "mean"),
        total_gen=("total_gen_actual", "sum"),
        avg_renewable=("total_renewable_energy", "mean"),
        avg_renewable_pct=("renewable_share_pct", "mean"),
        days=("date", "count"),
    )
    .reset_index()
)

print(f"✅ Monthly: {monthly.shape}")
monthly.tail(12)


# In[11]:


# ========================================
# 3e. Station-level summary
# ========================================
station_summary = (
    pg.groupby(["power_station", "state_name", "station_type", "sector", "region"])
    .agg(
        total_gen=("todays_gen_act", "sum"),
        avg_daily_gen=("todays_gen_act", "mean"),
        avg_capacity_mw=("monitored_capacity", "mean"),
        days_active=("date", "nunique"),
        first_seen=("date", "min"),
        last_seen=("date", "max"),
    )
    .reset_index()
    .sort_values("total_gen", ascending=False)
)

print(f"✅ Station summary: {station_summary.shape[0]} stations")
station_summary.head(15)


# ## 4. Time Series Plots

# ### 4a. Daily Generation — Full Timeline

# In[15]:


fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

# Total generation
axes[0].plot(daily_india["date"], daily_india["total_gen_actual"], linewidth=0.5, alpha=0.7, label="Conventional")
axes[0].plot(daily_india["date"], daily_india["total_renewable_energy"], linewidth=0.5, alpha=0.7, color="green", label="Renewable")
axes[0].set_ylabel("Generation (MU)")
axes[0].set_title("All India Daily Power Generation")
axes[0].legend()

# Deviation
axes[1].fill_between(daily_india["date"], daily_india["gen_deviation_pct"], 0, alpha=0.5,
                     where=daily_india["gen_deviation_pct"] >= 0, color="green", label="Over-gen")
axes[1].fill_between(daily_india["date"], daily_india["gen_deviation_pct"], 0, alpha=0.5,
                     where=daily_india["gen_deviation_pct"] < 0, color="red", label="Under-gen")
axes[1].set_ylabel("Deviation %")
axes[1].set_title("Actual vs Programmed Deviation")
axes[1].legend()

# Renewable share
re_mask = daily_india["renewable_share_pct"].notna()
axes[2].plot(daily_india.loc[re_mask, "date"], daily_india.loc[re_mask, "renewable_share_pct"],
             linewidth=0.5, color="green")
axes[2].set_ylabel("Renewable %")
axes[2].set_title("Renewable Share of Total Generation")
axes[2].xaxis.set_major_locator(mdates.YearLocator())
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.show()


# ### 4b. Weekly Volatility

# In[16]:


fig, ax = plt.subplots(figsize=(18, 5))
ax.bar(weekly["week_start"], weekly["volatility"], width=5, alpha=0.7, color="coral")
ax.set_ylabel("Std Dev (MU)")
ax.set_title("Weekly Generation Volatility")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.show()


# ### 4c. Monthly Trend + Renewable Overlay

# In[17]:


fig, ax1 = plt.subplots(figsize=(18, 6))

ax1.bar(monthly["month_start"], monthly["avg_daily_gen"], width=25, alpha=0.6, label="Avg Daily Conv. Gen", color="steelblue")
ax1.set_ylabel("Avg Daily Generation (MU)", color="steelblue")

ax2 = ax1.twinx()
re_mask = monthly["avg_renewable_pct"].notna()
ax2.plot(monthly.loc[re_mask, "month_start"], monthly.loc[re_mask, "avg_renewable_pct"],
         color="green", linewidth=2, marker="o", markersize=3, label="Renewable %")
ax2.set_ylabel("Renewable Share %", color="green")

ax1.set_title("Monthly Generation + Renewable Share")
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
plt.tight_layout()
plt.show()


# ### 4d. Day-of-Week Pattern

# In[18]:


daily_india["dow"] = daily_india["date"].dt.dayofweek  # 0=Mon
dow_avg = daily_india.groupby("dow")["total_gen_actual"].agg(["mean", "std"]).reset_index()
dow_avg["label"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(dow_avg["label"], dow_avg["mean"], yerr=dow_avg["std"], capsize=5, color="steelblue", alpha=0.8)
ax.set_ylabel("Avg Daily Gen (MU)")
ax.set_title("Day-of-Week Generation Pattern (All India)")
plt.tight_layout()
plt.show()


# ### 4e. Station Type & Sector Breakdown

# In[19]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

by_type = station_summary.groupby("station_type")["total_gen"].sum().sort_values(ascending=False)
axes[0].barh(by_type.index, by_type.values, color=sns.color_palette("muted", len(by_type)))
axes[0].set_xlabel("Total Generation (MU)")
axes[0].set_title("By Station Type")

by_sector = station_summary.groupby("sector")["total_gen"].sum().sort_values(ascending=False)
axes[1].barh(by_sector.index, by_sector.values, color=sns.color_palette("Set2", len(by_sector)))
axes[1].set_xlabel("Total Generation (MU)")
axes[1].set_title("By Sector")

plt.tight_layout()
plt.show()


# ### 4f. State × Month Heatmap

# In[20]:


state_monthly = (
    daily_state.assign(month_start=daily_state["date"].dt.to_period("M").dt.to_timestamp())
    .groupby(["month_start", "state_name"])["gen_actual"].sum()
    .reset_index()
)

top_states = state_monthly.groupby("state_name")["gen_actual"].sum().nlargest(15).index
pivot = state_monthly[state_monthly["state_name"].isin(top_states)].pivot_table(
    index="state_name", columns="month_start", values="gen_actual", aggfunc="sum"
)

fig, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(pivot, cmap="YlOrRd", ax=ax, xticklabels=6)
ax.set_title("State × Month Generation Heatmap (Top 15)")
plt.tight_layout()
plt.show()


# ## 5. Chronos Forecasting
# Amazon Chronos-T5-Small (~46M params, ~200MB RAM).
# 
# ```bash
# pip install amazon-chronos[inference] torch
# ```
# 

# In[41]:


"""import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="auto",
    dtype=torch.float32,  # float16 if GPU available
)
print("✅ Chronos loaded")
"""

from chronos import BaseChronosPipeline
import torch

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="auto",
    torch_dtype=torch.float32,
)
print("✅ Chronos loaded")

ts_df = daily_india[["date", "total_gen_actual"]].rename(
    columns={"date": "timestamp", "total_gen_actual": "target"}
)
ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])
ts_df = ts_df.sort_values("timestamp")
ts_df = ts_df.set_index("timestamp").asfreq("D").reset_index()
ts_df["target"] = ts_df["target"].interpolate()
ts_df["id"] = "all_india"

pred_df = pipeline.predict_df(
    ts_df,
    prediction_length=30,
    quantile_levels=[0.025, 0.5, 0.975], #quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target",
)

print(pred_df.columns.tolist())
print(pred_df.head())


# In[34]:


print(pred_df.columns.tolist())
print(pred_df.head())


# ### 5a. All India — 30 Day Forecast

# In[43]:


"""ts_data = daily_india.sort_values("date")["total_gen_actual"].values
context = torch.tensor(ts_data, dtype=torch.float32)

PREDICTION_LENGTH = 30

#forecast = pipeline.predict(context=context, prediction_length=PREDICTION_LENGTH, num_samples=20)
forecast = pipeline.predict(inputs=context, prediction_length=PREDICTION_LENGTH, num_samples=20)

forecast_median = forecast.median(dim=1).squeeze().numpy()
forecast_low = forecast.quantile(0.1, dim=1).squeeze().numpy()
forecast_high = forecast.quantile(0.9, dim=1).squeeze().numpy()

last_date = daily_india["date"].max()
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=PREDICTION_LENGTH)
"""
# Plot
"""
fig, ax = plt.subplots(figsize=(18, 6))

recent = daily_india.sort_values("date").tail(90)
ax.plot(recent["date"], recent["total_gen_actual"], color="steelblue", linewidth=1.5, label="Actual")
ax.plot(forecast_dates, forecast_median, color="red", linewidth=2, label="Chronos Forecast (median)")
ax.fill_between(forecast_dates, forecast_low, forecast_high, alpha=0.2, color="red", label="95% CI")#80% 
ax.axvline(x=last_date, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("Daily Generation (MU)")
ax.set_title("All India Power Generation — 30-Day Forecast (Chronos-T5-Small)")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""

fig, ax = plt.subplots(figsize=(18, 6))
recent = daily_india.sort_values("date").tail(90)
ax.plot(recent["date"], recent["total_gen_actual"], color="steelblue", label="Actual")
ax.plot(pred_df["timestamp"], pred_df["0.5"], color="red", linewidth=2, label="Forecast (median)")
ax.fill_between(pred_df["timestamp"], pred_df["0.025"], pred_df["0.975"], alpha=0.2, color="red", label="95% CI")#80% 
ax.axvline(x=recent["date"].max(), color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("Daily Generation (MU)")
ax.set_title("All India 30-Day Forecast (Chronos-T5-Small)")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### 5a.a Train-test splitting

# In[44]:


split_idx = int(len(ts_df) * 0.9)
train_df = ts_df.iloc[:split_idx].copy()
test_df = ts_df.iloc[split_idx:].copy()

# Cap at 64 as recommended
pred_length = min(len(test_df), 64)
test_df = test_df.iloc[:pred_length]

print(f"Train: {train_df.timestamp.min().date()} → {train_df.timestamp.max().date()} ({len(train_df)} days)")
print(f"Test:  {test_df.timestamp.min().date()} → {test_df.timestamp.max().date()} ({len(test_df)} days)")

pred_val = pipeline.predict_df(
    train_df,
    prediction_length=pred_length,
    quantile_levels=[0.025, 0.5, 0.975], #quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target",
)

from sklearn.metrics import mean_absolute_error, mean_squared_error

actual = test_df["target"].values
predicted = pred_val["0.5"].values[:len(test_df)]

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = (abs(actual - predicted) / actual).mean() * 100

print(f"\n📊 Conventional Generation Backtest:")
print(f"  MAE:  {mae:.2f} MU")
print(f"  RMSE: {rmse:.2f} MU")
print(f"  MAPE: {mape:.2f}%")

fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(train_df["timestamp"].tail(60), train_df["target"].tail(60), color="steelblue", label="Train")
ax.plot(test_df["timestamp"], test_df["target"], color="black", linewidth=2, label="Actual (test)")
ax.plot(pred_val["timestamp"], pred_val["0.5"], color="red", linewidth=2, label="Forecast")
ax.fill_between(pred_val["timestamp"], pred_val["0.025"], pred_val["0.975"], alpha=0.2, color="red", label="95% CI") #80% 
ax.axvline(x=train_df["timestamp"].max(), color="gray", linestyle="--")
ax.set_title("Backtest — Conventional Generation")
ax.set_ylabel("Daily Gen (MU)")
ax.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# ========================================
# BACKTEST VALIDATION — All India Conventional
# ========================================
split_idx = int(len(ts_df) * 0.9)
train_df = ts_df.iloc[:split_idx].copy()
test_df = ts_df.iloc[split_idx:].copy()

print(f"Train: {train_df.timestamp.min().date()} → {train_df.timestamp.max().date()} ({len(train_df)} days)")
print(f"Test:  {test_df.timestamp.min().date()} → {test_df.timestamp.max().date()} ({len(test_df)} days)")

pred_val = pipeline.predict_df(
    train_df,
    prediction_length=len(test_df),
    quantile_levels=[0.025, 0.5, 0.975], # quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target",
)

from sklearn.metrics import mean_absolute_error, mean_squared_error

actual = test_df["target"].values
predicted = pred_val["0.5"].values[:len(test_df)]

mae = mean_absolute_error(actual, predicted)
rmse = mean_squared_error(actual, predicted, squared=False)
mape = (abs(actual - predicted) / actual).mean() * 100

print(f"\n📊 Conventional Generation Backtest:")
print(f"  MAE:  {mae:.2f} MU")
print(f"  RMSE: {rmse:.2f} MU")
print(f"  MAPE: {mape:.2f}%")

fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(train_df["timestamp"].tail(60), train_df["target"].tail(60), color="steelblue", label="Train")
ax.plot(test_df["timestamp"], test_df["target"], color="black", linewidth=2, label="Actual (test)")
ax.plot(pred_val["timestamp"], pred_val["0.5"], color="red", linewidth=2, label="Forecast")
ax.fill_between(pred_val["timestamp"], pred_val["0.1"], pred_val["0.9"], alpha=0.2, color="red", label="95% CI") #80% 
ax.axvline(x=train_df["timestamp"].max(), color="gray", linestyle="--")
ax.set_title("Backtest — Conventional Generation")
ax.set_ylabel("Daily Gen (MU)")
ax.legend()
plt.tight_layout()
plt.show()


# ## 5.a.b Renewable energy backtesting

# In[51]:


# ========================================
# RENEWABLE — Prep + Backtest
# ========================================
re_india = daily_india[["date", "total_renewable_energy"]].dropna().rename(
    columns={"date": "timestamp", "total_renewable_energy": "target"}
)
re_india["timestamp"] = pd.to_datetime(re_india["timestamp"])
re_india = re_india.sort_values("timestamp")
re_india = re_india.set_index("timestamp").asfreq("D").reset_index()
re_india["target"] = re_india["target"].interpolate()
re_india["id"] = "all_india_renewable"

print(f"Renewable series: {re_india.timestamp.min().date()} → {re_india.timestamp.max().date()} ({len(re_india)} days)")

# Split
re_split = int(len(re_india) * 0.9)
re_train = re_india.iloc[:re_split].copy()
re_test = re_india.iloc[re_split:].copy()

re_pred_length = min(len(re_test), 64)
re_test = re_test.iloc[:re_pred_length]

print(f"Train: {re_train.timestamp.min().date()} → {re_train.timestamp.max().date()} ({len(re_train)} days)")
print(f"Test:  {re_test.timestamp.min().date()} → {re_test.timestamp.max().date()} ({len(re_test)} days)")

re_pred_val = pipeline.predict_df(
    re_train,
    prediction_length=re_pred_length,
    quantile_levels=[0.025, 0.5, 0.975],
    id_column="id",
    timestamp_column="timestamp",
    target="target",
)

re_actual = re_test["target"].values
re_predicted = re_pred_val["0.5"].values[:len(re_test)]

re_mae = mean_absolute_error(re_actual, re_predicted)
re_rmse = np.sqrt(mean_squared_error(re_actual, re_predicted))
re_mape = (abs(re_actual - re_predicted) / re_actual.clip(min=0.01)).mean() * 100

print(f"\n📊 Renewable Generation Backtest:")
print(f"  MAE:  {re_mae:.2f} MU")
print(f"  RMSE: {re_rmse:.2f} MU")
print(f"  MAPE: {re_mape:.2f}%")

fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(re_train["timestamp"].tail(240), re_train["target"].tail(240), color="green", label="Train")
ax.plot(re_test["timestamp"], re_test["target"], color="black", linewidth=2, label="Actual (test)")
ax.plot(re_pred_val["timestamp"], re_pred_val["0.5"], color="red", linewidth=2, label="Forecast")
ax.fill_between(re_pred_val["timestamp"], re_pred_val["0.025"], re_pred_val["0.975"], alpha=0.2, color="red", label="95% CI")
ax.axvline(x=re_train["timestamp"].max(), color="gray", linestyle="--")
ax.set_title("Backtest — Renewable Generation")
ax.set_ylabel("Daily Gen (MU)")
ax.legend()
plt.tight_layout()
plt.show()


# ## 5.a.c Combined view: convntional + renewable

# In[48]:


re_forecast = pipeline.predict_df(
    re_india,
    prediction_length=30,
    quantile_levels=[0.025, 0.5, 0.975],
    id_column="id",
    timestamp_column="timestamp",
    target="target",
)


# In[49]:


# ========================================
# COMBINED VIEW — Conv + Renewable forecasts
# ========================================
fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

# Conventional
axes[0].plot(daily_india["date"].tail(90), daily_india["total_gen_actual"].tail(90), color="steelblue", label="Actual")
axes[0].plot(pred_df["timestamp"], pred_df["0.5"], color="red", linewidth=2, label="Forecast")
axes[0].fill_between(pred_df["timestamp"], pred_df["0.025"], pred_df["0.975"], alpha=0.2, color="red", label="95% CI")
axes[0].set_ylabel("MU")
axes[0].set_title("Conventional Generation — 30 Day Forecast")
axes[0].legend()

# Renewable
axes[1].plot(re_india["timestamp"].tail(90), re_india["target"].tail(90), color="green", label="Actual")
axes[1].plot(re_forecast["timestamp"], re_forecast["0.5"], color="red", linewidth=2, label="Forecast")
axes[1].fill_between(re_forecast["timestamp"], re_forecast["0.025"], re_forecast["0.975"], alpha=0.2, color="red", label="95% CI")
axes[1].set_ylabel("MU")
axes[1].set_title("Renewable Generation — 30 Day Forecast")
axes[1].legend()

plt.tight_layout()
plt.show()


# ### 5b. State-level Forecast (Maharashtra)

# In[37]:


TARGET_STATE = "Maharashtra"

state_ts = daily_state[daily_state["state_name"] == TARGET_STATE][["date", "gen_actual"]].rename(
    columns={"date": "timestamp", "gen_actual": "target"}
)
state_ts["timestamp"] = pd.to_datetime(state_ts["timestamp"])
state_ts = state_ts.sort_values("timestamp")
state_ts = state_ts.set_index("timestamp").asfreq("D").reset_index()
state_ts["target"] = state_ts["target"].interpolate()
state_ts["id"] = TARGET_STATE

pred_state = pipeline.predict_df(
    state_ts,
    prediction_length=30,
    quantile_levels=[0.025, 0.5, 0.975], #quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target",
)

fig, ax = plt.subplots(figsize=(18, 5))
recent_st = state_ts.tail(90)
ax.plot(recent_st["timestamp"], recent_st["target"], color="steelblue", linewidth=1.5, label="Actual")
ax.plot(pred_state["timestamp"], pred_state["0.5"], color="red", linewidth=2, label="Forecast (median)")
ax.fill_between(pred_state["timestamp"], pred_state["0.1"], pred_state["0.9"], alpha=0.2, color="red", label="95% CI") #80% CI
ax.axvline(x=recent_st["timestamp"].max(), color="gray", linestyle="--", alpha=0.5)
ax.set_title(f"{TARGET_STATE} — 30-Day Generation Forecast (Chronos-T5-Small)")
ax.set_ylabel("Daily Gen (MU)")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## 6. Save Processed Data (for Graph Notebook)

# In[52]:


# Save all processed dataframes for next notebook
daily_india.to_csv("processed_daily_india.csv", index=False)
daily_state.to_csv("processed_daily_state.csv", index=False)
station_summary.to_csv("processed_station_summary.csv", index=False)
ds.to_csv("processed_demand_supply.csv", index=False)

print("✅ Saved processed CSVs for graph notebook:")
print("  - processed_daily_india.csv")
print("  - processed_daily_state.csv")
print("  - processed_station_summary.csv")
print("  - processed_demand_supply.csv")


# In[ ]:




