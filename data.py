import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🔧 PREDICTIVE MAINTENANCE FOR POWER PLANTS")
print("="*70)

# ============================================================================
# STEP 1: CREATE REALISTIC POWER PLANT DATA (NO DOWNLOADS NEEDED)
# ============================================================================

print("\n📊 STEP 1: CREATING REALISTIC POWER PLANT DATA...")

np.random.seed(42)
n_hours = 4 * 365 * 24  # 4 years of hourly data

# Create realistic gas turbine data based on real plant ranges
df = pd.DataFrame()

# Time features
df['hour'] = np.arange(n_hours)
df['day_of_year'] = (df['hour'] // 24) % 365

# Ambient conditions (with seasonal patterns)
df['AT'] = 15 + 10 * np.sin(2 * np.pi * df['day_of_year'] / 365) + np.random.normal(0, 2, n_hours)  # Ambient Temp
df['AP'] = 1013 + 5 * np.random.randn(n_hours)  # Ambient Pressure
df['AH'] = 70 + 15 * np.sin(2 * np.pi * df['day_of_year'] / 365) + np.random.normal(0, 5, n_hours)  # Humidity

# Turbine measurements
df['TIT'] = 1050 + 20 * np.sin(2 * np.pi * df['hour'] / 8760) + np.random.normal(0, 5, n_hours)  # Turbine Inlet Temp
df['TAT'] = 530 + 10 * np.sin(2 * np.pi * df['hour'] / 8760) + np.random.normal(0, 3, n_hours)   # Turbine After Temp
df['CDP'] = 12 + 0.5 * np.sin(2 * np.pi * df['hour'] / 4380) + np.random.normal(0, 1, n_hours)   # Compressor Discharge Pressure
df['GTEP'] = 25 + 3 * np.sin(2 * np.pi * df['hour'] / 8760) + np.random.normal(0, 2, n_hours)    # Exhaust Pressure
df['AFDP'] = 4 + 0.3 * np.sin(2 * np.pi * df['hour'] / 4380) + np.random.normal(0, 0.5, n_hours) # Air Filter Pressure

# Performance
df['TEY'] = 140 - 5 * (df['TIT'] - 1050)/50 + 10 * np.random.randn(n_hours)  # Turbine Energy Yield

# Emissions (what we'll use for failures)
df['CO'] = 10 + 0.1 * (df['TIT'] - 1050) + 2 * np.random.randn(n_hours)      # CO emissions
df['NOX'] = 50 + 0.5 * (df['TIT'] - 1050) + 5 * np.random.randn(n_hours)     # NOx emissions

# Add degradation over time (equipment wears out)
degradation = np.linspace(0, 0.3, n_hours)  # 30% degradation over 4 years
df['TIT'] = df['TIT'] * (1 + degradation * 0.1)
df['CO'] = df['CO'] * (1 + degradation)
df['NOX'] = df['NOX'] * (1 + degradation * 0.8)
df['CDP'] = df['CDP'] * (1 - degradation * 0.05)  # Pressure drops with wear

# Add some random failure events (about 5% of the time)
failure_indices = np.random.choice(n_hours, size=int(n_hours*0.05), replace=False)
df.loc[failure_indices, 'CO'] = df.loc[failure_indices, 'CO'] * 3  # Triple CO during failures
df.loc[failure_indices, 'TIT'] = df.loc[failure_indices, 'TIT'] * 1.1  # Higher temp during failures

print(f"✅ Created {len(df):,} hours of power plant data")
print(f"   Time period: 4 years ({(len(df)/8760):.1f} years)")
print(f"   Features: {list(df.columns)}")

# Show sample
print("\n📋 First 5 rows:")
print(df.head())

df.to_csv('data.csv', index=False)