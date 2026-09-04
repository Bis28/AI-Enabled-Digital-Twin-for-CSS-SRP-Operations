import pandas as pd
import numpy as np
from pycaret.regression import setup as setup_reg, create_model as create_reg, finalize_model as finalize_reg, save_model as save_reg
from pycaret.classification import setup as setup_clf, create_model as create_clf, finalize_model as finalize_clf, save_model as save_clf

print("1. Fixing Thermodynamic Data...")
df = pd.read_csv("SIH26120_CSS_SRP_Training_Data.csv")

# Inject realistic heavy oil temperature cooling and exponential viscosity
np.random.seed(42)
df['Prod_Temp_C'] = np.random.uniform(40, 180, len(df)).round(1)
# Corrected Andrade's formula for Heavy Crude Oil
df['Oil_Viscosity_cP'] = (0.005 * np.exp(4700 / (df['Prod_Temp_C'] + 273.15))).round(1)

# Recalculate failure risk (Viscosity > 4000 + High SPM = Failure)
float_prob = np.clip(((df['Oil_Viscosity_cP'] - 4000) / 5000) * (df['VFD_SPM'] / 4.0), 0, 0.95)
df['Failure_Rod_Float'] = (np.random.rand(len(df)) < float_prob).astype(int)

# Overwrite the CSV so it is permanently fixed
df.to_csv("SIH26120_CSS_SRP_Training_Data.csv", index=False)
print(f"Fixed! Found {df['Failure_Rod_Float'].sum()} failure cases out of 5000.")

print("2. Rapidly Retraining AI Models (Random Forest)...")
# Retrain Regression
reg_data = df.drop(columns=["Well_ID", "Failure_Rod_Float", "Failure_Pump_Unset"])
setup_reg(data=reg_data, target="Oil_Rate_bpd", session_id=42, verbose=False)
reg_model = create_reg('rf', verbose=False)
save_reg(finalize_reg(reg_model), "best_oil_production_model")

# Retrain Classification
clf_data = df.drop(columns=["Well_ID", "Oil_Rate_bpd", "Failure_Pump_Unset"])
setup_clf(data=clf_data, target="Failure_Rod_Float", session_id=42, verbose=False)
clf_model = create_clf('rf', verbose=False)
save_clf(finalize_clf(clf_model), "best_rod_floating_model")

print("--> Models successfully retrained with failure parameters! You can close this script.")


