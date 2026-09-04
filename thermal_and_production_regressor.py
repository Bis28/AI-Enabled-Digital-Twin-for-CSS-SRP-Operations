import pandas as pd
from pycaret.regression import setup, compare_models, finalize_model, save_model

# 1. Load the dataset
data = pd.read_csv("SIH26120_CSS_SRP_Training_Data.csv")

# 2. Drop non-predictive identifiers and classification target flags for this run
reg_data = data.drop(columns=["Well_ID", "Failure_Rod_Float", "Failure_Pump_Unset"])

# 3. Initialize PyCaret setup (Target: Oil_Rate_bpd)
reg_setup = setup(
    data=reg_data,
    target="Oil_Rate_bpd",
    train_size=0.8,
    session_id=42,
    verbose=True
)

# 4. Automatically train and rank all ML algorithms (XGBoost, Random Forest, etc.)
best_reg_model = compare_models()

# 5. Finalize and save the best performing model pipeline
final_reg = finalize_model(best_reg_model)
save_model(final_reg, "best_oil_production_model")
print("--> Regression Model Successfully Trained and Saved!")
