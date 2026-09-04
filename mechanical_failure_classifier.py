import pandas as pd
from pycaret.classification import setup, compare_models, finalize_model, save_model

# 1. Load the dataset
data = pd.read_csv("SIH26120_CSS_SRP_Training_Data.csv")

# 2. Drop non-predictive columns and the other targets
clf_data = data.drop(columns=["Well_ID", "Oil_Rate_bpd", "Failure_Pump_Unset"])

# 3. Initialize PyCaret setup (Target: Failure_Rod_Float)
clf_setup = setup(
    data=clf_data,
    target="Failure_Rod_Float",
    train_size=0.8,
    session_id=42,
    verbose=True
)

# 4. Automatically train and evaluate classification algorithms
best_clf_model = compare_models()

# 5. Finalize and save the best failure detection model
final_clf = finalize_model(best_clf_model)
save_model(final_clf, "best_rod_floating_model")
print("--> Classification Model Successfully Trained and Saved!")

