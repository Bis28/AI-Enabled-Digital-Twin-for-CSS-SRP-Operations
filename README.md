# AI-Enabled Digital Twin for CSS & SRP Operations

This project was developed for the **Smart India Hackathon (SIH) - Oil India Limited (OIL)** Problem Statement SIH26120. It is a robust, physics-driven SCADA dashboard that integrates reservoir thermodynamics, kinematic mechanical constraints, and an Explainable AI (XAI) Copilot.

## Key Features
* **Thermodynamic & Mechanical Soft Sensors:** Dynamically calculates downhole viscosity, thermal decay, and rod stress.
* **Live IoT Telemetry HMI:** A bidirectional, synchronized UI simulating real-time sensor streams.
* **Process Economics Engine:** Calculates operational OPEX, actively penalizing dangerous parameters like Fluid Pound or Rod Floating.
* **Llama 3.2 AI Copilot:** A zero-hallucination industrial dispatcher that enforces deterministic mechanical safety overrides.

## Tech Stack
* **Frontend/UI:** Streamlit, Plotly
* **Physics Engine:** Python (NumPy, Pandas)
* **Machine Learning:** PyCaret (Production Forecasting & Risk Classification)
* **Generative AI:** Ollama (Llama 3.2)

## How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure [Ollama](https://ollama.com/) is installed and running locally with the `llama3.2` model.
4. Run the dashboard: `streamlit run app.py`

## How This Architecture Solves the OIL Problem Statement

Oil India Limited (OIL) identified a critical operational gap in the Baghewala Field: **Cyclic Steam Stimulation (CSS) and Sucker Rod Pump (SRP) operations are currently optimized separately.** As the reservoir cools, heavy crude viscosity spikes, leading to severe mechanical failures (rod floating, fluid pound) and poor energy efficiency. 

This Digital Twin eliminates that gap by unifying thermodynamic reservoir modeling with mechanical surface pumping limits into a single deterministic engine.

### 1. Unifying CSS & SRP via Thermodynamic Soft Sensors
Instead of reacting to wellbore cooling after the fact, the engine mathematically infers downhole conditions in real-time. 
*   **Predicting Reservoir Heating/Cooling:** The `est_prod_temp` soft sensor calculates thermal decay based on the injection volume, CSS cycle degradation, and soaking duration.
*   **Dynamic Viscosity Calculation:** The inferred temperature is fed into an Arrhenius-type exponential equation to calculate real-time fluid thickness:
    $$ \mu = 0.005 \exp\left(\frac{4700.0}{T_{prod} + 273.15}\right) $$
*   **Result:** The system continuously links surface steam injection (CSS) directly to the dynamic state of the underground fluid.

### 2. Kinematic Constraint Modeling & Rod Failure Prevention
The heavy crude of the Jodhpur Sandstone reservoir causes impact loading and rod floating. The Digital Twin physically models these constraints to prevent equipment failure before it happens.
*   **Load Prediction:** Peak Polished Rod Load (PPRL) is calculated dynamically: `pprl_est = 8000.0 + (stroke_len * 70.0) + (est_viscosity * vfd_spm * 0.6)`. This proves how viscous drag impacts rod stress.
*   **Rod Floating & Fluid Pound Detection:** The application explicitly flags **Rod Floating** when thick oil ($\mu > 2500$ cP) prevents the rod from falling on the downstroke at high speeds ($>5.0$ SPM). Conversely, it flags **Fluid Pound** when the pump barrel empties too fast.
*   **Result:** SRP operating parameters (SPM and VFD settings) are instantly optimized by the system's strict mechanical safety overrides, minimizing impact loading.

### 3. Process Economics & Energy Optimization
To address the high Steam-Oil Ratio (SOR) and reduce operating costs, the dashboard includes a live Process Economics engine.
*   **Data-Driven Decision Making:** The system predicts gross revenue against dynamic lifting costs and steam generation OPEX. 
*   **Risk/Wear Penalties:** If the operator pushes the VFD speed into a kinematic anomaly (e.g., Fluid Pound), the engine injects a severe financial maintenance penalty into the daily OPEX.
*   **Result:** The AI Copilot and the human operator are both financially incentivized to optimize steam energy consumption and reduce the SOR, directly maximizing net profit and equipment life.

---
*Developed by Bishes Sarkar (Dept. of Chemical Engineering, NIT Durgapur).*
