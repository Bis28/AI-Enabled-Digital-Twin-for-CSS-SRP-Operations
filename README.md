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
