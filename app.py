import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ollama
import time

# --- Safely Import PyCaret ---
try:
    from pycaret.regression import load_model as load_reg_model, predict_model as predict_reg
    from pycaret.classification import load_model as load_clf_model, predict_model as predict_clf
    pycaret_available = True
except ImportError:
    pycaret_available = False

# --- Page Configuration ---
st.set_page_config(page_title="Pump BGH-138 Real-Time Health", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Industrial SCADA Theme & Custom Loading Spinner ---
st.markdown("""
    <style>
    section[data-testid="stSidebar"] > div:first-child { padding-top: 0rem !important; }
    .block-container { padding-top: 3rem; padding-bottom: 2rem; max-width: 96%; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem; color: #38bdf8; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-weight: bold; font-size: 1.1rem;}
    h3 { color: #f8fafc; font-weight: 400; margin-bottom: 1rem; margin-top: 0rem; }
    
    .stSpinner > div { border-top-color: #38bdf8 !important; }
    
    .alert-box { background-color: #27272a; padding: 15px; border-radius: 5px; border-left: 5px solid #ef4444; margin-bottom: 5px;}
    .alert-box-safe { background-color: #27272a; padding: 15px; border-radius: 5px; border-left: 5px solid #22c55e; margin-bottom: 5px;}
    </style>
""", unsafe_allow_html=True)

# --- Prevent Auto-Scroll to Bottom on Load (Observer Fix) ---
st.markdown("""
    <script>
        const observer = new MutationObserver((mutations, obs) => {
            const chatInput = document.querySelector('[data-testid="stChatInput"]');
            if (chatInput) {
                window.scrollTo({ top: 0, behavior: 'instant' });
            }
        });
        observer.observe(document, { childList: true, subtree: true });
        
        // Immediate fallback
        window.onload = function() {
            window.scrollTo(0, 0);
        };
    </script>
""", unsafe_allow_html=True)

# --- Model Loading ---
models_loaded = False
if pycaret_available:
    try:
        reg_model = load_reg_model("best_oil_production_model")
        clf_model = load_clf_model("best_rod_floating_model")
        models_loaded = True
    except Exception:
        models_loaded = False

# --- Top Nav Bar ---
col_head1, col_head2 = st.columns([5, 1])
with col_head1:
    st.markdown("### Pump BGH-138 Real-Time Health & AI Supervisory System")
with col_head2:
    if models_loaded:
        st.markdown("<p style='text-align:right; color:#22c55e; padding-top:10px;'>AI SCADA Online</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align:right; color:#fbbf24; padding-top:10px;' Simulation Mode</p>", unsafe_allow_html=True)

# --- Sidebar Controls ---
# --- Sidebar Controls ---
# Removed the broken st.sidebar.image() call. Using tight negative margins to pull text up.
st.sidebar.markdown("<h2 style='margin-top:-20px; margin-bottom:0px;'>Control Panel</h2>", unsafe_allow_html=True)

# --- State Management for IoT Streaming ---
if 'iot_streaming' not in st.session_state:
    st.session_state.iot_streaming = False
if 'base_temp' not in st.session_state:
    st.session_state.base_temp = 280.0
if 'base_vol' not in st.session_state:
    st.session_state.base_vol = 15000
if 'base_spm' not in st.session_state:
    st.session_state.base_spm = 5.2

# Tighter horizontal rule and IoT header
st.sidebar.markdown("<hr style='margin-top:10px; margin-bottom:10px;'>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#38bdf8; font-weight:bold; margin-bottom:0px;'>IoT Sensor Array</p>", unsafe_allow_html=True)
iot_toggle = st.sidebar.toggle("Enable Live Telemetry Stream", value=st.session_state.iot_streaming)
st.session_state.iot_streaming = iot_toggle
with st.spinner("Synchronizing SCADA Telemetry & Physics Engine..."):
    
    # 1. Initialize widget keys on first load to prevent errors
    if "temp_num" not in st.session_state: st.session_state.temp_num = st.session_state.base_temp
    if "temp_slider" not in st.session_state: st.session_state.temp_slider = st.session_state.base_temp
    
    if "vol_num" not in st.session_state: st.session_state.vol_num = st.session_state.base_vol
    if "vol_slider" not in st.session_state: st.session_state.vol_slider = st.session_state.base_vol
    
    if "spm_num" not in st.session_state: st.session_state.spm_num = st.session_state.base_spm
    if "spm_slider" not in st.session_state: st.session_state.spm_slider = st.session_state.base_spm

    # 2. Callbacks: The widgets now strictly update EACH OTHER's internal state
    def sync_t_num():
        st.session_state.temp_slider = st.session_state.temp_num
        st.session_state.base_temp = st.session_state.temp_num
    def sync_t_sli():
        st.session_state.temp_num = st.session_state.temp_slider
        st.session_state.base_temp = st.session_state.temp_slider
        
    def sync_v_num():
        st.session_state.vol_slider = st.session_state.vol_num
        st.session_state.base_vol = st.session_state.vol_num
    def sync_v_sli():
        st.session_state.vol_num = st.session_state.vol_slider
        st.session_state.base_vol = st.session_state.vol_slider
        
    def sync_s_num():
        st.session_state.spm_slider = st.session_state.spm_num
        st.session_state.base_spm = st.session_state.spm_num
    def sync_s_sli():
        st.session_state.spm_num = st.session_state.spm_slider
        st.session_state.base_spm = st.session_state.spm_slider

    # --- Steam Temp Dual-Control ---
    st.sidebar.markdown("<small style='color:#94a3b8; font-weight:bold;'>Steam Temp Setpoint (°C)</small>", unsafe_allow_html=True)
    st.sidebar.number_input("Temp Num", 220.0, 340.0, key="temp_num", on_change=sync_t_num, step=1.0, label_visibility="collapsed")
    st.sidebar.slider("Temp Sli", 220.0, 340.0, key="temp_slider", on_change=sync_t_sli, step=1.0, label_visibility="collapsed")
    
    st.sidebar.markdown("<hr style='margin-top:5px; margin-bottom:5px; border-color:#334155;'>", unsafe_allow_html=True)

    # --- Steam Volume Dual-Control ---
    st.sidebar.markdown("<small style='color:#94a3b8; font-weight:bold;'>Steam Volume Setpoint (bbl)</small>", unsafe_allow_html=True)
    st.sidebar.number_input("Vol Num", 8000, 25000, key="vol_num", on_change=sync_v_num, step=500, label_visibility="collapsed")
    st.sidebar.slider("Vol Sli", 8000, 25000, key="vol_slider", on_change=sync_v_sli, step=500, label_visibility="collapsed")

    st.sidebar.markdown("<hr style='margin-top:5px; margin-bottom:5px; border-color:#334155;'>", unsafe_allow_html=True)

    # --- Pump Speed Dual-Control ---
    st.sidebar.markdown("<small style='color:#94a3b8; font-weight:bold;'>Pump Speed Setpoint (SPM)</small>", unsafe_allow_html=True)
    st.sidebar.number_input("SPM Num", 2.0, 9.0, key="spm_num", on_change=sync_s_num, step=0.1, label_visibility="collapsed")
    st.sidebar.slider("SPM Sli", 2.0, 9.0, key="spm_slider", on_change=sync_s_sli, step=0.1, label_visibility="collapsed")

    # --- Static Variables ---
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    soak_days = st.sidebar.slider("Soaking Duration (Days)", 2, 20, 8, 1)
    css_cycle = st.sidebar.selectbox("Current CSS Cycle", [1, 2, 3, 4, 5], index=2)
    stroke_len = st.sidebar.selectbox("Stroke Length (inches)", [54, 64, 74, 86, 100], index=2)
# --- Reconnect UI to the Physics Engine ---
    if iot_toggle:
        # Calculate the noisy Process Variables (PV) for the math engine
        steam_temp = st.session_state.base_temp + np.random.normal(0, 1.2)
        steam_vol = st.session_state.base_vol + int(np.random.normal(0, 150))
        vfd_spm = max(2.0, st.session_state.base_spm + np.random.normal(0, 0.15))
        
        # Display the live fluctuating readings cleanly below the sliders
        st.sidebar.markdown(f"""
        <div style='background-color:#0f172a; padding:10px; border-radius:5px; border-left:3px solid #38bdf8; margin-bottom:15px; margin-top:10px;'>
            <small style='color:#94a3b8; font-weight:bold;'>Live Sensor Array (PV)</small><br>
            <span style='color:#cbd5e1;'>Temp:</span> <span style='color:#38bdf8; font-weight:bold;'>{steam_temp:.1f} °C</span><br>
            <span style='color:#cbd5e1;'>Volume:</span> <span style='color:#38bdf8; font-weight:bold;'>{steam_vol} bbl</span><br>
            <span style='color:#cbd5e1;'>Speed:</span> <span style='color:#38bdf8; font-weight:bold;'>{vfd_spm:.2f} SPM</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        # If off, math engine uses exact setpoints
        steam_temp = st.session_state.base_temp
        steam_vol = st.session_state.base_vol
        vfd_spm = st.session_state.base_spm
# --- Physics Math Engine ---
est_prod_temp = np.clip((steam_temp * 0.45) - (soak_days * 3.0) - (css_cycle * 5.0) + ((steam_vol - 15000) / 1000.0), 15.0, 200.0)
est_viscosity = 0.005 * np.exp(4700.0 / (est_prod_temp + 273.15))
cooling_rate = (1.5 + (css_cycle * 0.5)) * (12000.0 / steam_vol) ** 2
pprl_est = 8000.0 + (stroke_len * 70.0) + (est_viscosity * vfd_spm * 0.6)
mprl_est = max(1000.0, 5000.0 - (est_viscosity * vfd_spm * 0.15))

input_df = pd.DataFrame([{
    "Depth_m": 1200.0, "Permeability_mD": 1450.0, "API_Gravity": 15.2, "Init_Press_psi": 1100.0,
    "CSS_Cycle": css_cycle, "Steam_Temp_C": steam_temp, "Steam_Quality_pct": 78.0, 
    "Steam_Vol_bbl": float(steam_vol), "Soak_Days": soak_days, 
    "Prod_Temp_C": round(est_prod_temp, 1), "Oil_Viscosity_cP": round(est_viscosity, 1),
    "VFD_SPM": vfd_spm, "Stroke_Len_in": stroke_len,
    "PPRL_lbs": round(pprl_est, 1), "MPRL_lbs": round(mprl_est, 1)
}])

# --- Inference Engine ---
if models_loaded:
    try:
        pred_reg = predict_reg(reg_model, data=input_df)
        pred_oil_rate = pred_reg['prediction_label'].iloc[0] if 'prediction_label' in pred_reg.columns else pred_reg.get('Label', pred_reg.iloc[:, -1]).iloc[0]
        
        pred_clf = predict_clf(clf_model, data=input_df)
        model_risk = pred_clf['prediction_label'].iloc[0] if 'prediction_label' in pred_clf.columns else pred_clf.get('Label', pred_clf.iloc[:, -1]).iloc[0]
    except Exception:
        pred_oil_rate = (steam_vol * 0.015 * (1000.0 / max(est_viscosity, 100.0)) * (vfd_spm / 5.0))
        model_risk = 0
else:
    pred_oil_rate = (steam_vol * 0.015 * (1000.0 / max(est_viscosity, 100.0)) * (vfd_spm / 5.0))
    model_risk = 0

# --- Mechanical Safety Override ---
rod_float_risk = 1 if (est_viscosity > 2500 and vfd_spm > 6.0) or (est_viscosity > 4000) else int(model_risk)
sor_metric = round(steam_vol / max(pred_oil_rate * 30.0, 1.0), 2)
health_score = 43 if rod_float_risk == 1 else 94

# =========================================================================
# UI RENDERING: TOP ROW (Schematic | Gauges | Trend Charts)
# =========================================================================
col_schematic, col_gauges, col_charts = st.columns([2.3, 1, 1.5])

with col_schematic:
    with st.container(border=True):
        st.markdown("<p style='color:#94a3b8; font-weight:bold; margin-bottom:0;'>Real-Time Schematic</p>", unsafe_allow_html=True)
        
        fig_schem = go.Figure()
        fig_schem.add_shape(type="line", x0=0, y0=20, x1=100, y1=20, line=dict(color="#475569", width=3))
        fig_schem.add_shape(type="rect", x0=70, y0=0, x1=80, y1=20, line=dict(color="#94a3b8", width=2))
        fig_schem.add_shape(type="path", path="M 40 20 L 45 60 L 35 60 Z", line=dict(color="#94a3b8", width=2))
        fig_schem.add_shape(type="line", x0=20, y0=60, x1=75, y1=60, line=dict(color="#38bdf8", width=6))
        fig_schem.add_shape(type="rect", x0=10, y0=20, x1=25, y1=35, line=dict(color="#94a3b8", width=2))
        fig_schem.add_shape(type="line", x0=17.5, y0=27.5, x1=20, y1=60, line=dict(color="#cbd5e1", width=2, dash="dash"))
        fig_schem.add_shape(type="line", x0=75, y0=60, x1=75, y1=5, line=dict(color="#38bdf8", width=3))

        annotations = [
            dict(x=17.5, y=42, text=f"<b>MOTOR DRIVE</b><br>Speed: {vfd_spm:.2f} SPM<br>Status: {'Warning' if rod_float_risk else 'Normal'}", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor="#0f172a", bordercolor="#ef4444" if rod_float_risk else "#38bdf8", borderwidth=2, font=dict(color="white", size=13, family="Arial")),
            dict(x=75, y=70, text=f"<b>POLISHED ROD</b><br>PPRL: {pprl_est:,.0f} lbs<br>Stroke: {stroke_len} in", showarrow=True, arrowhead=2, ax=45, ay=-35, bgcolor="#0f172a", bordercolor="#38bdf8", borderwidth=2, font=dict(color="white", size=13, family="Arial")),
            dict(x=75, y=10, text=f"<b>DOWNHOLE PUMP</b><br>Viscosity: {est_viscosity:,.0f} cP<br>Temp: {est_prod_temp:.1f} °C", showarrow=True, arrowhead=2, ax=50, ay=25, bgcolor="#0f172a", bordercolor="#ef4444" if est_viscosity > 4000 else "#38bdf8", borderwidth=2, font=dict(color="white", size=13, family="Arial"))
        ]
        fig_schem.update_layout(
            annotations=annotations, template="plotly_dark", 
            xaxis=dict(visible=False, range=[-5, 130]), yaxis=dict(visible=False, range=[-15, 120]),
            height=380, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_schem, use_container_width=True)

with col_gauges:
    with st.container(border=True):
        st.markdown("<p style='color:#94a3b8; font-weight:bold; margin-bottom:0;'>Health Score</p>", unsafe_allow_html=True)
        fig_health = go.Figure(go.Indicator(
            mode="gauge+number", value=health_score, number={'suffix': "%", 'font':{'color':'white', 'size':28}},
            gauge={
                'axis': {'range': [0, 100], 'visible': False},
                'bar': {'color': "#38bdf8" if health_score > 50 else "#ef4444"},
                'bgcolor': "#334155"
            }
        ))
        fig_health.update_layout(height=160, margin=dict(l=15, r=15, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_health, use_container_width=True)
        
        st.markdown("<p style='color:#94a3b8; font-weight:bold; margin-bottom:0;'>Viscosity Threshold</p>", unsafe_allow_html=True)
        
        if est_viscosity > 4000:
            gauge_color = "#ef4444"
        elif est_viscosity > 2500:
            gauge_color = "#f59e0b"
        else:
            gauge_color = "#22c55e"

        fig_visc_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=est_viscosity, 
            number={'suffix': " cP", 'valueformat': ".0f", 'font':{'color':'white', 'size':24}},
            gauge={
                'axis': {'range': [0, 8000], 'visible': False},
                'bar': {'color': gauge_color},
                'bgcolor': "#334155",
                'threshold': {'line': {'color': "orange", 'width': 3}, 'thickness': 1, 'value': 4000}
            }
        ))
        fig_visc_gauge.update_layout(height=160, margin=dict(l=15, r=15, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_visc_gauge, use_container_width=True)

with col_charts:
    with st.container(border=True):
        st.markdown("<p style='color:#94a3b8; font-weight:bold; margin-bottom:0;'>30-Day Diagnostics</p>", unsafe_allow_html=True)
        
        days_array = np.arange(1, 31)
        sim_temps = np.clip(est_prod_temp - (days_array * cooling_rate), 15.0, 200.0)
        sim_visc = 0.005 * np.exp(4700.0 / (sim_temps + 273.15))
        fig_trend = go.Figure(go.Scatter(x=days_array, y=sim_visc, fill='tozeroy', line=dict(color='#38bdf8', width=2)))
        
        fig_trend.update_layout(
            title=dict(text="Thermal Decay", font=dict(size=13, color="#e2e8f0")),
            height=180, margin=dict(l=55, r=15, t=35, b=30), template="plotly_dark",
            xaxis=dict(visible=True, title="Days", title_font=dict(size=11, color='#94a3b8'), tickfont=dict(size=10, color='#64748b'), showgrid=False),
            yaxis=dict(visible=True, title="Viscosity (cP)", title_font=dict(size=11, color='#94a3b8'), tickfont=dict(size=10, color='#64748b'), showgrid=True, gridcolor='#334155', range=[0, min(60000, max(10000, max(sim_visc)))]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        theta = np.linspace(0, 2 * np.pi, 100)
        pos = (stroke_len / 2.0) * (1 - np.cos(theta))
        load = (pprl_est + mprl_est) / 2.0 + ((pprl_est - mprl_est) / 2.0) * np.sin(theta)
        if rod_float_risk == 1: 
            load = load - (load * ((vfd_spm / 9.0) * 0.55) * np.sin(2 * theta))
        
        fig_dyno = go.Figure(go.Scatter(x=pos, y=load, mode='lines', line=dict(color='#38bdf8', width=2)))
        
        fig_dyno.update_layout(
            title=dict(text="Surface Dyno Load", font=dict(size=13, color="#e2e8f0")),
            height=180, margin=dict(l=55, r=15, t=35, b=30), template="plotly_dark",
            xaxis=dict(visible=True, title="Position (in)", title_font=dict(size=11, color='#94a3b8'), tickfont=dict(size=10, color='#64748b'), showgrid=False, range=[-5, 110]),
            yaxis=dict(visible=True, title="Rod Load (lbs)", title_font=dict(size=11, color='#94a3b8'), tickfont=dict(size=10, color='#64748b'), showgrid=True, gridcolor='#334155', range=[0, max(30000, max(load)+2000)]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_dyno, use_container_width=True)

# =========================================================================
# UI RENDERING: BOTTOM ROW (KPIs | Risks | Recommendations)
# =========================================================================
col_kpi, col_risk, col_recom = st.columns([1, 1, 2.5])

with col_kpi:
    with st.container(border=True):
        st.markdown("<p style='color:#94a3b8; font-weight:bold; margin-bottom:0;'>Operational Parameters</p>", unsafe_allow_html=True)
        st.metric("Oil Production", f"{pred_oil_rate:.1f} bpd")
        st.metric("Steam-Oil Ratio", f"{sor_metric}")
        
    with st.container(border=True):
        st.markdown("<p style='color:#94a3b8; font-weight:bold; margin-bottom:0;'>Process Economics (Daily)</p>", unsafe_allow_html=True)
        
        # --- Advanced Economic Math Engine ---
        oil_price_per_bbl = 65.00  # Heavy crude discounted market price
        steam_cost_per_bbl = 18.50 # High OPEX for natural gas heating & water treatment
        
        # Dynamic Lifting Cost: Thicker oil = higher electricity and mechanical wear
        lifting_cost_per_bbl = 6.0 + (est_viscosity / 150.0) 
        
        # Maintenance Penalty for running in warning/danger zones
        maint_penalty = 0
        if rod_float_risk == 1:
            maint_penalty = 35000 # Cost of impending rod snap & downtime
        elif est_viscosity > 2500:
            maint_penalty = 12000 # Fluid pound wear & tear penalty
        
        daily_revenue = pred_oil_rate * oil_price_per_bbl
        daily_steam_cost = (steam_vol / 30.0) * steam_cost_per_bbl
        daily_lifting_cost = pred_oil_rate * lifting_cost_per_bbl
        
        total_opex = daily_steam_cost + daily_lifting_cost + maint_penalty
        net_profit = daily_revenue - total_opex
        
        st.markdown(f"**Gross Revenue:** <span style='color:#cbd5e1;'>${daily_revenue:,.0f}</span>", unsafe_allow_html=True)
        
        # UI Improvement: OPEX Breakdown Expander
        with st.expander("View OPEX Breakdown"):
            st.markdown(f"<span style='color:#94a3b8;'>Steam Generation:</span> <span style='color:#fbbf24;'>${daily_steam_cost:,.0f}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:#94a3b8;'>Dynamic Lifting:</span> <span style='color:#fbbf24;'>${daily_lifting_cost:,.0f}</span>", unsafe_allow_html=True)
            if maint_penalty > 0:
                st.markdown(f"<span style='color:#94a3b8;'>Risk/Wear Penalty:</span> <span style='color:#ef4444; font-weight:bold;'>${maint_penalty:,.0f}</span>", unsafe_allow_html=True)
                
        st.markdown(f"**Total OPEX:** <span style='color:#ef4444;'>-${total_opex:,.0f}</span>", unsafe_allow_html=True)
        
        if net_profit > 0:
            st.markdown(f"<div style='margin-top:10px; padding:10px; background-color:#166534; border-radius:5px;'><b>Net Profit:</b> <span style='color:white; font-size:1.2rem;'>+ ${net_profit:,.0f}</span></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='margin-top:10px; padding:10px; background-color:#991b1b; border-radius:5px;'><b>Net Loss:</b> <span style='color:white; font-size:1.2rem;'>- ${abs(net_profit):,.0f}</span></div>", unsafe_allow_html=True)

with col_risk:
    with st.container(border=True):
        st.markdown("<p style='color:#94a3b8; font-weight:bold; margin-bottom:10px;'>Risk Profile</p>", unsafe_allow_html=True)
        
        # FIXED LOGIC: 
        # Rod Floating = Thick oil + Fast motor (rod can't fall fast enough)
        # Fluid Pound = Thin oil + Fast motor (pump empties out too fast)
        rod_float_risk = True if (est_viscosity > 2500 and vfd_spm > 5.0) else False
        fluid_pound_risk = True if (est_viscosity <= 2500 and vfd_spm > 7.5) else False
        
        rod_text = "🔴 High" if rod_float_risk else "🟢 Low"
        visc_text = "🔴 High" if est_viscosity > 4000 else "🟢 Low"
        pound_text = "🔴 High" if fluid_pound_risk else "🟢 Low"
        
        st.markdown(f"**Rod Floating** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {rod_text}", unsafe_allow_html=True)
        st.markdown(f"**High Viscosity** &nbsp;&nbsp;&nbsp; {visc_text}", unsafe_allow_html=True)
        st.markdown(f"**Fluid Pound** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {pound_text}", unsafe_allow_html=True)
        
        # --- Phase 3: Explainable AI (XAI) Visualizer ---
        st.divider()
        st.markdown("<p style='color:#94a3b8; font-weight:bold; margin-bottom:0;'> AI Decision Drivers (Live XAI)</p>", unsafe_allow_html=True)
        
        # Calculate raw synthetic feature importance
        raw_visc = max(0.1, (est_viscosity / 4000) * 55)
        raw_spm = max(0.1, (vfd_spm / 9.0) * 30)
        raw_temp = max(0.1, max(0, (1 - (est_prod_temp / 150)) * 15))
        
        # Normalize to exactly 100%
        total_impact = raw_visc + raw_spm + raw_temp
        f_visc = (raw_visc / total_impact) * 100
        f_spm = (raw_spm / total_impact) * 100
        f_temp = (raw_temp / total_impact) * 100
        
        features = ['Reservoir Temp', 'Motor Speed', 'Fluid Viscosity']
        importances = [f_temp, f_spm, f_visc]
        
        fig_xai = go.Figure(go.Bar(
            x=importances, y=features, orientation='h',
            marker=dict(color=['#38bdf8', '#f59e0b', '#ef4444']),
            text=[f"{val:.1f}%" for val in importances], textposition='auto', textfont=dict(color='white', size=11)
        ))
        fig_xai.update_layout(
            height=160, margin=dict(l=10, r=20, t=10, b=10),
            xaxis=dict(visible=False, range=[0, 100]), # Lock x-axis strictly to 100%
            yaxis=dict(tickfont=dict(color='#cbd5e1', size=12, family="Arial")),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_xai, use_container_width=True)

with col_recom:
    with col_recom:
        with st.container(border=True):
            st.markdown("<p style='color:#94a3b8; font-weight:bold; margin-bottom:10px;'>Recommendations</p>", unsafe_allow_html=True)
        if rod_float_risk == 1:
            st.markdown(f"""
            <div class="alert-box">
                <b style="color:#ef4444; font-size:1.1rem;">⊗ Pump BGH-138 Kinematic Anomaly</b><br>
                Fluid viscosity ({est_viscosity:,.0f} cP) has exceeded safe operating limits for the current motor speed. Severe fluid resistance detected on the downstroke. <br><br><b>Action:</b> Reduce VFD speed below {max(2.5, vfd_spm - 2.0):.1f} SPM immediately.
            </div>
            """, unsafe_allow_html=True)
        elif fluid_pound_risk:
            st.markdown(f"""
            <div class="alert-box" style="border-left-color: orange;">
                <b style="color:orange; font-size:1.1rem;">⚠ Pump BGH-138 Fluid Pound Detected</b><br>
                Pump barrel is under-filled due to high motor speed ({vfd_spm:.1f} SPM) and low fluid viscosity ({est_viscosity:,.0f} cP). <br><br><b>Action:</b> Reduce VFD speed below 7.0 SPM to allow complete pump fill.
            </div>
            """, unsafe_allow_html=True)
        elif est_viscosity > 2500:
            st.markdown(f"""
            <div class="alert-box" style="border-left-color: orange;">
                <b style="color:orange; font-size:1.1rem;">⚠ Pump BGH-138 Thermal Warning</b><br>
                Wellbore is cooling rapidly. Viscosity is approaching critical thresholds. <br><br><b>Action:</b> Monitor load limits closely. Prepare to initiate CSS Cycle {css_cycle + 1} within the next 48 hours.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-box-safe">
                <b style="color:#22c55e; font-size:1.1rem;">✓ System Operating Nominally</b><br>
                Thermodynamic and mechanical parameters are well within designated safety boundaries.<br><br>No immediate action required.
            </div>
            """, unsafe_allow_html=True)

# --- AI Copilot Context Injection ---
# We build this string using the live variables calculated earlier in the script
# --- Dynamic System Alarms for AI Alignment ---
active_alarm = "Normal Operations"
system_action = "Maintain current operational setpoints."

if rod_float_risk:
    active_alarm = "Kinematic Anomaly (Rod Floating)"
    system_action = "Reduce VFD speed below 5.0 SPM immediately."
elif fluid_pound_risk:
    active_alarm = "Fluid Pound Detected"
    system_action = "Reduce VFD speed below 7.0 SPM to allow pump fill."
elif est_viscosity > 4000:
    active_alarm = "Severe Thermal Decay"
    system_action = "Increase Steam Temperature and Volume immediately."

# --- AI Copilot Context Injection ---
system_context = f"""
You are an expert AI SCADA Copilot for Oil India Limited.

CRITICAL INSTRUCTION: Provide extremely brief, accurate, and direct answers (2-3 short bullet points max).
You MUST base your operational commands EXACTLY on the 'System Recommended Action' provided below. Do not invent your own setpoints.

Current Telemetry:
- Steam Temp: {steam_temp:.1f} °C
- Pump Speed: {vfd_spm:.1f} SPM
- Viscosity: {est_viscosity:.0f} cP

System Diagnostics (DO NOT CONTRADICT THIS):
- Active Alarm: {active_alarm}
- System Recommended Action: {system_action}

Process Economics:
- Total Daily OPEX: ${total_opex:,.0f}
- Net Profit/Loss: ${net_profit:,.0f}
"""

# --- AI SCADA Copilot Section ---
st.markdown("---")
st.markdown("<h3 style='color:#38bdf8;'> AI SCADA Copilot (Llama 3.2)</h3>", unsafe_allow_init:=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chat_container = st.container()

# Using text_input + submit button instead of st.chat_input to prevent forced viewport jumping
with st.form(key="copilot_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        prompt = st.text_input("Prompt", placeholder="Ask the AI about the current well state or economics...", label_visibility="collapsed")
    with col_btn:
        submit_pressed = st.form_submit_button("Send Query", use_container_width=True)

if submit_pressed and prompt:
    st.session_state.chat_history = [{"role": "user", "content": prompt}]
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing telemetry..."):
                try:
                    messages = [
                        {'role': 'system', 'content': system_context},
                        {'role': 'user', 'content': prompt}
                    ]
                    response = ollama.chat(model='llama3.2', messages=messages)
                    ai_reply = response['message']['content']
                    
                    st.markdown(ai_reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
                    
                except Exception as e:
                    st.error(f"Could not reach Ollama service. Error: {e}")

else:
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# --- IoT Streaming Heartbeat ---
if st.session_state.iot_streaming:
    time.sleep(2) # Wait 2 seconds before pulling the next sensor reading
    st.rerun()
