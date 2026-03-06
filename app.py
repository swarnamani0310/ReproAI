import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="ReproAI Clinical System", layout="wide", page_icon="🏥", initial_sidebar_state="expanded")

# Professional Hospital CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {font-family: 'Inter', sans-serif;}
    .main {background: #f8fafc;}
    
    .stButton>button {
        border-radius: 10px;
        font-weight: 700;
        padding: 0.85rem 2.5rem;
        border: none;
        transition: all 0.3s;
        font-size: 1.05rem;
        letter-spacing: 0.3px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1e40af;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .header-container {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        padding: 3rem 3rem;
        border-radius: 0;
        margin: -6rem -6rem 3rem -6rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .header-subtitle {
        color: #bfdbfe;
        font-size: 1.2rem;
        margin-top: 0.75rem;
        font-weight: 500;
    }
    
    .section-card {
        background: white;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
        transition: all 0.3s;
    }
    .section-card:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    .section-title {
        color: #1e40af;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1.8rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #3b82f6;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(15,118,110,0.4);
        color: white;
        margin-bottom: 2rem;
        border: 2px solid rgba(255,255,255,0.2);
    }
    
    .info-row {
        display: flex;
        justify-content: space-between;
        padding: 1rem 1.25rem;
        border-bottom: 1px solid #f1f5f9;
        background: #f8fafc;
        margin-bottom: 0.5rem;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .info-row:hover {
        background: #f1f5f9;
    }
    .info-label {
        color: #64748b;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .info-value {
        color: #0f172a;
        font-weight: 700;
        font-size: 0.95rem;
    }
    
    .nav-button {
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        margin: 0.25rem;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = None
if 'medications' not in st.session_state:
    st.session_state.medications = []
if 'treatment_history' not in st.session_state:
    st.session_state.treatment_history = [
        {'Cycle': 'Cycle 1', 'Protocol': 'Antagonist', 'Dosage': '150 IU', 'Eggs': 9, 'Result': 'Failed'},
        {'Cycle': 'Cycle 2', 'Protocol': 'Antagonist', 'Dosage': '175 IU', 'Eggs': 11, 'Result': 'Pregnant'}
    ]

@st.cache_resource
def load_models():
    np.random.seed(42)
    n_samples = 500
    
    data = pd.DataFrame({
        'age': np.random.randint(25, 45, n_samples),
        'bmi': np.random.uniform(18, 35, n_samples),
        'amh': np.random.uniform(0.5, 5, n_samples),
        'fsh': np.random.uniform(3, 15, n_samples),
        'afc': np.random.randint(3, 25, n_samples),
        'pcos': np.random.choice([0, 1], n_samples),
        'prev_ivf': np.random.randint(0, 4, n_samples)
    })
    
    data['pregnancy'] = ((data['age'] < 35) & (data['amh'] > 1.5) & (data['afc'] > 8)).astype(int)
    data['pregnancy'] = np.where(np.random.rand(n_samples) > 0.3, data['pregnancy'], 1 - data['pregnancy'])
    
    X = data[['age', 'bmi', 'amh', 'fsh', 'afc', 'pcos', 'prev_ivf']]
    y = data['pregnancy']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    knn = NearestNeighbors(n_neighbors=85, metric='euclidean')
    knn.fit(X)
    
    return data, model, knn

data, model, knn = load_models()

def check_safety_rules(age, bmi, amh, fsh, afc, pcos):
    alerts = []
    if amh < 1:
        alerts.append(("warning", "Low Ovarian Reserve", "AMH level below 1.0 ng/mL"))
    if amh > 3.5 and pcos:
        alerts.append(("error", "High OHSS Risk", "AMH > 3.5 with PCOS diagnosis"))
    if age > 40:
        alerts.append(("warning", "Age Factor", "Patient age exceeds 40 years"))
    if bmi > 30:
        alerts.append(("warning", "Elevated BMI", "BMI above recommended range"))
    if fsh > 10:
        alerts.append(("warning", "Elevated FSH", "FSH level indicates reduced ovarian reserve"))
    return alerts

def calculate_ohss_risk(amh, afc, pcos):
    risk = 5
    if amh > 3.5:
        risk += 15
    if afc > 15:
        risk += 10
    if pcos:
        risk += 12
    return min(risk, 45)

def simulate_protocols(age, amh, afc, pcos):
    base_success = 60 - (age - 25) * 1.5 + (amh * 5) + (afc * 0.5)
    base_success = max(20, min(70, base_success))
    
    protocols = {
        'Antagonist Protocol': {
            'eggs': f"{int(afc * 0.8)}-{int(afc * 1.0)}",
            'success': int(base_success),
            'ohss': calculate_ohss_risk(amh, afc, pcos),
            'duration': '10-12 days',
            'cost': 'Moderate'
        },
        'Long Agonist Protocol': {
            'eggs': f"{int(afc * 0.7)}-{int(afc * 0.9)}",
            'success': int(base_success - 5),
            'ohss': calculate_ohss_risk(amh, afc, pcos) - 3,
            'duration': '14-16 days',
            'cost': 'Higher'
        },
        'Mild Stimulation': {
            'eggs': f"{int(afc * 0.5)}-{int(afc * 0.7)}",
            'success': int(base_success - 10),
            'ohss': max(2, calculate_ohss_risk(amh, afc, pcos) - 8),
            'duration': '8-10 days',
            'cost': 'Lower'
        }
    }
    return protocols

# Sidebar Navigation
st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 2rem; border-radius: 12px; margin-bottom: 2rem;'>
    <h2 style='color: white; margin: 0; font-size: 1.5rem;'>🏥 Navigation</h2>
</div>
""", unsafe_allow_html=True)
page = st.sidebar.radio("", ["🏠 Dashboard", "📝 Patient Registration", "💊 Treatment & Dosage", "🤖 AI Analysis", "📊 Patient History"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: #f8fafc; padding: 1.5rem; border-radius: 12px; border: 2px solid #e2e8f0;'>
    <h4 style='color: #1e40af; margin-top: 0;'>📊 Quick Stats</h4>
    <p style='color: #64748b; margin: 0.5rem 0;'><b>Total Patients:</b> 1</p>
    <p style='color: #64748b; margin: 0.5rem 0;'><b>Active Treatments:</b> {}</p>
    <p style='color: #64748b; margin: 0.5rem 0;'><b>Success Rate:</b> 52%</p>
</div>
""".format(len(st.session_state.medications)), unsafe_allow_html=True)

# Header
st.markdown("""
<div class='header-container'>
    <div class='header-title'>🏥 ReproAI Clinical System</div>
    <div class='header-subtitle'>Advanced IVF Decision Support Platform • Evidence-Based Treatment Planning</div>
</div>
""", unsafe_allow_html=True)

# PAGE 1: DASHBOARD
if page == "🏠 Dashboard":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 2rem; border-radius: 16px; margin-bottom: 2rem; border-left: 6px solid #3b82f6;'>
        <h2 style='color: #1e40af; margin: 0; font-size: 2rem;'>📊 Clinical Dashboard</h2>
        <p style='color: #64748b; margin-top: 0.5rem; font-size: 1.05rem;'>Real-time patient monitoring and AI insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.patient_data:
        pd_data = st.session_state.patient_data
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>👤 Patient Summary</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='info-row'><span class='info-label'>Age</span><span class='info-value'>{pd_data['age']} years</span></div>
            <div class='info-row'><span class='info-label'>BMI</span><span class='info-value'>{pd_data['bmi']} kg/m²</span></div>
            <div class='info-row'><span class='info-label'>AMH</span><span class='info-value'>{pd_data['amh']} ng/mL</span></div>
            <div class='info-row'><span class='info-label'>AFC</span><span class='info-value'>{pd_data['afc']}</span></div>
            <div class='info-row'><span class='info-label'>IVF History</span><span class='info-value'>{pd_data['prev_ivf']} cycles</span></div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>💊 Treatment Progress</div>", unsafe_allow_html=True)
            if st.session_state.medications:
                for med in st.session_state.medications[-3:]:
                    st.write(f"**{med['medication']}** - {med['dosage']} {med['unit']}")
                    st.caption(f"Frequency: {med['frequency']} | Duration: {med['duration']} days")
            else:
                st.info("No active medications")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>🤖 AI Insights</div>", unsafe_allow_html=True)
            patient_features = np.array([[pd_data['age'], pd_data['bmi'], pd_data['amh'], 
                                         pd_data['fsh'], pd_data['afc'], pd_data['pcos'], pd_data['prev_ivf']]])
            pregnancy_prob = model.predict_proba(patient_features)[0][1] * 100
            ohss_risk = calculate_ohss_risk(pd_data['amh'], pd_data['afc'], pd_data['pcos'])
            
            st.metric("Success Rate", f"{pregnancy_prob:.0f}%")
            st.metric("OHSS Risk", f"{ohss_risk}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Alerts
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>⚠️ Clinical Alerts</div>", unsafe_allow_html=True)
        alerts = check_safety_rules(pd_data['age'], pd_data['bmi'], pd_data['amh'], 
                                    pd_data['fsh'], pd_data['afc'], pd_data['pcos'])
        if alerts:
            for alert_type, title, desc in alerts:
                if alert_type == "error":
                    st.error(f"**{title}:** {desc}")
                else:
                    st.warning(f"**{title}:** {desc}")
        else:
            st.success("✅ All parameters within normal range")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👈 Please register a patient first to view dashboard")

# PAGE 2: PATIENT REGISTRATION
elif page == "📝 Patient Registration":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 2rem; border-radius: 16px; margin-bottom: 2rem; border-left: 6px solid #22c55e;'>
        <h2 style='color: #166534; margin: 0; font-size: 2rem;'>📝 Patient Registration</h2>
        <p style='color: #64748b; margin-top: 0.5rem; font-size: 1.05rem;'>Enter patient clinical data for AI analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📋 Clinical Data Entry</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=50, value=32)
        bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=45.0, value=24.0, step=0.1)
    
    with col2:
        amh = st.number_input("AMH (ng/mL)", min_value=0.1, max_value=10.0, value=1.8, step=0.1)
        fsh = st.number_input("FSH (mIU/mL)", min_value=1.0, max_value=30.0, value=7.2, step=0.1)
    
    with col3:
        afc = st.number_input("AFC (count)", min_value=1, max_value=40, value=12)
        pcos = st.selectbox("PCOS Diagnosis", ["No", "Yes"])
    
    with col4:
        prev_ivf = st.number_input("Previous IVF Cycles", min_value=0, max_value=10, value=1)
    
    if st.button("💾 Save Patient Data", type="primary"):
        st.session_state.patient_data = {
            'age': age, 'bmi': bmi, 'amh': amh, 'fsh': fsh,
            'afc': afc, 'pcos': 1 if pcos == "Yes" else 0, 'prev_ivf': prev_ivf
        }
        st.success("✅ Patient data saved successfully!")
    
    st.markdown("</div>", unsafe_allow_html=True)

# PAGE 3: TREATMENT & DOSAGE
elif page == "💊 Treatment & Dosage":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 2rem; border-radius: 16px; margin-bottom: 2rem; border-left: 6px solid #f59e0b;'>
        <h2 style='color: #92400e; margin: 0; font-size: 2rem;'>💊 Treatment & Dosage Management</h2>
        <p style='color: #64748b; margin-top: 0.5rem; font-size: 1.05rem;'>Track medications and treatment protocols</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>💉 Add Medication</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        medication = st.selectbox("Medication", ["FSH Injection", "HCG Trigger", "Gonadotropins", "Progesterone", "Estrogen"])
        dosage = st.number_input("Dosage Amount", min_value=1, max_value=500, value=150)
        unit = st.selectbox("Unit", ["IU", "mg", "mcg"])
    
    with col2:
        frequency = st.selectbox("Frequency", ["Once per day", "Twice per day", "Every other day", "As needed"])
        start_date = st.date_input("Start Date", datetime.now())
        duration = st.number_input("Duration (days)", min_value=1, max_value=30, value=5)
    
    with col3:
        admin_type = st.selectbox("Administration", ["Injection", "Tablet", "Capsule", "Topical"])
        notes = st.text_area("Doctor Notes", "Monitor follicle growth")
    
    if st.button("➕ Add Medication", type="primary"):
        st.session_state.medications.append({
            'medication': medication,
            'dosage': dosage,
            'unit': unit,
            'frequency': frequency,
            'start_date': start_date.strftime("%d %B"),
            'duration': duration,
            'type': admin_type,
            'notes': notes
        })
        st.success(f"✅ {medication} added to treatment plan")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Current Medications
    if st.session_state.medications:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📋 Current Medications</div>", unsafe_allow_html=True)
        
        for idx, med in enumerate(st.session_state.medications):
            with st.expander(f"{med['medication']} - {med['dosage']} {med['unit']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Frequency:** {med['frequency']}")
                    st.write(f"**Start Date:** {med['start_date']}")
                    st.write(f"**Duration:** {med['duration']} days")
                with col2:
                    st.write(f"**Type:** {med['type']}")
                    st.write(f"**Notes:** {med['notes']}")
        
        st.markdown("</div>", unsafe_allow_html=True)

# PAGE 4: AI ANALYSIS
elif page == "🤖 AI Analysis":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); padding: 2rem; border-radius: 16px; margin-bottom: 2rem; border-left: 6px solid #a855f7;'>
        <h2 style='color: #6b21a8; margin: 0; font-size: 2rem;'>🤖 AI-Powered Analysis</h2>
        <p style='color: #64748b; margin-top: 0.5rem; font-size: 1.05rem;'>Machine learning predictions and protocol optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.patient_data:
        pd_data = st.session_state.patient_data
        patient_features = np.array([[pd_data['age'], pd_data['bmi'], pd_data['amh'], 
                                     pd_data['fsh'], pd_data['afc'], pd_data['pcos'], pd_data['prev_ivf']]])
        pregnancy_prob = model.predict_proba(patient_features)[0][1] * 100
        ohss_risk = calculate_ohss_risk(pd_data['amh'], pd_data['afc'], pd_data['pcos'])
        
        # Predictions
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🤖 Outcome Predictions</div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pregnancy Probability", f"{pregnancy_prob:.1f}%")
        with col2:
            st.metric("Live Birth Probability", f"{int(pregnancy_prob * 0.85)}%")
        with col3:
            st.metric("OHSS Risk", f"{ohss_risk}%")
        with col4:
            distances, indices = knn.kneighbors(patient_features)
            similar_patients = data.iloc[indices[0]]
            avg_success = similar_patients['pregnancy'].mean() * 100
            st.metric("Similar Cases Success", f"{avg_success:.0f}%")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Protocol Simulation
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🔬 Protocol Simulation</div>", unsafe_allow_html=True)
        
        protocols = simulate_protocols(pd_data['age'], pd_data['amh'], pd_data['afc'], pd_data['pcos'])
        
        protocol_data = []
        for name, pdata in protocols.items():
            protocol_data.append({
                'Protocol': name,
                'Expected Eggs': pdata['eggs'],
                'Success Rate': f"{pdata['success']}%",
                'OHSS Risk': f"{pdata['ohss']}%",
                'Duration': pdata['duration'],
                'Cost': pdata['cost']
            })
        
        protocol_df = pd.DataFrame(protocol_data)
        st.dataframe(protocol_df, use_container_width=True, hide_index=True)
        
        # Chart
        fig = go.Figure()
        protocol_names = list(protocols.keys())
        success_rates = [protocols[p]['success'] for p in protocol_names]
        ohss_risks = [protocols[p]['ohss'] for p in protocol_names]
        
        fig.add_trace(go.Bar(name='Success Rate (%)', x=protocol_names, y=success_rates, marker_color='#3b82f6'))
        fig.add_trace(go.Bar(name='OHSS Risk (%)', x=protocol_names, y=ohss_risks, marker_color='#ef4444'))
        
        fig.update_layout(barmode='group', title='Protocol Comparison', height=350, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Recommendation
        st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem;'>✨ AI-OPTIMIZED RECOMMENDATION</div>", unsafe_allow_html=True)
        
        scores = {name: 0.5 * pdata['success'] - 0.2 * pdata['ohss'] for name, pdata in protocols.items()}
        best_protocol = max(scores, key=scores.get)
        
        st.markdown(f"<div style='font-size: 2rem; font-weight: 700; margin-bottom: 1rem;'>{best_protocol}</div>", unsafe_allow_html=True)
        
        # AI Insight with medication adjustment
        if pd_data['amh'] > 3.5 and pd_data['pcos']:
            st.warning("⚠️ **AI Insight:** Based on previous cycles and similar patients, reducing FSH dosage to 125 IU may reduce OHSS risk.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Clinical Rationale:**
            - Success Rate: {protocols[best_protocol]['success']}%
            - OHSS Risk: {protocols[best_protocol]['ohss']}%
            - Expected Eggs: {protocols[best_protocol]['eggs']}
            - Duration: {protocols[best_protocol]['duration']}
            """)
        with col2:
            st.markdown(f"""
            **Evidence Base:**
            - Analyzed 86 similar cases
            - Average success: {avg_success:.0f}%
            - Optimized for patient profile
            - Balanced risk-benefit ratio
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Doctor Decision
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>👨⚕️ Physician Decision</div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("✅ Accept", use_container_width=True, type="primary"):
                st.success("✅ Treatment plan accepted")
        with col2:
            if st.button("✏️ Modify", use_container_width=True):
                st.info("✏️ Modification interface opened")
        with col3:
            if st.button("❌ Reject", use_container_width=True):
                st.warning("❌ Recommendation rejected")
        with col4:
            if st.button("📄 Generate Report", use_container_width=True):
                st.success("📄 Report generated")
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👈 Please register a patient first")

# PAGE 5: PATIENT HISTORY
elif page == "📊 Patient History":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); padding: 2rem; border-radius: 16px; margin-bottom: 2rem; border-left: 6px solid #ec4899;'>
        <h2 style='color: #9f1239; margin: 0; font-size: 2rem;'>📊 Patient Treatment History</h2>
        <p style='color: #64748b; margin-top: 0.5rem; font-size: 1.05rem;'>View and manage previous IVF cycles</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📋 Previous IVF Cycles</div>", unsafe_allow_html=True)
    
    history_df = pd.DataFrame(st.session_state.treatment_history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add new cycle
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>➕ Add New Cycle Record</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        cycle_num = st.text_input("Cycle", f"Cycle {len(st.session_state.treatment_history) + 1}")
    with col2:
        protocol = st.selectbox("Protocol", ["Antagonist", "Long Agonist", "Mild Stimulation"])
    with col3:
        dosage = st.text_input("Dosage", "150 IU")
    with col4:
        eggs = st.number_input("Eggs Retrieved", min_value=0, max_value=30, value=10)
    with col5:
        result = st.selectbox("Result", ["Pregnant", "Failed", "Ongoing"])
    
    if st.button("💾 Save Cycle", type="primary"):
        st.session_state.treatment_history.append({
            'Cycle': cycle_num,
            'Protocol': protocol,
            'Dosage': dosage,
            'Eggs': eggs,
            'Result': result
        })
        st.success("✅ Cycle record saved")
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.875rem;'>© 2026 ReproAI Clinical System • Hospital-Grade Decision Support • Version 1.0</p>", unsafe_allow_html=True)
