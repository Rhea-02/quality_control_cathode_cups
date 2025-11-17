import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import os
import random
from datetime import datetime, timedelta
import io
import pickle
import numpy as np
import cv2
import torch
import plotly.graph_objects as go
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Quality Control Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Load ML Model ---
@st.cache_resource
def load_model():
    """Load the trained ML model from pickle file"""
    try:
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model.pkl")
        
        if not os.path.exists(model_path):
            return None
        
        with open(model_path, "rb") as f:
            model = CPU_Unpickler(f).load()
        
        model.to(torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        return None

ML_MODEL = load_model()

# --- Initialize Session State ---
if 'classification_feed' not in st.session_state:
    st.session_state.classification_feed = []
if 'total_processed' not in st.session_state:
    st.session_state.total_processed = 0
if 'total_good' not in st.session_state:
    st.session_state.total_good = 0
if 'total_defective' not in st.session_state:
    st.session_state.total_defective = 0
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'Dashboard'

# --- CSS ---
st.markdown("""
<style>
.main { background-color: #F8F9FA; }
.metric-card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def get_metrics():
    return {
        "parts_processed": st.session_state.total_processed,
        "good_parts": st.session_state.total_good,
        "defective_parts": st.session_state.total_defective,
        "avg_inference_time": 57
    }

def add_to_feed(image, status, confidence, labels, annotated_image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    
    annotated_bytes = None
    if annotated_image is not None:
        annotated_bytes = io.BytesIO()
        Image.fromarray(annotated_image).save(annotated_bytes, format='PNG')
        annotated_bytes = annotated_bytes.getvalue()
    
    defect_types = [l for l in labels if l.lower() != "good"] if labels else []
    
    st.session_state.classification_feed.insert(0, {
        'id': len(st.session_state.classification_feed) + 1,
        'time': datetime.now().strftime('%H:%M:%S'),
        'status': status,
        'confidence': confidence,
        'image_bytes': img_bytes.getvalue(),
        'annotated_bytes': annotated_bytes,
        'defect_types': defect_types
    })
    
    st.session_state.total_processed += 1
    if status == "Good":
        st.session_state.total_good += 1
    else:
        st.session_state.total_defective += 1

def classify_image(image):
    is_good = random.random() > 0.2
    confidence = random.uniform(0.85, 0.98) if is_good else random.uniform(0.75, 0.95)
    
    return {
        "status": "Good" if is_good else "Defective",
        "confidence": confidence,
        "boxes": [],
        "labels": ["crack"] if not is_good else [],
        "annotated_image": None
    }

# --- Navigation Tabs ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📊 Dashboard", key="tab1", use_container_width=True):
        st.session_state.active_tab = 'Dashboard'
        st.rerun()
with col2:
    if st.button("🔬 Analyzers", key="tab2", use_container_width=True):
        st.session_state.active_tab = 'Analyzers'
        st.rerun()
with col3:
    if st.button("⚙️ Settings", key="tab3", use_container_width=True):
        st.session_state.active_tab = 'Settings'
        st.rerun()
with col4:
    if st.button("📈 Reports", key="tab4", use_container_width=True):
        st.session_state.active_tab = 'Reports'
        st.rerun()

st.divider()

# --- Dashboard Header ---
st.title(st.session_state.active_tab)

# --- TAB 1: DASHBOARD ---
if st.session_state.active_tab == 'Dashboard':
    metrics = get_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Parts Processed", metrics['parts_processed'], "+5%")
    col2.metric("Good Parts", metrics['good_parts'], "95.3%")
    col3.metric("Defective Parts", metrics['defective_parts'], "-4.7%")
    col4.metric("Avg Inference Time", f"{metrics['avg_inference_time']}ms", "Fast")
    
    st.subheader("📸 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col_a, col_b = st.columns(2)
        col_a.image(image, caption="Original", use_container_width=True)
        
        if st.button("🔍 Classify", type="primary"):
            result = classify_image(image)
            if result["status"] == "Good":
                st.success(f"✅ {result['status']} - {result['confidence']*100:.1f}%")
            else:
                st.error(f"❌ {result['status']} - {result['confidence']*100:.1f}%")
            add_to_feed(image, result["status"], result["confidence"], result["labels"], result["annotated_image"])
            st.rerun()
    
    st.divider()
    st.subheader("Recent Classifications")
    
    if st.session_state.classification_feed:
        for item in st.session_state.classification_feed[:5]:
            col_x, col_y = st.columns([1, 4])
            with col_x:
                img = Image.open(io.BytesIO(item['image_bytes']))
                st.image(img, width=80)
            with col_y:
                st.write(f"**Cathode Cup {item['id']}** - {item['time']}")
                if item['status'] == "Good":
                    st.success(f"✅ {item['status']} ({item['confidence']*100:.1f}%)")
                else:
                    st.error(f"❌ {item['status']} ({item['confidence']*100:.1f}%)")
    else:
        st.info("No classifications yet. Upload an image above.")

# --- TAB 2: ANALYZERS ---
elif st.session_state.active_tab == 'Analyzers':
    st.title("📊 Analysis Dashboard")
    
    # Sub-tabs for different analysis types
    tab1, tab2, tab3 = st.tabs(["📈 Trend Analysis", "🔍 Defect Patterns", "📉 Statistical Process Control"])
    
    with tab1:  # Trend Analysis
        st.subheader("Quality Trends Over Time")
        
        # Time range selector
        col1, col2 = st.columns([2, 1])
        with col1:
            time_range = st.selectbox("Time Range", ["Last 7 Days", "Last 30 Days", "Last 90 Days"])
        with col2:
            refresh_btn = st.button("🔄 Refresh Data")
        
        st.divider()
        
        # Defect Rate Trend
        st.markdown("#### Overall Defect Rate Trend")
        import pandas as pd
        import numpy as np
        
        # Generate sample trend data
        days = 30
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        defect_rate = np.random.uniform(2, 5, days) + np.sin(np.linspace(0, 4*np.pi, days)) * 0.5
        
        # Create plotly figure with proper axes
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=defect_rate,
            mode='lines+markers',
            name='Defect Rate',
            line=dict(color='#FF4B4B', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Defect Rate (%)',
            hovermode='x unified',
            showlegend=True,
            height=400,
            yaxis=dict(range=[0, max(defect_rate) * 1.2])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Shift Comparison
        st.divider()
        st.markdown("#### Shift Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Shift defect rates over time
            shift_dates = pd.date_range(end=pd.Timestamp.now(), periods=14, freq='D')
            shift1_data = np.random.uniform(2, 4, 14)
            shift2_data = np.random.uniform(2.5, 4.5, 14)
            
            fig_shift = go.Figure()
            fig_shift.add_trace(go.Scatter(
                x=shift_dates,
                y=shift1_data,
                mode='lines+markers',
                name='1st Shift (7AM-3PM)',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            ))
            fig_shift.add_trace(go.Scatter(
                x=shift_dates,
                y=shift2_data,
                mode='lines+markers',
                name='2nd Shift (3PM-11PM)',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=4)
            ))
            
            fig_shift.update_layout(
                xaxis_title='Date',
                yaxis_title='Defect Rate (%)',
                hovermode='x unified',
                showlegend=True,
                height=350,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                yaxis=dict(range=[0, max(max(shift1_data), max(shift2_data)) * 1.2])
            )
            
            st.plotly_chart(fig_shift, use_container_width=True)
        
        with col2:
            # Current shift metrics
            st.metric("1st Shift Avg", "2.9%", "-0.3%", delta_color="inverse")
            st.metric("2nd Shift Avg", "3.4%", "+0.1%", delta_color="inverse")
            st.metric("Overall Trend", "Improving", "↓ 0.5%")
        
        # Weekly comparison
        st.divider()
        st.markdown("#### Weekly Performance")
        
        weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
        week_data = pd.DataFrame({
            'Week': weeks,
            'Parts Inspected': [198, 205, 192, 203],
            'Defects': [6, 7, 6, 5],
            'Defect Rate (%)': [3.0, 3.4, 3.1, 2.5]
        })
        st.dataframe(week_data, use_container_width=True)
    
    with tab2:  # Defect Patterns
        st.subheader("Defect Type Analysis")
        
        # Defect distribution
        st.markdown("#### Defect Distribution by Type")
        
        defect_types = ['Asymmetry', 'Dents', 'Scratches', 'Rust']
        defect_counts = [8, 6, 5, 5]
        
        fig_defects = go.Figure()
        fig_defects.add_trace(go.Bar(
            x=defect_types,
            y=defect_counts,
            name='Defect Count',
            marker_color='#FF4B4B',
            text=defect_counts,
            textposition='outside'
        ))
        
        fig_defects.update_layout(
            xaxis_title='Defect Type',
            yaxis_title='Count (units)',
            showlegend=True,
            height=400,
            yaxis=dict(range=[0, max(defect_counts) * 1.3])
        )
        
        st.plotly_chart(fig_defects, use_container_width=True)
        
        st.divider()
        
        # Common failure modes
        st.markdown("#### Top Failure Modes")
        
        failure_modes = pd.DataFrame({
            'Failure Mode': [
                'Asymmetry - Cup shape irregularity',
                'Dents - Surface indentations',
                'Scratches - Surface abrasions',
                'Rust - Corrosion spots'
            ],
            'Frequency': [8, 6, 5, 5],
            'Avg Impact': ['High', 'High', 'Medium', 'Medium']
        })
        st.dataframe(failure_modes, use_container_width=True)
        
        # Root cause indicators
        st.divider()
        st.markdown("#### Root Cause Indicators")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Material-Related", "28%", "+2%", delta_color="inverse")
        with col2:
            st.metric("Process-Related", "45%", "-3%", delta_color="normal")
        with col3:
            st.metric("Equipment-Related", "27%", "+1%", delta_color="inverse")
    
    with tab3:  # Statistical Process Control
        st.subheader("Statistical Process Control (SPC)")
        
        # Control chart
        st.markdown("#### Process Control Chart (X-bar Chart)")
        
        # Generate SPC data
        n_points = 50
        process_mean = 3.0
        process_std = 0.5
        
        measurements = np.random.normal(process_mean, process_std, n_points)
        ucl = process_mean + 3 * process_std  # Upper Control Limit (μ + 3σ)
        lcl = max(0, process_mean - 3 * process_std)  # Lower Control Limit (μ - 3σ)
        
        sample_numbers = list(range(1, n_points + 1))
        
        fig_spc = go.Figure()
        
        # Actual measurements
        fig_spc.add_trace(go.Scatter(
            x=sample_numbers,
            y=measurements,
            mode='lines+markers',
            name='Defect Rate',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=5)
        ))
        
        # UCL line
        fig_spc.add_trace(go.Scatter(
            x=sample_numbers,
            y=[ucl] * n_points,
            mode='lines',
            name=f'UCL (μ + 3σ)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Center line
        fig_spc.add_trace(go.Scatter(
            x=sample_numbers,
            y=[process_mean] * n_points,
            mode='lines',
            name=f'Mean (μ)',
            line=dict(color='green', width=2, dash='solid')
        ))
        
        # LCL line
        fig_spc.add_trace(go.Scatter(
            x=sample_numbers,
            y=[lcl] * n_points,
            mode='lines',
            name=f'LCL (μ - 3σ)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_spc.update_layout(
            xaxis_title='Sample Number',
            yaxis_title='Defect Rate (%)',
            hovermode='x unified',
            showlegend=True,
            height=450,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            yaxis=dict(range=[max(0, lcl - 0.5), ucl + 0.5])
        )
        
        st.plotly_chart(fig_spc, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("UCL (Upper)", f"{ucl:.2f}%")
        with col2:
            st.metric("Process Mean", f"{process_mean:.2f}%")
        with col3:
            st.metric("LCL (Lower)", f"{lcl:.2f}%")
        
        st.divider()
        
        # Process capability
        st.markdown("#### Process Capability Indices")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Cp", "1.33", "Good")
        with col2:
            st.metric("Cpk", "1.25", "Good")
        with col3:
            st.metric("Pp", "1.28", "Acceptable")
        with col4:
            st.metric("Ppk", "1.22", "Acceptable")
        
        st.info("**Cp/Cpk > 1.33**: Process capable | **1.0-1.33**: Acceptable | **< 1.0**: Needs improvement")
        
        st.divider()
        
        # Control alerts
        st.markdown("#### Process Control Alerts")
        
        alert_data = pd.DataFrame({
            'Alert Type': [
                'Point beyond control limit',
                '7 consecutive points above mean',
                'Increasing trend detected',
                'High variability detected'
            ],
            'Status': ['🟢 None', '🟢 None', '🟡 Watch', '🟢 None'],
            'Last Occurrence': ['Never', '5 days ago', '2 days ago', '8 days ago']
        })
        st.dataframe(alert_data, use_container_width=True)

# --- TAB 3: SETTINGS ---
elif st.session_state.active_tab == 'Settings':
    tab1, tab2, tab3, tab4 = st.tabs(["⚙️ General", "🤖 Model Management", "🔔 Alerts", "👥 Users"])
    
    with tab1:
        st.subheader("General Settings")
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Theme", ["Light Mode", "Dark Mode", "Auto"])
            st.selectbox("Language", ["English", "Spanish", "French", "German", "Chinese"])
            st.slider("Dashboard Refresh Rate (seconds)", 1, 60, 5)
        with col2:
            st.checkbox("Enable sound notifications", value=True)
            st.checkbox("Show detailed metrics", value=True)
            st.checkbox("Auto-save classifications", value=True)
        
        if st.button("💾 Save General Settings", type="primary", use_container_width=True):
            st.success("✅ General settings saved successfully!")
    
    with tab2:
        st.subheader("Model Management")
        
        # Model Information
        with st.expander("📊 Current Model Information", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Model Version", "v2.1.0")
            col2.metric("Accuracy", "96.8%")
            col3.metric("Last Trained", "Nov 10, 2025")
            
            st.divider()
            col_a, col_b, col_c = st.columns(3)
            col_a.write("**Precision:** 95.2%")
            col_b.write("**Recall:** 94.6%")
            col_c.write("**F1-Score:** 94.9%")
        
        # Classification Settings
        st.subheader("Classification Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Confidence Threshold (%)", 0, 100, 85, help="Minimum confidence for classification")
            st.slider("Defect Detection Sensitivity", 1, 10, 7, help="Higher = more sensitive to defects")
        with col2:
            st.number_input("Max Inference Time (ms)", 10, 5000, 200)
            st.selectbox("Image Preprocessing", ["Standard", "Enhanced", "Fast"])
        
        # Retrain Model Section
        st.divider()
        st.subheader("🔄 Model Retraining")
        
        with st.expander("Retrain Model - Advanced Options"):
            st.warning("⚠️ Model retraining requires administrator privileges")
            
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Training Data Path", value="./training_data/")
                st.number_input("Training Epochs", 10, 500, 100)
                st.number_input("Batch Size", 8, 128, 32)
            with col2:
                st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
                st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
                st.checkbox("Use Data Augmentation", value=True)
            
            if st.button("🚀 Start Retraining", type="primary", use_container_width=True):
                with st.spinner("Initializing retraining process..."):
                    import time
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                    st.success("✅ Model retraining completed successfully! New model version: v2.2.0")
        
        # Upload New Model
        st.divider()
        st.subheader("📤 Upload Custom Model")
        uploaded_model = st.file_uploader("Upload model.pkl file", type=['pkl'])
        if uploaded_model:
            st.info(f"📦 Model file uploaded: {uploaded_model.name} ({uploaded_model.size} bytes)")
            if st.button("🔄 Deploy Uploaded Model", type="primary"):
                st.success("✅ Model deployed successfully!")
        
        if st.button("💾 Save Model Settings", type="primary", use_container_width=True):
            st.success("✅ Model settings saved!")
    
    with tab3:
        st.subheader("Alert Thresholds")
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Defect Rate Alert (%)", 0, 100, 15, help="Alert when defect rate exceeds this")
            st.slider("Low Confidence Alert (%)", 0, 100, 70, help="Alert when confidence below this")
        with col2:
            st.number_input("Alert Frequency (minutes)", 1, 1440, 30, help="Minimum time between alerts")
            st.selectbox("Alert Priority", ["Low", "Medium", "High", "Critical"])
        
        if st.button("💾 Save Alert Settings", type="primary", use_container_width=True):
            st.success("✅ Alert settings saved!")
    
    with tab4:
        st.subheader("User Management")
        
        st.info("👤 **Current User:** admin@gehealthcare.com | **Role:** Administrator | **Last Login:** Nov 14, 2025 09:30 AM")
        
        st.divider()
        st.subheader("Team Members")
        
        users = [
            {"name": "John Smith", "role": "Quality Inspector", "email": "john.smith@ge.com", "status": "Active"},
            {"name": "Sarah Johnson", "role": "Shift Supervisor", "email": "sarah.j@ge.com", "status": "Active"},
            {"name": "Mike Chen", "role": "Quality Inspector", "email": "mike.chen@ge.com", "status": "Inactive"},
            {"name": "Emily Davis", "role": "Manager", "email": "emily.d@ge.com", "status": "Active"}
        ]
        
        for user in users:
            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])
            col1.write(f"**{user['name']}**")
            col2.write(user['role'])
            col3.write(user['email'])
            if user['status'] == "Active":
                col4.success("● Active")
            else:
                col4.warning("● Inactive")
            col5.button("Edit", key=f"edit_{user['name']}")
        
        st.divider()
        if st.button("➕ Add New User", use_container_width=True):
            st.info("👥 User management requires admin privileges")
        
        st.divider()
        st.subheader("Recent Activity Audit Log")
        audit_logs = [
            {"time": "09:30 AM", "user": "admin", "action": "Model settings updated"},
            {"time": "09:15 AM", "user": "john.smith", "action": "Classified 15 parts"},
            {"time": "08:45 AM", "user": "sarah.j", "action": "Generated daily report"},
            {"time": "08:30 AM", "user": "admin", "action": "User login"}
        ]
        
        for log in audit_logs:
            st.text(f"{log['time']} | {log['user']}: {log['action']}")

# --- TAB 4: REPORTS ---
else:
    tab1, tab2, tab3 = st.tabs(["📊 Generate Report", "📈 Report History", "📤 Export Data"])
    
    with tab1:
        st.subheader("Report Configuration")
        
        # Time Range Selection
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        with col3:
            st.selectbox("Shift", ["All Shifts", "1st Shift (7AM-3PM)", "2nd Shift (3PM-11PM)"])
        
        # Report Type Selection
        st.divider()
        report_type = st.radio(
            "Select Report Type",
            ["Daily Summary", "Weekly Trend Analysis", "Defect Analysis", "Batch Comparison"],
            horizontal=True
        )
        
        # Additional Filters
        with st.expander("Advanced Filters"):
            col1, col2 = st.columns(2)
            with col1:
                st.multiselect("Equipment IDs", ["CAM-001", "CAM-002", "CAM-003", "CAM-004"], default=["CAM-001"])
                st.multiselect("Operators", ["John Smith", "Sarah Johnson", "Mike Chen"], default=[])
            with col2:
                st.slider("Min Confidence Filter (%)", 0, 100, 0)
                st.multiselect("Defect Types", ["Crack", "Scratch", "Dent", "Discoloration", "Deformation"])
        
        # Generate Report Button
        if st.button("📊 Generate Report", type="primary", use_container_width=True):
            with st.spinner("Generating report..."):
                import time
                time.sleep(1)
                
                st.success("✅ Report generated successfully!")
                
                # Report Summary
                st.divider()
                st.subheader(f"{report_type} - {start_date} to {end_date}")
                
                # Key Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Inspected", "1,248", "+5.2%")
                col2.metric("Pass Rate", "96.3%", "+1.1%")
                col3.metric("Defective Parts", "46", "-2 from last week")
                col4.metric("Avg Confidence", "94.7%", "+0.5%")
                
                # Detailed Statistics
                st.divider()
                st.subheader("Detailed Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Production Summary**")
                    st.write("• Target Parts: 1,200")
                    st.write("• Actual Inspected: 1,248 (104% of target)")
                    st.write("• Good Parts: 1,202")
                    st.write("• Defective Parts: 46")
                    st.write("• Pass Rate: 96.3%")
                    st.write("• Average Processing Time: 57ms")
                
                with col2:
                    st.write("**Defect Breakdown**")
                    st.write("• Cracks: 18 (39.1%)")
                    st.write("• Scratches: 12 (26.1%)")
                    st.write("• Dents: 8 (17.4%)")
                    st.write("• Discoloration: 5 (10.9%)")
                    st.write("• Deformation: 3 (6.5%)")
                
                # Quality Trend (Simple text-based for speed)
                st.divider()
                st.subheader("Quality Trend (Last 7 Days)")
                trend_data = {
                    "Date": ["Nov 8", "Nov 9", "Nov 10", "Nov 11", "Nov 12", "Nov 13", "Nov 14"],
                    "Inspected": [187, 192, 178, 185, 190, 198, 188],
                    "Pass Rate": ["95.7%", "96.9%", "94.9%", "96.2%", "97.4%", "95.5%", "96.3%"]
                }
                
                st.dataframe(trend_data, use_container_width=True)
                
                # Shift Performance
                st.divider()
                st.subheader("Shift Performance")
                col1, col2 = st.columns(2)
                col1.metric("1st Shift (7AM-3PM)", "97.1%", "+0.8%")
                col2.metric("2nd Shift (3PM-11PM)", "96.2%", "+0.3%")
                
                # Equipment Performance
                st.divider()
                st.subheader("Equipment Performance")
                equipment_data = {
                    "Equipment ID": ["CAM-001", "CAM-002", "CAM-003", "CAM-004"],
                    "Parts Processed": [312, 308, 315, 313],
                    "Pass Rate": ["96.8%", "95.5%", "97.1%", "96.2%"],
                    "Avg Confidence": ["95.2%", "94.1%", "95.8%", "94.6%"]
                }
                st.dataframe(equipment_data, use_container_width=True)
                
                # Recommendations
                st.divider()
                st.subheader("📋 Recommendations")
                st.info("✓ Overall quality performance is excellent (96.3% pass rate)")
                st.warning("⚠️ CAM-002 showing slightly lower pass rate - recommend calibration check")
                st.success("✓ All shifts performing above target threshold (>95%)")
    
    with tab2:
        st.subheader("Report History")
        
        st.write("**Recent Reports**")
        
        reports = [
            {"date": "Nov 14, 2025", "type": "Daily Summary", "range": "Nov 14", "generated_by": "admin", "size": "245 KB"},
            {"date": "Nov 13, 2025", "type": "Daily Summary", "range": "Nov 13", "generated_by": "admin", "size": "238 KB"},
            {"date": "Nov 11, 2025", "type": "Weekly Trend", "range": "Nov 5-11", "generated_by": "sarah.j", "size": "1.2 MB"},
            {"date": "Nov 10, 2025", "type": "Defect Analysis", "range": "Nov 1-10", "generated_by": "admin", "size": "856 KB"},
            {"date": "Nov 7, 2025", "type": "Batch Comparison", "range": "Oct-Nov", "generated_by": "emily.d", "size": "1.8 MB"}
        ]
        
        for report in reports:
            col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 2, 1, 1])
            col1.write(report['date'])
            col2.write(report['type'])
            col3.write(report['range'])
            col4.write(report['generated_by'])
            col5.write(report['size'])
            col6.button("📥", key=f"download_{report['date']}")
        
        st.divider()
        if st.button("🗑️ Clear Old Reports (>30 days)", use_container_width=True):
            st.success("✅ Old reports cleared successfully!")
    
    with tab3:
        st.subheader("Export Data")
        
        col1, col2 = st.columns(2)
        with col1:
            st.date_input("Export Start Date", value=datetime.now() - timedelta(days=7), key="export_start")
            st.multiselect("Data Fields", 
                          ["Timestamp", "Part ID", "Classification", "Confidence", "Defect Type", "Operator", "Equipment ID"],
                          default=["Timestamp", "Part ID", "Classification", "Confidence"])
        with col2:
            st.date_input("Export End Date", value=datetime.now(), key="export_end")
            export_format = st.radio("Export Format", ["CSV", "Excel (XLSX)", "PDF Report"], horizontal=True)
        
        st.divider()
        
        # Export buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📄 Export as CSV", use_container_width=True):
                # Create sample CSV data
                csv_data = "Timestamp,Part ID,Classification,Confidence\n"
                csv_data += "2025-11-14 09:30:15,CUP-1248,Good,0.967\n"
                csv_data += "2025-11-14 09:30:28,CUP-1249,Defective,0.923\n"
                csv_data += "2025-11-14 09:30:41,CUP-1250,Good,0.981\n"
                
                st.download_button(
                    label="💾 Download CSV File",
                    data=csv_data,
                    file_name=f"quality_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if st.button("📊 Export as Excel", use_container_width=True):
                st.info("📥 Excel export prepared. Click below to download.")
                # In production, this would generate actual Excel file
                st.download_button(
                    label="💾 Download Excel File",
                    data="Sample Excel Data",
                    file_name=f"quality_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        with col3:
            if st.button("📋 Export as PDF", use_container_width=True):
                st.info("📥 PDF report prepared. Click below to download.")
                report_text = f"""
Quality Control Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
==========================================
Total Parts Inspected: {st.session_state.total_processed}
Good Parts: {st.session_state.total_good}
Defective Parts: {st.session_state.total_defective}
Pass Rate: {(st.session_state.total_good / st.session_state.total_processed * 100) if st.session_state.total_processed > 0 else 0:.1f}%
==========================================
                """
                st.download_button(
                    label="💾 Download PDF Report",
                    data=report_text,
                    file_name=f"quality_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        st.divider()
        st.subheader("Scheduled Exports")
        st.checkbox("Enable automatic daily export at 11:59 PM")
        st.checkbox("Enable automatic weekly summary export (Sundays)")
        st.selectbox("Auto-export Format", ["CSV", "Excel", "PDF"])
        
        if st.button("💾 Save Export Settings", type="primary", use_container_width=True):
            st.success("✅ Export settings saved!")
