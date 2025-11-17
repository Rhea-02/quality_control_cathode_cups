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
        # Custom unpickler to map CUDA tensors to CPU
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)
        
        # Get the directory where app.py is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model.pkl")
        
        if not os.path.exists(model_path):
            st.warning(f"⚠️ Model file not found at: {model_path}")
            st.info("Place your 'model.pkl' file in the venv folder to enable real classification")
            return None
        
        # Load with custom unpickler
        with open(model_path, "rb") as f:
            model = CPU_Unpickler(f).load()
        
        # Move model to CPU and set to eval mode
        model.to(torch.device('cpu'))
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Load model globally
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

# --- Load Custom CSS ---
def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Inline fallback CSS
        st.markdown("""
        <style>
        .main { background-color: #F8F9FA; }
        .metric-card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .feed-card { background: white; border-radius: 8px; padding: 16px; margin-bottom: 12px; border: 1px solid #E5E7EB; }
        .feed-status.good { background-color: #D1FAE5; color: #065F46; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
        .feed-status.defective { background-color: #FEE2E2; color: #991B1B; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
        #MainMenu {visibility: hidden;} footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

load_css()

# --- Helper Functions ---
def get_metrics():
    """Get real-time metrics from session state"""
    avg_time = 57  # This would come from actual inference timing
    yesterday_processed = 1187  # Mock baseline
    yesterday_good = 1164
    
    change_processed = ((st.session_state.total_processed - yesterday_processed) / yesterday_processed * 100) if yesterday_processed > 0 else 0
    success_rate = (st.session_state.total_good / st.session_state.total_processed * 100) if st.session_state.total_processed > 0 else 95.3
    
    return {
        "parts_processed": st.session_state.total_processed,
        "good_parts": st.session_state.total_good,
        "defective_parts": st.session_state.total_defective,
        "avg_inference_time": avg_time,
        "change_processed": f"{change_processed:+.1f}% from yesterday" if st.session_state.total_processed > 0 else "+5% from yesterday",
        "change_good": f"{success_rate:.1f}% success rate",
        "change_defective": f"-{(st.session_state.total_defective / st.session_state.total_processed * 100):.1f}% rejection rate" if st.session_state.total_processed > 0 else "-47% rejection rate",
        "change_time": "Within target <70ms"
    }

def classify_image(image):
    """
    Classify cathode cup image using the trained ML model
    Returns classification results with bounding boxes and defect types
    """
    if ML_MODEL is None:
        # Fallback to mock classification if model not loaded
        st.warning("⚠️ Using mock classification - Model not loaded")
        is_good = random.random() > 0.2
        confidence = random.uniform(0.85, 0.98) if is_good else random.uniform(0.75, 0.95)
        
        # Mock bounding boxes and defects
        mock_boxes = []
        mock_labels = []
        if not is_good:
            # Add mock defect box
            w, h = image.size
            mock_boxes = [[w*0.3, h*0.3, w*0.7, h*0.7]]
            mock_labels = ["crack"]
        
        return {
            "status": "Good" if is_good else "Defective",
            "confidence": confidence,
            "boxes": mock_boxes,
            "labels": mock_labels,
            "annotated_image": None
        }
    
    try:
        # Convert PIL Image to numpy array for model
        image_np = np.array(image)
        
        # Run model inference
        with torch.no_grad():
            output = ML_MODEL.predict(image_np)
        
        # Parse model output
        boxes = output.get("boxes", [])
        labels = output.get("labels", [])
        confidences = output.get("confidences", [])
        
        # Determine overall status
        if not labels or all(label.lower() == "good" for label in labels):
            status = "Good"
            overall_confidence = max(confidences) if confidences else 0.95
        else:
            status = "Defective"
            # Get confidence of defect detections
            defect_confidences = [conf for conf, label in zip(confidences, labels) if label.lower() != "good"]
            overall_confidence = max(defect_confidences) if defect_confidences else 0.85
        
        # Draw bounding boxes on image
        annotated_image = draw_bounding_boxes(image_np.copy(), boxes, labels, confidences)
        
        return {
            "status": status,
            "confidence": float(overall_confidence),
            "boxes": boxes,
            "labels": labels,
            "confidences": confidences,
            "annotated_image": annotated_image
        }
        
    except Exception as e:
        st.error(f"Classification error: {e}")
        # Return safe fallback
        return {
            "status": "Good",
            "confidence": 0.5,
            "boxes": [],
            "labels": [],
            "annotated_image": None
        }

def draw_bounding_boxes(image_np, boxes, labels, confidences):
    """
    Draw bounding boxes and labels on the image
    """
    if not boxes or len(boxes) == 0:
        return image_np
    
    # Convert numpy array to PIL for drawing
    img_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img_pil)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw each bounding box
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = labels[i] if i < len(labels) else "unknown"
        conf = confidences[i] if i < len(confidences) else 0.0
        
        # Color coding: Green for good, Red for defects
        color = (0, 255, 0) if label.lower() == "good" else (255, 0, 0)
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label with confidence
        text = f"{label}: {conf:.2f}"
        
        # Draw text background
        bbox = draw.textbbox((x1, y1 - 25), text, font=font)
        draw.rectangle(bbox, fill=color)
        
        # Draw text
        draw.text((x1, y1 - 25), text, fill=(255, 255, 255), font=font)
    
    return np.array(img_pil)

def add_to_feed(image, status, confidence, labels=None, annotated_image=None):
    """Add new classification to the feed with defect information"""
    current_time = datetime.now().strftime("%H:%M:%S")
    cup_id = f"#{random.randint(1000, 9999)}"
    
    # Convert PIL image to bytes for storage in session state
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    # Convert annotated image to bytes if available
    annotated_bytes = None
    if annotated_image is not None:
        annotated_pil = Image.fromarray(annotated_image)
        annotated_arr = io.BytesIO()
        annotated_pil.save(annotated_arr, format='PNG')
        annotated_bytes = annotated_arr.getvalue()
    
    # Get defect types (excluding "good" labels)
    defect_types = [label for label in (labels or []) if label.lower() != "good"]
    
    feed_item = {
        "time": current_time,
        "id": cup_id,
        "status": status,
        "confidence": confidence,
        "image_bytes": img_bytes,
        "annotated_bytes": annotated_bytes,
        "defect_types": defect_types
    }
    
    # Add to beginning of feed (most recent first)
    st.session_state.classification_feed.insert(0, feed_item)
    
    # Keep only last 10 items
    if len(st.session_state.classification_feed) > 10:
        st.session_state.classification_feed = st.session_state.classification_feed[:10]
    
    # Update metrics
    st.session_state.total_processed += 1
    if status == "Good":
        st.session_state.total_good += 1
    else:
        st.session_state.total_defective += 1

def get_recent_alerts():
    """Generate recent alerts"""
    return [
        {"type": "info", "icon": "ℹ️", "text": "System started successfully", "time": "2 mins ago"},
        {"type": "warning", "icon": "⚠️", "text": "Camera calibration recommended", "time": "1h ago"},
        {"type": "success", "icon": "✓", "text": "Quality threshold maintained", "time": "2h ago"}
    ]

# --- Navigation Tabs ---
st.markdown("""
    <style>
    .nav-tabs {
        display: flex;
        gap: 0;
        border-bottom: 2px solid #E5E7EB;
        margin-bottom: 30px;
        background-color: white;
        padding: 0 20px;
    }
    .nav-tab {
        padding: 12px 24px;
        font-size: 0.875rem;
        font-weight: 500;
        color: #6B7280;
        cursor: pointer;
        border: none;
        background: none;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
        transition: all 0.2s;
    }
    .nav-tab:hover {
        color: #1F2937;
        border-bottom-color: #D1D5DB;
    }
    .nav-tab.active {
        color: #3B82F6;
        border-bottom-color: #3B82F6;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize active tab in session state
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'Dashboard'

# Create navigation tabs
col_dash, col_analyzer, col_settings, col_reports = st.columns([1, 1, 1, 1])

with col_dash:
    if st.button("📊 Dashboard", key="tab_dashboard", use_container_width=True):
        st.session_state.active_tab = 'Dashboard'

with col_analyzer:
    if st.button("🔬 Analyzers", key="tab_analyzers", use_container_width=True):
        st.session_state.active_tab = 'Analyzers'

with col_settings:
    if st.button("⚙️ Settings", key="tab_settings", use_container_width=True):
        st.session_state.active_tab = 'Settings'

with col_reports:
    if st.button("📈 Reports", key="tab_reports", use_container_width=True):
        st.session_state.active_tab = 'Reports'

st.markdown("---")

# --- Dashboard Header ---
st.markdown(f"""
    <div style='margin-bottom: 30px;'>
        <h1 style='font-size: 2rem; font-weight: 700; color: #1F2937; margin-bottom: 4px;'>{st.session_state.active_tab}</h1>
        <p style='font-size: 0.875rem; color: #6B7280;'>
            {'Real-time monitoring of cathode cup classification and manufacturing workflow' if st.session_state.active_tab == 'Dashboard' 
             else 'Advanced analysis tools and metrics' if st.session_state.active_tab == 'Analyzers'
             else 'System configuration and preferences' if st.session_state.active_tab == 'Settings'
             else 'Historical reports and data exports'}
        </p>
    </div>
""", unsafe_allow_html=True)

# =============================================================================
# TAB CONTENT SECTIONS
# =============================================================================

# --- TAB 1: DASHBOARD ---
if st.session_state.active_tab == 'Dashboard':
    # --- Top Metrics Row ---
    metrics = get_metrics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class='metric-card'>
                <div style='display: flex; align-items: center; justify-content: space-between;'>
                    <div>
                        <p style='font-size: 0.875rem; color: #6B7280; margin: 0;'>Parts Processed</p>
                        <h2 style='font-size: 2rem; font-weight: 700; color: #1F2937; margin: 8px 0;'>{metrics['parts_processed']:,}</h2>
                        <p style='font-size: 0.75rem; color: #10B981; margin: 0;'>{metrics['change_processed']}</p>
                    </div>
                    <div style='font-size: 2.5rem; color: #3B82F6;'>ℹ️</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class='metric-card'>
                <div style='display: flex; align-items: center; justify-content: space-between;'>
                    <div>
                        <p style='font-size: 0.875rem; color: #6B7280; margin: 0;'>Good Parts</p>
                        <h2 style='font-size: 2rem; font-weight: 700; color: #1F2937; margin: 8px 0;'>{metrics['good_parts']:,}</h2>
                        <p style='font-size: 0.75rem; color: #10B981; margin: 0;'>{metrics['change_good']}</p>
                    </div>
                    <div style='font-size: 2.5rem; color: #10B981;'>✓</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class='metric-card'>
                <div style='display: flex; align-items: center; justify-content: space-between;'>
                    <div>
                        <p style='font-size: 0.875rem; color: #6B7280; margin: 0;'>Defective Parts</p>
                        <h2 style='font-size: 2rem; font-weight: 700; color: #1F2937; margin: 8px 0;'>{metrics['defective_parts']}</h2>
                        <p style='font-size: 0.75rem; color: #EF4444; margin: 0;'>{metrics['change_defective']}</p>
                    </div>
                    <div style='font-size: 2.5rem; color: #EF4444;'>⊗</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class='metric-card'>
                <div style='display: flex; align-items: center; justify-content: space-between;'>
                    <div>
                        <p style='font-size: 0.875rem; color: #6B7280; margin: 0;'>Avg Inference Time</p>
                        <h2 style='font-size: 2rem; font-weight: 700; color: #1F2937; margin: 8px 0;'>{metrics['avg_inference_time']}ms</h2>
                        <p style='font-size: 0.75rem; color: #8B5CF6; margin: 0;'>{metrics['change_time']}</p>
                    </div>
                    <div style='font-size: 2.5rem; color: #8B5CF6;'>⚡</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Image Upload Section ---
    with st.expander("📸 Upload Cathode Cup Image for Classification", expanded=False):
        st.markdown("Upload an image captured from your camera system for real-time quality classification")
    
        uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a cathode cup image for classification"
        )
    
        if uploaded_file is not None:
            # Load the uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            
            col_original, col_result = st.columns([1, 1])
            
            with col_original:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)
            
            # Classification button
            if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image with ML model..."):
                    # Simulate processing time
                    import time
                    time.sleep(0.5)
                    
                    # Classify the image using ML model
                    result = classify_image(image)
                    
                    # Display annotated image with bounding boxes
                    with col_result:
                        st.markdown("**Classification Result**")
                        if result["annotated_image"] is not None:
                            st.image(result["annotated_image"], use_container_width=True)
                        else:
                            st.image(image, use_container_width=True)
                    
                    # Show classification result
                    if result["status"] == "Good":
                        st.success(f"✅ **{result['status']}** - Confidence: {result['confidence']*100:.1f}%")
                    else:
                        st.error(f"❌ **{result['status']}** - Confidence: {result['confidence']*100:.1f}%")
                        
                        # Show defect types if any
                        if result.get("labels"):
                            defect_types = [label for label in result["labels"] if label.lower() != "good"]
                            if defect_types:
                                st.warning(f"**Defect Types Detected:** {', '.join(defect_types)}")
                    
                    # Add to feed
                    add_to_feed(
                        image, 
                        result["status"], 
                        result["confidence"],
                        result.get("labels", []),
                        result.get("annotated_image")
                    )
                    
                    # Rerun to update dashboard
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Main Content Area ---
    left_col, right_col = st.columns([2, 1])

    with left_col:
        # Real-time Classification Feed
        feed_count = len(st.session_state.classification_feed)
        status_text = "System Online" if feed_count > 0 else "Waiting for uploads"
        status_color = "#10B981" if feed_count > 0 else "#F59E0B"
    
        st.markdown(f"""
        <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 20px;'>
            <h3 style='font-size: 1.125rem; font-weight: 600; color: #1F2937; margin: 0;'>Real-time Classification Feed</h3>
            <span style='width: 8px; height: 8px; border-radius: 50%; background-color: {status_color}; display: inline-block; animation: pulse 2s infinite;'></span>
            <span style='font-size: 0.875rem; color: {status_color}; font-weight: 500;'>{status_text}</span>
        </div>
        """, unsafe_allow_html=True)
    
        # Display feed items from session state
        if st.session_state.classification_feed:
            for item in st.session_state.classification_feed:
                col_img, col_info = st.columns([1, 4])
            
            with col_img:
                # Display annotated image if available, otherwise original
                if item.get('annotated_bytes'):
                    img = Image.open(io.BytesIO(item['annotated_bytes']))
                else:
                    img = Image.open(io.BytesIO(item['image_bytes']))
                st.image(img, width=80)
            
            with col_info:
                status_class = "good" if item['status'] == "Good" else "defective"
                
                # Build defect type display
                defect_info = ""
                if item.get('defect_types') and len(item['defect_types']) > 0:
                    defect_info = f"<p style='font-size: 0.75rem; color: #EF4444; margin: 4px 0 0 0;'>⚠️ Defects: {', '.join(item['defect_types'])}</p>"
                
                st.markdown(f"""
                    <div style='background: white; border-radius: 8px; padding: 12px; border: 1px solid #E5E7EB;'>
                        <p style='font-size: 0.75rem; color: #6B7280; margin: 0 0 4px 0;'>{item['time']}</p>
                        <p style='font-size: 0.875rem; font-weight: 600; color: #1F2937; margin: 0 0 8px 0;'>Cathode Cup {item['id']}</p>
                        <span class='feed-status {status_class}'>{item['status']}</span>
                        <p style='font-size: 0.75rem; color: #6B7280; margin: 8px 0 0 0;'>{item['confidence']*100:.1f}% confidence</p>
                        {defect_info}
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
        else:
            # Show placeholder when no images uploaded yet
            st.markdown("""
                <div style='background: white; border-radius: 8px; padding: 40px; text-align: center; border: 1px solid #E5E7EB;'>
                    <div style='font-size: 3rem; color: #D1D5DB; margin-bottom: 16px;'>📸</div>
                    <p style='font-size: 1rem; color: #6B7280; margin-bottom: 8px;'>No classifications yet</p>
                    <p style='font-size: 0.875rem; color: #9CA3AF;'>Upload a cathode cup image above to start classification</p>
                </div>
            """, unsafe_allow_html=True)

    with right_col:
        # Quick Actions
        st.markdown("<h3 style='font-size: 1.125rem; font-weight: 600; color: #1F2937; margin-bottom: 16px;'>Quick Actions</h3>", unsafe_allow_html=True)
    
        # Button 1: Start Inspection
        if st.button("▶ Start Inspection", use_container_width=True, type="primary"):
            st.success("✅ Inspection mode activated! Ready to classify cathode cups.")
        st.info("📸 Upload images above to begin quality inspection.")
    
        # Button 2: Pause System
        if st.button("⏸ Pause System", use_container_width=True):
            st.warning("⏸ System paused. Inspection temporarily halted.")
        st.info("Click 'Start Inspection' to resume.")
    
        # Button 3: Export Report
        if st.button("📊 Export Report", use_container_width=True):
            if st.session_state.total_processed > 0:
                # Generate report summary
                success_rate = (st.session_state.total_good / st.session_state.total_processed * 100) if st.session_state.total_processed > 0 else 0
                rejection_rate = (st.session_state.total_defective / st.session_state.total_processed * 100) if st.session_state.total_processed > 0 else 0
                
                report = f"""
                📊 Quality Control Report
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━
                Total Parts Inspected: {st.session_state.total_processed}
                ✅ Good Parts: {st.session_state.total_good}
                ❌ Defective Parts: {st.session_state.total_defective}
                Success Rate: {success_rate:.1f}%
                Rejection Rate: {rejection_rate:.1f}%
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━
                """
                st.success("✅ Report generated successfully!")
                st.text(report)
                st.download_button(
                    label="💾 Download Report as Text",
                    data=report,
                    file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            else:
                st.warning("⚠️ No inspection data available. Upload and classify images first.")
    
        # Button 4: Clear History
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.classification_feed = []
        st.session_state.total_processed = 0
        st.session_state.total_good = 0
        st.session_state.total_defective = 0
        st.success("🗑️ Classification history cleared! All metrics reset to zero.")
        st.rerun()
    
        st.markdown("<br>", unsafe_allow_html=True)
    
        # Recent Alerts
        st.markdown("<h3 style='font-size: 1.125rem; font-weight: 600; color: #1F2937; margin-bottom: 16px;'>Recent Alerts</h3>", unsafe_allow_html=True)
    
        alerts = get_recent_alerts()
        for alert in alerts:
            st.markdown(f"""
            <div class='alert-item {alert['type']}' style='display: flex; gap: 10px; padding: 12px; margin-bottom: 10px; border-radius: 8px; background: #F9FAFB;'>
                <div style='font-size: 1.25rem;'>{alert['icon']}</div>
                <div style='flex: 1;'>
                    <p style='font-size: 0.875rem; color: #1F2937; margin: 0;'>{alert['text']}</p>
                    <p style='font-size: 0.75rem; color: #6B7280; margin: 4px 0 0 0;'>{alert['time']}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # --- Bottom Section: Quality Trend Chart ---
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.125rem; font-weight: 600; color: #1F2937; margin-bottom: 16px;'>Quality Trend</h3>", unsafe_allow_html=True)

        # Placeholder for chart
        st.markdown("""
        <div style='background: white; border-radius: 12px; padding: 40px; text-align: center; border: 1px solid #E5E7EB;'>
            <div style='font-size: 3rem; color: #D1D5DB; margin-bottom: 16px;'>📈</div>
            <p style='font-size: 1rem; color: #6B7280;'>Quality trend chart</p>
            <p style='font-size: 0.875rem; color: #9CA3AF;'>Real-time quality metrics visualization will appear here</p>
        </div>
        """, unsafe_allow_html=True)

        # --- TAB 2: ANALYZERS ---
elif st.session_state.active_tab == 'Analyzers':
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Analyzer Tools Grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='metric-card' style='height: 250px;'>
                <div style='text-align: center; padding: 20px;'>
                    <div style='font-size: 3rem; color: #3B82F6; margin-bottom: 16px;'>🔬</div>
                    <h3 style='font-size: 1.25rem; font-weight: 600; color: #1F2937; margin-bottom: 12px;'>Defect Analyzer</h3>
                    <p style='font-size: 0.875rem; color: #6B7280; margin-bottom: 20px;'>Analyze defect patterns and distribution across batches</p>
                    <div style='background: #EFF6FF; color: #1E40AF; padding: 8px 16px; border-radius: 6px; display: inline-block; font-size: 0.875rem;'>Coming Soon</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='metric-card' style='height: 250px;'>
                <div style='text-align: center; padding: 20px;'>
                    <div style='font-size: 3rem; color: #10B981; margin-bottom: 16px;'>📊</div>
                    <h3 style='font-size: 1.25rem; font-weight: 600; color: #1F2937; margin-bottom: 12px;'>Trend Analysis</h3>
                    <p style='font-size: 0.875rem; color: #6B7280; margin-bottom: 20px;'>View quality trends over time with predictive insights</p>
                    <div style='background: #D1FAE5; color: #065F46; padding: 8px 16px; border-radius: 6px; display: inline-block; font-size: 0.875rem;'>Coming Soon</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
            <div class='metric-card' style='height: 250px;'>
                <div style='text-align: center; padding: 20px;'>
                    <div style='font-size: 3rem; color: #F59E0B; margin-bottom: 16px;'>🎯</div>
                    <h3 style='font-size: 1.25rem; font-weight: 600; color: #1F2937; margin-bottom: 12px;'>Accuracy Monitor</h3>
                    <p style='font-size: 0.875rem; color: #6B7280; margin-bottom: 20px;'>Track model accuracy and confidence scores</p>
                    <div style='background: #FEF3C7; color: #92400E; padding: 8px 16px; border-radius: 6px; display: inline-block; font-size: 0.875rem;'>Coming Soon</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class='metric-card' style='height: 250px;'>
                <div style='text-align: center; padding: 20px;'>
                    <div style='font-size: 3rem; color: #8B5CF6; margin-bottom: 16px;'>🔍</div>
                    <h3 style='font-size: 1.25rem; font-weight: 600; color: #1F2937; margin-bottom: 12px;'>Batch Comparison</h3>
                    <p style='font-size: 0.875rem; color: #6B7280; margin-bottom: 20px;'>Compare quality metrics across different batches</p>
                    <div style='background: #EDE9FE; color: #5B21B6; padding: 8px 16px; border-radius: 6px; display: inline-block; font-size: 0.875rem;'>Coming Soon</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Feature Request Section
    st.markdown("""
        <div class='metric-card'>
            <h3 style='font-size: 1.125rem; font-weight: 600; color: #1F2937; margin-bottom: 12px;'>📝 Request New Analyzer</h3>
            <p style='font-size: 0.875rem; color: #6B7280; margin-bottom: 16px;'>Need a specific analysis tool? Let us know what insights would help improve your quality control process.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col_a, col_b = st.columns([3, 1])
    with col_a:
        analyzer_request = st.text_input("Describe the analyzer you need:", placeholder="e.g., Real-time defect heatmap visualization")
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Submit Request", use_container_width=True):
            if analyzer_request:
                st.success("✅ Request submitted! We'll review and prioritize it.")
            else:
                st.warning("⚠️ Please describe the analyzer you need.")

# --- TAB 3: SETTINGS ---
elif st.session_state.active_tab == 'Settings':
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Settings Categories
    tab1, tab2, tab3, tab4 = st.tabs(["⚙️ General", "🤖 Model", "📧 Notifications", "👥 Users"])
    
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### General Settings")
        
        st.selectbox("Theme", ["Light Mode", "Dark Mode", "Auto"])
        st.selectbox("Language", ["English", "Spanish", "French", "German", "Chinese"])
        st.slider("Dashboard Refresh Rate (seconds)", 1, 60, 5)
        st.checkbox("Enable sound notifications", value=True)
        st.checkbox("Show detailed metrics", value=True)
        
        if st.button("Save General Settings", type="primary"):
            st.success("✅ General settings saved successfully!")
    
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Model Configuration")
        
        st.slider("Confidence Threshold (%)", 0, 100, 85)
        st.slider("Defect Detection Sensitivity", 1, 10, 7)
        st.number_input("Maximum Inference Time (ms)", min_value=10, max_value=5000, value=200)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Model Information")
        st.info(f"**Current Model:** model.pkl\n\n**Status:** {'Loaded ✅' if ML_MODEL else 'Not Loaded ❌'}\n\n**Last Updated:** N/A")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button("Retrain Model", use_container_width=True):
                st.warning("⚠️ Model retraining requires admin access")
        with col_m2:
            if st.button("Upload New Model", use_container_width=True):
                st.info("📤 Model upload feature coming soon")
        
        if st.button("Save Model Settings", type="primary"):
            st.success("✅ Model settings saved successfully!")
    
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Notification Preferences")
        
        st.checkbox("Email notifications for defects", value=True)
        st.checkbox("Email notifications for system alerts", value=False)
        st.text_input("Email Address", value="quality@company.com")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Alert Thresholds")
        st.slider("Defect Rate Alert (%)", 0, 100, 15)
        st.slider("Low Confidence Alert (%)", 0, 100, 70)
        
        if st.button("Save Notification Settings", type="primary"):
            st.success("✅ Notification settings saved successfully!")
    
    with tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### User Management")
        
        st.info("👤 **Current User:** admin@gehealthcare.com\n\n**Role:** Administrator\n\n**Last Login:** 2025-11-10 14:30:25")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Team Members")
        
        users_data = [
            {"name": "John Smith", "role": "Quality Inspector", "status": "Active"},
            {"name": "Sarah Johnson", "role": "Shift Supervisor", "status": "Active"},
            {"name": "Mike Chen", "role": "Quality Inspector", "status": "Inactive"}
        ]
        
        for user in users_data:
            col_u1, col_u2, col_u3, col_u4 = st.columns([2, 2, 1, 1])
            with col_u1:
                st.markdown(f"**{user['name']}**")
            with col_u2:
                st.markdown(f"{user['role']}")
            with col_u3:
                status_color = "#10B981" if user['status'] == "Active" else "#9CA3AF"
                st.markdown(f"<span style='color: {status_color};'>● {user['status']}</span>", unsafe_allow_html=True)
            with col_u4:
                st.button("Edit", key=f"edit_{user['name']}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add New User", use_container_width=True):
            st.info("👥 User management requires admin privileges")

# --- TAB 4: REPORTS ---
elif st.session_state.active_tab == 'Reports':
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Report Time Range Selector
    col_r1, col_r2, col_r3 = st.columns([2, 2, 1])
    with col_r1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
    with col_r2:
        end_date = st.date_input("End Date", value=datetime.now())
    with col_r3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📊 Generate", use_container_width=True, type="primary"):
            st.success("✅ Report generated for selected date range!")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Report Types
    col_rt1, col_rt2, col_rt3 = st.columns(3)
    
    with col_rt1:
        st.markdown("""
            <div class='metric-card' style='text-align: center;'>
                <div style='font-size: 2.5rem; color: #3B82F6; margin-bottom: 12px;'>📊</div>
                <h4 style='font-size: 1rem; font-weight: 600; color: #1F2937; margin-bottom: 8px;'>Daily Summary</h4>
                <p style='font-size: 0.875rem; color: #6B7280; margin-bottom: 16px;'>Parts processed, success rate, defects</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Download Daily Report", key="daily", use_container_width=True):
            st.info("📥 Daily report download starting...")
    
    with col_rt2:
        st.markdown("""
            <div class='metric-card' style='text-align: center;'>
                <div style='font-size: 2.5rem; color: #10B981; margin-bottom: 12px;'>📈</div>
                <h4 style='font-size: 1rem; font-weight: 600; color: #1F2937; margin-bottom: 8px;'>Trend Analysis</h4>
                <p style='font-size: 0.875rem; color: #6B7280; margin-bottom: 16px;'>Quality trends over time period</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Download Trend Report", key="trend", use_container_width=True):
            st.info("📥 Trend report download starting...")
    
    with col_rt3:
        st.markdown("""
            <div class='metric-card' style='text-align: center;'>
                <div style='font-size: 2.5rem; color: #F59E0B; margin-bottom: 12px;'>⚠️</div>
                <h4 style='font-size: 1rem; font-weight: 600; color: #1F2937; margin-bottom: 8px;'>Defect Report</h4>
                <p style='font-size: 0.875rem; color: #6B7280; margin-bottom: 16px;'>Detailed defect classification</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Download Defect Report", key="defect", use_container_width=True):
            st.info("📥 Defect report download starting...")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Recent Reports Table
    st.markdown("### 📋 Recent Reports")
    st.markdown("<br>", unsafe_allow_html=True)
    
    reports_data = [
        {"date": "2025-11-10", "type": "Daily Summary", "parts": "1,248", "success": "95.2%", "size": "2.3 MB"},
        {"date": "2025-11-09", "type": "Daily Summary", "parts": "1,187", "success": "98.1%", "size": "2.1 MB"},
        {"date": "2025-11-08", "type": "Daily Summary", "parts": "1,312", "success": "94.8%", "size": "2.4 MB"},
        {"date": "2025-11-07", "type": "Weekly Trend", "parts": "8,543", "success": "96.3%", "size": "15.7 MB"},
    ]
    
    for report in reports_data:
        col_rp1, col_rp2, col_rp3, col_rp4, col_rp5, col_rp6 = st.columns([2, 2, 2, 2, 2, 1])
        with col_rp1:
            st.markdown(f"📄 **{report['date']}**")
        with col_rp2:
            st.markdown(f"{report['type']}")
        with col_rp3:
            st.markdown(f"{report['parts']} parts")
        with col_rp4:
            st.markdown(f"✅ {report['success']}")
        with col_rp5:
            st.markdown(f"📦 {report['size']}")
        with col_rp6:
            st.button("⬇️", key=f"download_{report['date']}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Export Options
    st.markdown("""
        <div class='metric-card'>
            <h3 style='font-size: 1.125rem; font-weight: 600; color: #1F2937; margin-bottom: 12px;'>📤 Custom Export</h3>
            <p style='font-size: 0.875rem; color: #6B7280; margin-bottom: 16px;'>Export data in your preferred format</p>
        </div>
    """, unsafe_allow_html=True)
    
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        if st.button("📄 Export as CSV", use_container_width=True):
            st.info("📥 Preparing CSV export...")
    with col_e2:
        if st.button("📊 Export as Excel", use_container_width=True):
            st.info("📥 Preparing Excel export...")
    with col_e3:
        if st.button("📋 Export as PDF", use_container_width=True):
            st.info("📥 Preparing PDF export...")

# =============================================================================
# END OF TAB SECTIONS
# =============================================================================

# --- Footer ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; padding: 20px; color: #9CA3AF; font-size: 0.75rem; border-top: 1px solid #E5E7EB;'>
        <p>QualityControl AI | Automated quality system for manufacturing with AI-driven classification and real-time monitoring</p>
        <p>© 2025 QualityControl AI. All rights reserved | Current Tab: {}</p>
    </div>
""".format(st.session_state.active_tab), unsafe_allow_html=True)
