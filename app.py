import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Set up the page layout
st.set_page_config(page_title="Kidney Stone AI", page_icon="🩺", layout="wide")

@st.cache_resource
def load_model():
    # Pointing EXACTLY to the model you are training right now
    return YOLO("runs/detect/AI_Demo/kidney_stone_test4/weights/best.pt")

st.title("🩺 AI-Powered Kidney Stone Detection")
st.markdown("**Minor Project Demo:** Single-Stage YOLOv10 Localization Pipeline")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.warning("⏳ Model is still training! The AI is currently baking in your other terminal. You can preview the UI, but wait for training to finish before running the detection.")
    model_loaded = False

# Sidebar for our "Academic/Clinical" features
st.sidebar.header("⚙️ Model Parameters")
confidence = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.05, max_value=0.95, value=0.25, step=0.05,
    help="Lower values increase sensitivity (Recall) but may increase false positives."
)

# Simulated pixel spacing for our Size Estimator (e.g., 1 pixel = 0.8mm)
pixel_spacing = st.sidebar.number_input("CT Pixel Spacing (mm/px)", value=0.80, step=0.05)

uploaded_file = st.file_uploader("Upload a CT scan (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Original Scan")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("AI Analysis")
        if st.button("Run Detection Pipeline", type="primary"):
            with st.spinner("Analyzing tissue density..."):
                # Run YOLO with the user's custom slider confidence
                results = model.predict(source=image, conf=confidence)
                result = results[0]
                boxes = result.boxes
                
                if len(boxes) == 0:
                    st.success("✅ **Clear:** No kidney stones detected at the current threshold.")
                else:
                    st.error(f"⚠️ **Alert:** Detected {len(boxes)} kidney stone(s)!")
                    
                    # Draw bounding boxes
                    res_plotted = result.plot() 
                    res_rgb = res_plotted[:, :, ::-1] 
                    st.image(res_rgb, use_container_width=True)
                    
                    # The Clinical Size Estimator
                    st.markdown("### 📊 Clinical Size Estimation")
                    for i, box in enumerate(boxes):
                        # Get width and height of the bounding box
                        w, h = box.xywh[0][2].item(), box.xywh[0][3].item()
                        
                        # Calculate physical size based on pixel spacing
                        width_mm = w * pixel_spacing
                        height_mm = h * pixel_spacing
                        max_dimension = max(width_mm, height_mm)
                        
                        st.info(f"**Stone {i+1}:** Estimated max dimension is **~{max_dimension:.1f} mm**.")
                        
                        if max_dimension > 5.0:
                            st.warning(f"💡 *Clinical Note for Stone {i+1}: Stones larger than 5mm have a decreased chance of passing naturally and may require urological intervention.*")