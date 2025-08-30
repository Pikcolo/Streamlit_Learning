import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import requests
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Image Processing Studio", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .section-header {
        background-color: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stats-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .download-section {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Utility function
def _convert_processed_to_bytes(img):
    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            pil = Image.fromarray(img)
        else:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        pil = img
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

# Header
st.markdown('<div class="main-header"><h1>🖼️ Steamlit Learning!</h1></div>', unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.markdown('<div class="section-header"><h3>📁 Image Source</h3></div>', unsafe_allow_html=True)
    
    source = st.radio(
        "Choose your image source:",
        ("📷 Webcam", "🌐 Image URL", "📁 Upload File"),
        help="Select how you want to input your image"
    )
    
    img = None
    
    if source == "📷 Webcam":
        st.markdown("**Take a photo using your camera:**")
        camera_file = st.camera_input("📸 Capture Image")
        if camera_file:
            img = Image.open(camera_file).convert("RGB")
            st.success("✅ Image captured successfully!")

    elif source == "🌐 Image URL":
        st.markdown("**Enter image URL:**")
        url = st.text_input(
            "Image URL (http/https)",
            placeholder="https://example.com/image.jpg",
            help="Enter a direct link to an image file"
        )
        if url:
            try:
                with st.spinner("Loading image..."):
                    resp = requests.get(url, timeout=8)
                    resp.raise_for_status()
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                st.success("✅ Image loaded successfully!")
            except Exception as e:
                st.error(f"❌ Couldn't load image: {str(e)}")

    else:  # Upload File
        st.markdown("**Upload your image file:**")
        upload = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
        )
        if upload:
            img = Image.open(upload).convert("RGB")
            st.success("✅ Image uploaded successfully!")

    # Processing Options
    st.markdown("---")
    st.markdown('<div class="section-header"><h3>⚙️ Processing Options</h3></div>', unsafe_allow_html=True)
    
    mode = st.selectbox(
        "🎨 Processing Mode",
        ["Grayscale", "Blur", "Canny Edges", "Threshold", "Custom Mix"],
        help="Choose the type of image processing to apply"
    )

    # Advanced options in an expander
    with st.expander("🔧 Advanced Settings", expanded=True):
        # Resize option
        resize = st.toggle(
            "📏 Auto-resize (max width: 800px)",
            value=True,
            help="Automatically resize large images for better performance"
        )
        
        # Mode-specific parameters
        if mode in ["Blur", "Custom Mix"]:
            k = st.slider("🔵 Blur Kernel Size", 1, 31, 5, 2, help="Larger values create more blur")
            if k % 2 == 0:
                k += 1
        else:
            k = 1

        if mode in ["Canny Edges", "Custom Mix"]:
            st.markdown("**Edge Detection Settings:**")
            low = st.slider("📉 Low Threshold", 0, 500, 50, help="Lower values detect more edges")
            high = st.slider("📈 High Threshold", 0, 500, 150, help="Higher values are more selective")
        else:
            low, high = 50, 150

        if mode in ["Threshold", "Custom Mix"]:
            th = st.slider("🎯 Threshold Value", 0, 255, 127, help="Pixel intensity cutoff point")
        else:
            th = 127

# Main content area
if img is not None:
    # Apply resize if requested
    original_size = img.size
    if resize and img.size[0] > 800:
        w, h = img.size
        new_h = int(h * (800.0 / w))
        img = img.resize((800, new_h))
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["🖼️ Image Preview", "📊 Analysis", "💾 Export"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header"><h3>📷 Original Image</h3></div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True, caption=f"Size: {img.size[0]}×{img.size[1]} pixels")
            
        with col2:
            st.markdown('<div class="section-header"><h3>✨ Processed Image</h3></div>', unsafe_allow_html=True)
            
            # Convert to OpenCV format
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Apply processing
            processed = None
            if mode == "Grayscale":
                processed = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            elif mode == "Blur":
                processed = cv2.GaussianBlur(cv_img, (k, k), 0)
            elif mode == "Canny Edges":
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                processed = cv2.Canny(gray, low, high)
            elif mode == "Threshold":
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                _, processed = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
            elif mode == "Custom Mix":
                blur = cv2.GaussianBlur(cv_img, (k, k), 0)
                gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, low, high)
                _, processed = cv2.threshold(edges, th, 255, cv2.THRESH_BINARY)
            else:
                processed = cv_img.copy()
            
            # Display processed image
            if len(processed.shape) == 2:
                st.image(processed, clamp=True, channels="GRAY", use_container_width=True, 
                        caption=f"Mode: {mode} | Grayscale Output")
            else:
                st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True,
                        caption=f"Mode: {mode} | Color Output")
    
    with tab2:
        st.markdown('<div class="section-header"><h3>📊 Image Analysis</h3></div>', unsafe_allow_html=True)
        
        # Create histogram and stats
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            
            if len(processed.shape) == 2:
                # Grayscale histogram
                hist = cv2.calcHist([processed], [0], None, [256], [0, 256]).flatten()
                ax.plot(hist, color='gray', linewidth=2)
                ax.set_title("📈 Pixel Intensity Distribution", fontsize=14, pad=20)
                ax.set_xlabel("Intensity Value")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)
            else:
                # Color histogram
                colors = ['blue', 'green', 'red']
                labels = ['Blue', 'Green', 'Red']
                chans = cv2.split(processed)
                
                for chan, color, label in zip(chans, colors, labels):
                    hist = cv2.calcHist([chan], [0], None, [256], [0, 256]).flatten()
                    ax.plot(hist, color=color, label=label, linewidth=2, alpha=0.7)
                
                ax.set_title("📈 RGB Channel Distribution", fontsize=14, pad=20)
                ax.set_xlabel("Intensity Value")
                ax.set_ylabel("Frequency")
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown('<div class="stats-container">', unsafe_allow_html=True)
            st.markdown("### 📋 Image Statistics")
            
            if len(processed.shape) == 2:
                stats = {
                    "📏 Dimensions": f"{processed.shape[1]} × {processed.shape[0]}",
                    "🎯 Mean Intensity": f"{processed.mean():.2f}",
                    "📊 Std Deviation": f"{processed.std():.2f}",
                    "⚫ Non-zero Pixels": f"{(processed > 0).sum():,}",
                    "⚪ Zero Pixels": f"{(processed == 0).sum():,}",
                    "📈 Max Intensity": f"{processed.max()}",
                    "📉 Min Intensity": f"{processed.min()}"
                }
            else:
                chans = cv2.split(processed)
                stats = {
                    "📏 Dimensions": f"{processed.shape[1]} × {processed.shape[0]}",
                    "🔴 Red Mean": f"{np.mean(chans[2]):.2f}",
                    "🟢 Green Mean": f"{np.mean(chans[1]):.2f}",
                    "🔵 Blue Mean": f"{np.mean(chans[0]):.2f}",
                    "📊 Overall Std": f"{np.std(processed):.2f}",
                    "💾 File Size": f"~{processed.size * processed.itemsize / 1024:.1f} KB"
                }
            
            for key, value in stats.items():
                st.metric(key, value)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="section-header"><h3>💾 Export Options</h3></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="download-section">', unsafe_allow_html=True)
            st.markdown("### 🎉 Your processed image is ready!")
            
            # Show processing summary
            st.info(f"""
            **Processing Summary:**
            - Mode: {mode}
            - Original size: {original_size[0]}×{original_size[1]} pixels
            - Final size: {processed.shape[1] if len(processed.shape) > 1 else processed.shape[1]}×{processed.shape[0]} pixels
            - Format: {'Grayscale' if len(processed.shape) == 2 else 'Color'}
            """)
            
            # Download button
            st.download_button(
                label="📥 Download Processed Image",
                data=_convert_processed_to_bytes(processed),
                file_name=f"processed_{mode.lower().replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True,
                type="primary"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional export options
            st.markdown("---")
            st.markdown("### 🔧 Advanced Export")
            
            export_format = st.selectbox(
                "Choose export format:",
                ["PNG (best quality)", "JPEG (smaller size)", "BMP (uncompressed)"]
            )
            
            if export_format.startswith("JPEG"):
                quality = st.slider("JPEG Quality", 50, 100, 85, help="Higher values = better quality, larger file size")

else:
    # Welcome screen when no image is loaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background-color: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
        <h2>👋 Welcome to Steamlit Learning!</h2>
        <p style="font-size: 1.1em; color: #666;">
            Get started by selecting an image source from the sidebar on the left.
        </p>
        <div style="margin-top: 2rem;">
    </div>
    """, unsafe_allow_html=True)