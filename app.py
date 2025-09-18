import streamlit as st
import os
import tempfile
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np
import faiss
from transformers import AutoModel
try:
    from transformers import AutoImageProcessor
    processor_class = AutoImageProcessor
except ImportError:
    try:
        from transformers import AutoFeatureExtractor
        processor_class = AutoFeatureExtractor
    except ImportError:
        from transformers import ViTFeatureExtractor
        processor_class = ViTFeatureExtractor
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import cv2
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine

# Set page config
st.set_page_config(
    page_title="Jersey Pattern Matcher",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        text-align: center;
        color: #A23B72;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .results-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load YOLO and DINO models"""
    try:
        # Load YOLO model
        yolo_model = YOLO("models/deepfashion2_yolov8s-seg.pt")
        
        # Load DINO model
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        processor = processor_class.from_pretrained('facebook/dinov2-small', use_fast=True)
        dino_model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)
        
        return yolo_model, dino_model, processor, device
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

@st.cache_resource
def load_index():
    """Load the pre-built FAISS index and image paths"""
    try:
        index = faiss.read_index("index/vector.index")
        
        with open("index/vector.index.paths.txt", "r") as f:
            image_paths = [line.strip() for line in f.readlines()]
        
        return index, image_paths
    except Exception as e:
        st.error(f"Error loading index: {str(e)}")
        return None, None

def extract_yolo_coordinates(image, yolo_model):
    """Extract coordinates from YOLO model"""
    try:
        device_id = 0 if torch.cuda.is_available() else "cpu"
        results = yolo_model(image, device=device_id, verbose=False)[0]
        
        polygons = []
        if hasattr(results, "masks") and results.masks is not None and hasattr(results.masks, "xy"):
            for mask in results.masks.xy:
                polygons.append(mask.tolist())
        
        return polygons
    except Exception as e:
        st.error(f"Error in YOLO processing: {str(e)}")
        return []

def crop_image_with_polygon(image, polygons):
    """Crop image using polygon coordinates"""
    if not polygons or len(polygons) == 0:
        return image
    
    try:
        # Use the first polygon (largest detection)
        area = polygons[0]
        width, height = image.size
        
        # Get bounding rectangle from polygon points
        xs = [x for x, y in area]
        ys = [y for x, y in area]
        min_x, min_y = max(0, int(min(xs))), max(0, int(min(ys)))
        max_x, max_y = min(width, int(max(xs))), min(height, int(max(ys)))
        
        # Ensure width and height > 0
        crop_width = max(1, max_x - min_x)
        crop_height = max(1, max_y - min_y)
        
        cropped = image.crop((min_x, min_y, min_x + crop_width, min_y + crop_height))
        return cropped
    except Exception as e:
        st.error(f"Error in image cropping: {str(e)}")
        return image

def extract_color_features(image):
    """Extract color histogram features (30% weight)"""
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Extract color histograms for each channel
        hist_r = cv2.calcHist([img_array], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([img_array], [1], None, [32], [0, 256])
        hist_b = cv2.calcHist([img_array], [2], None, [32], [0, 256])
        
        # Normalize histograms
        hist_r = hist_r.flatten() / np.sum(hist_r)
        hist_g = hist_g.flatten() / np.sum(hist_g)
        hist_b = hist_b.flatten() / np.sum(hist_b)
        
        # Combine RGB histograms
        color_features = np.concatenate([hist_r, hist_g, hist_b])
        
        # Add dominant color features
        pixels = img_array.reshape(-1, 3)
        dominant_colors = np.mean(pixels, axis=0) / 255.0  # Normalize to [0,1]
        
        # Combine all color features
        combined_color = np.concatenate([color_features, dominant_colors])
        
        return combined_color.astype(np.float32)
    except Exception as e:
        st.error(f"Error in color feature extraction: {str(e)}")
        return np.zeros(99, dtype=np.float32)  # 32*3 + 3 = 99 features

def extract_pattern_features(image, dino_model, processor, device):
    """Extract pattern/texture features using DINO model (70% weight)"""
    try:
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = dino_model(**inputs)
        
        # Get DINO features
        dino_features = outputs.last_hidden_state[:, 0]
        dino_vector = dino_features.detach().cpu().numpy().flatten()
        
        # Add texture analysis using Local Binary Pattern-like features
        img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Resize for consistent processing
        img_resized = cv2.resize(img_gray, (224, 224))
        
        # Extract edge features using Sobel operators
        sobel_x = cv2.Sobel(img_resized, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_resized, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Create edge histogram
        edge_hist = np.histogram(edge_magnitude.flatten(), bins=32, range=(0, 255))[0]
        edge_hist = edge_hist.astype(np.float32) / np.sum(edge_hist)
        
        # Extract gradient direction features
        gradient_direction = np.arctan2(sobel_y, sobel_x)
        direction_hist = np.histogram(gradient_direction.flatten(), bins=16, range=(-np.pi, np.pi))[0]
        direction_hist = direction_hist.astype(np.float32) / np.sum(direction_hist)
        
        # Combine pattern features
        pattern_features = np.concatenate([
            dino_vector.astype(np.float32),
            edge_hist,
            direction_hist
        ])
        
        return pattern_features
    except Exception as e:
        st.error(f"Error in pattern feature extraction: {str(e)}")
        return np.zeros(432, dtype=np.float32)  # 384 (DINO) + 32 (edge) + 16 (direction)

def extract_basic_dino_features(image, dino_model, processor, device):
    """Extract basic DINO features (384 dimensions) for compatibility with existing index"""
    try:
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = dino_model(**inputs)
        
        features = outputs.last_hidden_state[:, 0]
        vector = features.detach().cpu().numpy()
        vector = np.float32(vector)
        
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        
        faiss.normalize_L2(vector)
        return vector
    except Exception as e:
        st.error(f"Error in basic DINO feature extraction: {str(e)}")
        return None

def extract_features(image, dino_model, processor, device):
    """Extract features compatible with existing FAISS index"""
    try:
        # Check existing index dimensions
        index = faiss.read_index("index/vector.index")
        expected_dim = index.d
        
        if expected_dim == 384:
            # Use basic DINO features for compatibility with existing index
            st.info("Using basic DINO features (384D) for compatibility with existing index")
            return extract_basic_dino_features(image, dino_model, processor, device)
        else:
            # Use enhanced features if index supports it
            st.info("Using enhanced features (70% pattern + 30% color)")
            return extract_enhanced_features(image, dino_model, processor, device)
            
    except Exception as e:
        st.warning(f"Could not determine index dimensions, using basic DINO features: {str(e)}")
        return extract_basic_dino_features(image, dino_model, processor, device)

def extract_enhanced_features(image, dino_model, processor, device):
    """Extract combined features: 70% pattern + 30% color"""
    try:
        # Extract pattern features (70% weight)
        pattern_features = extract_pattern_features(image, dino_model, processor, device)
        
        # Extract color features (30% weight)
        color_features = extract_color_features(image)
        
        # Apply weights
        pattern_weight = 0.7
        color_weight = 0.3
        
        # Normalize features individually
        pattern_features = normalize([pattern_features], norm='l2')[0]
        color_features = normalize([color_features], norm='l2')[0]
        
        # Apply weights
        weighted_pattern = pattern_features * pattern_weight
        weighted_color = color_features * color_weight
        
        # Combine features
        combined_features = np.concatenate([weighted_pattern, weighted_color])
        
        # Final normalization
        combined_features = normalize([combined_features], norm='l2')[0]
        
        if combined_features.ndim == 1:
            combined_features = combined_features.reshape(1, -1)
        
        # Convert to float32 for FAISS
        combined_features = combined_features.astype(np.float32)
        
        return combined_features
    except Exception as e:
        st.error(f"Error in enhanced feature extraction: {str(e)}")
        # Fallback to basic DINO features
        return extract_basic_dino_features(image, dino_model, processor, device)

def search_similar_patterns(query_vector, index, image_paths, top_k=15):
    """Search for similar patterns in the index"""
    try:
        distances, indices = index.search(query_vector, top_k)
        
        result_paths = []
        scores = []
        
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(image_paths):
                result_paths.append(image_paths[idx])
                scores.append(score)
        
        return result_paths, scores
    except Exception as e:
        st.error(f"Error in similarity search: {str(e)}")
        return [], []

def create_results_visualization(query_image, result_paths, scores):
    """Create a visualization of the results"""
    try:
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()
        
        # Display query image
        axes[0].imshow(query_image)
        axes[0].set_title('Your Uploaded Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Display top 15 results
        for i in range(min(15, len(result_paths))):
            try:
                img = Image.open(result_paths[i])
                axes[i+1].imshow(img)
                axes[i+1].set_title(f'Match {i+1}\nScore: {scores[i]:.3f}', fontsize=10)
                axes[i+1].axis('off')
            except Exception as e:
                axes[i+1].text(0.5, 0.5, f'Error loading\nimage {i+1}', 
                             ha='center', va='center', transform=axes[i+1].transAxes)
                axes[i+1].axis('off')
        
        # Hide unused subplots
        for i in range(16, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">👕 Jersey Pattern Matcher</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to find the top 15 matching jersey patterns from our catalogue</p>', unsafe_allow_html=True)
    
    # Load models and index
    with st.spinner("Loading AI models..."):
        yolo_model, dino_model, processor, device = load_models()
        index, image_paths = load_index()
    
    if yolo_model is None or dino_model is None or index is None:
        st.error("Failed to load required models or index. Please check your setup.")
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ How it works")
        st.markdown("""
        1. **Upload** your jersey image
        2. **YOLO** detects and crops the jersey area
        3. **Enhanced feature extraction**:
           - 70% Pattern features (DINO + texture analysis)
           - 30% Color features (histograms + dominant colors)
        4. **Similarity search** finds matching patterns
        5. **Results** show top 15 matches with scores
        """)
        
        st.header("🎨 Feature Analysis")
        try:
            index = faiss.read_index("index/vector.index")
            if index.d == 384:
                st.markdown("""
                **Current Mode: Basic DINO Features**
                - Using 384D DINO visual transformer features
                - Compatible with existing catalogue index
                
                *To use enhanced features (70% pattern + 30% color), rebuild the catalogue index with the new feature extraction.*
                """)
            else:
                st.markdown("""
                **Enhanced Features (70% Pattern + 30% Color)**:
                
                **Pattern Features (70%)**:
                - DINO visual transformer features
                - Edge detection (Sobel operators) 
                - Gradient direction analysis
                
                **Color Features (30%)**:
                - RGB color histograms
                - Dominant color extraction
                """)
        except:
            st.markdown("**Basic DINO Features**: 384D visual transformer features")
        
        st.header("📊 Model Info")
        st.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        st.info(f"Catalogue size: {len(image_paths)} images")
    
    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📤 Upload Your Jersey Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a jersey for pattern matching"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width='stretch')
        
        with col2:
            st.subheader("Processing Pipeline")
            
            # Process button
            if st.button("🔍 Find Matching Patterns", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: YOLO detection
                    status_text.text("Step 1/4: Detecting jersey area with YOLO...")
                    progress_bar.progress(25)
                    polygons = extract_yolo_coordinates(image, yolo_model)
                    
                    if polygons:
                        st.success(f"✅ Detected {len(polygons)} jersey region(s)")
                    else:
                        st.warning("⚠️ No jersey regions detected, using full image")
                    
                    # Step 2: Crop image
                    status_text.text("Step 2/4: Cropping detected region...")
                    progress_bar.progress(50)
                    processed_image = crop_image_with_polygon(image, polygons)
                    
                    # Step 3: Feature extraction
                    status_text.text("Step 3/4: Extracting pattern features...")
                    progress_bar.progress(75)
                    query_vector = extract_features(processed_image, dino_model, processor, device)
                    
                    if query_vector is None:
                        st.error("Failed to extract features")
                        return
                    
                    # Step 4: Similarity search
                    status_text.text("Step 4/4: Searching for similar patterns...")
                    progress_bar.progress(100)
                    result_paths, scores = search_similar_patterns(query_vector, index, image_paths)
                    
                    status_text.text("✅ Processing complete!")
                    
                    # Display results
                    if result_paths:
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('<div class="results-section">', unsafe_allow_html=True)
                        st.subheader("🎯 Top 15 Matching Patterns")
                        
                        # Create and display visualization
                        with st.spinner("Creating results visualization..."):
                            viz_buffer = create_results_visualization(processed_image, result_paths, scores)
                        
                        if viz_buffer:
                            st.image(viz_buffer, caption="Pattern Matching Results", width='stretch')
                            
                            # Download button for results
                            st.download_button(
                                label="📥 Download Results",
                                data=viz_buffer.getvalue(),
                                file_name="pattern_matching_results.png",
                                mime="image/png"
                            )
                        
                        # Display individual results in expandable sections
                        st.subheader("📋 Detailed Results")
                        for i, (path, score) in enumerate(zip(result_paths[:15], scores[:15])):
                            with st.expander(f"Match {i+1} - Score: {score:.3f}"):
                                try:
                                    result_img = Image.open(path)
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        st.image(result_img, width='stretch')
                                    with col2:
                                        st.write(f"**File:** {os.path.basename(path)}")
                                        st.write(f"**Similarity Score:** {score:.4f}")
                                        st.write(f"**Path:** {path}")
                                except Exception as e:
                                    st.error(f"Error loading image: {str(e)}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("No matching patterns found")
                        
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
