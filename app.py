import streamlit as st
import cv2
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# Define the directory and JSON file
AUTHORIZED_DIR = "authorized_images"
JSON_FILE = "authorized_images.json"

class ImprovedPersonDetector:
    def __init__(self):
        """Initialize the detector with authorized images"""
        self.authorized_faces = []
        self.authorized_image_paths = []
        
        # Create directories if they don't exist
        os.makedirs(AUTHORIZED_DIR, exist_ok=True)
        
        # Initialize JSON file if it doesn't exist
        if not os.path.exists(JSON_FILE):
            with open(JSON_FILE, 'w') as f:
                json.dump([], f)
        
        # Load existing data
        try:
            with open(JSON_FILE, 'r') as f:
                self.authorized_image_paths = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self.authorized_image_paths = []
            with open(JSON_FILE, 'w') as f:
                json.dump([], f)
        
        self.mtcnn = MTCNN(keep_all=True, device='cpu', margin=20, min_face_size=80)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        
    def load_images(self):
        """Load authorized images with error handling"""
        self.authorized_faces = []
        loaded_count = 0
        
        for idx, image_path in enumerate(self.authorized_image_paths):
            try:
                if not os.path.exists(image_path):
                    st.sidebar.warning(f"File not found: {os.path.basename(image_path)}")
                    continue
                    
                person_image = Image.open(image_path).convert('RGB')
                faces = self.mtcnn(person_image)
                
                if faces is not None and len(faces) > 0:
                    face_tensor = faces[0]
                    face_embedding = self.resnet(face_tensor.unsqueeze(0)).detach().numpy()
                    
                    self.authorized_faces.append({
                        'id': idx + 1,
                        'image': person_image,
                        'embedding': face_embedding,
                        'name': f"Person_{idx + 1}",
                        'path': image_path
                    })
                    loaded_count += 1
                    st.sidebar.success(f"✅ Loaded: Person_{idx + 1}")
                else:
                    st.sidebar.error(f"❌ No face found in: {os.path.basename(image_path)}")
                    
            except Exception as e:
                st.sidebar.error(f"Error loading {os.path.basename(image_path)}: {str(e)}")
        
        st.sidebar.info(f"Successfully loaded {loaded_count}/{len(self.authorized_image_paths)} authorized persons")
        
    def remove_images(self):
        """Remove all authorized images with proper cleanup"""
        try:
            # Remove files from disk
            removed_count = 0
            for file_path in self.authorized_image_paths:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        removed_count += 1
                except Exception as e:
                    st.sidebar.error(f"Failed to remove {os.path.basename(file_path)}: {str(e)}")
            
            # Reset in-memory data
            self.authorized_faces = []
            self.authorized_image_paths = []
            
            # Reset JSON file
            with open(JSON_FILE, 'w') as f:
                json.dump([], f)
            
            st.sidebar.success(f"Removed {removed_count} authorized images")
            return True
            
        except Exception as e:
            st.sidebar.error(f"Error during removal: {str(e)}")
            return False
    
    def match_face(self, face_embedding, threshold=0.75):
        """Match a detected face with authorized faces"""
        if not self.authorized_faces:
            return None, 0
        
        best_match = None
        best_score = 0
        
        for auth_face in self.authorized_faces:
            try:
                # Calculate cosine similarity
                cosine_sim = np.dot(face_embedding, auth_face['embedding'].T) / (
                    np.linalg.norm(face_embedding) * np.linalg.norm(auth_face['embedding'])
                )
                score = (cosine_sim + 1) / 2  # Convert to [0,1] range
                
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = auth_face
            except Exception as e:
                continue
        
        if best_match:
            confidence = min(100, max(0, best_score * 100))  # Ensure 0-100 range
            return best_match, confidence
        return None, 0

def save_uploaded_file(uploaded_file):
    """Save uploaded file with proper error handling"""
    try:
        os.makedirs(AUTHORIZED_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        file_path = os.path.join(AUTHORIZED_DIR, filename)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Update JSON file
        if os.path.exists(JSON_FILE):
            with open(JSON_FILE, 'r') as f:
                try:
                    image_paths = json.load(f)
                except json.JSONDecodeError:
                    image_paths = []
        else:
            image_paths = []
        
        image_paths.append(os.path.abspath(file_path))
        
        with open(JSON_FILE, 'w') as f:
            json.dump(image_paths, f)
        
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def run_detection(detector, threshold, webcam_placeholder, log_placeholder):
    """Run detection with proper resource management"""
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            webcam_placeholder.error("❌ Could not open webcam")
            st.session_state.detection_active = False
            return
        
        while st.session_state.detection_active:
            ret, frame = cap.read()
            if not ret:
                st.warning("Could not read frame from camera")
                break
            
            try:
                frame, detected_persons = detector.process_frame(frame, threshold)
                
                # Display status
                status = "✅ DETECTED" if detected_persons else "Scanning..."
                color = (0, 255, 0) if detected_persons else (255, 255, 255)
                cv2.putText(frame, status, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Update detection log
                if detected_persons:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    for person in detected_persons:
                        log_entry = {
                            "Timestamp": current_time,
                            "Person": person["name"],
                            "Confidence": f"{person['confidence']:.1f}%"
                        }
                        if log_entry not in st.session_state.detection_log:
                            st.session_state.detection_log.insert(0, log_entry)
                
                # Display frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                webcam_placeholder.image(rgb_frame, channels="RGB")
                
                # Update log display
                if st.session_state.detection_log:
                    log_placeholder.dataframe(
                        st.session_state.detection_log[:20],  # Limit to 20 entries
                        height=400,
                        column_config={
                            "Timestamp": "Time",
                            "Person": "Person",
                            "Confidence": st.column_config.ProgressColumn(
                                "Confidence",
                                format="%.1f%%",
                                min_value=0,
                                max_value=100
                            )
                        }
                    )
                
                time.sleep(0.05)
                
            except Exception as e:
                st.error(f"Detection error: {str(e)}")
                break
                
    finally:
        if cap is not None:
            cap.release()

def main():
    st.set_page_config(
        page_title="Authorized Person Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = ImprovedPersonDetector()
        st.session_state.detection_active = False
        st.session_state.detection_log = []
    
    # Debugging info
    with st.expander("Debug Info", expanded=False):
        st.write("Current directory:", os.getcwd())
        st.write("Directory contents:", os.listdir())
        if os.path.exists(AUTHORIZED_DIR):
            st.write(f"{AUTHORIZED_DIR} contents:", os.listdir(AUTHORIZED_DIR))
        else:
            st.warning(f"{AUTHORIZED_DIR} directory not found!")
        
        if os.path.exists(JSON_FILE):
            with open(JSON_FILE, 'r') as f:
                try:
                    contents = json.load(f)
                    st.write("JSON file contents:", contents)
                except json.JSONDecodeError:
                    st.error("Invalid JSON file contents")
        else:
            st.warning("JSON file not found")
    
    # Main UI
    st.title("👤 Authorized Person Detection System")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        webcam_placeholder = st.empty()
        st.markdown("### Camera Controls")
        
        if st.button("Start/Stop Detection", key="detection_toggle"):
            st.session_state.detection_active = not st.session_state.detection_active
            if st.session_state.detection_active:
                st.session_state.detection_log = []
                run_detection(
                    st.session_state.detector,
                    threshold=0.75,
                    webcam_placeholder=webcam_placeholder,
                    log_placeholder=col2.empty()
                )
        
        status_color = "green" if st.session_state.detection_active else "red"
        st.markdown(f"**Status:** <span style='color:{status_color}'>"
                   f"{'ACTIVE' if st.session_state.detection_active else 'INACTIVE'}</span>",
                   unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Detection Log")
        log_placeholder = st.empty()
        if st.session_state.detection_log:
            log_placeholder.dataframe(st.session_state.detection_log)
        else:
            log_placeholder.info("No detections yet")
        
        if st.button("Clear Log"):
            st.session_state.detection_log = []
            log_placeholder.info("Log cleared")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Authorized Persons")
        if st.button("Load Authorized Images"):
            st.session_state.detector.load_images()
        
        if st.button("Remove All Authorized Images"):
            if st.session_state.detector.remove_images():
                st.success("All authorized images removed")
            else:
                st.error("Failed to remove images")
        
        st.subheader("Add New Authorized Persons")
        uploaded_files = st.file_uploader(
            "Upload images of authorized persons",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Save Uploaded Images"):
            for uploaded_file in uploaded_files:
                saved_path = save_uploaded_file(uploaded_file)
                if saved_path:
                    st.success(f"Saved: {os.path.basename(saved_path)}")
                    st.session_state.detector.authorized_image_paths.append(saved_path)
            
            # Reload images after adding new ones
            st.session_state.detector.load_images()
        
        # Display current authorized persons
        if st.session_state.detector.authorized_faces:
            st.subheader("Current Authorized Persons")
            cols = st.columns(2)
            for idx, face in enumerate(st.session_state.detector.authorized_faces):
                with cols[idx % 2]:
                    st.image(face['image'], caption=face['name'], width=150)
                    st.caption(f"ID: {face['id']}")

if __name__ == "__main__":
    main()
