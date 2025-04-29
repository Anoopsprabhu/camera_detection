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

# Server configuration for camera
DEFAULT_CAMERA_SOURCE = 0  # Default webcam
SERVER_RTSP_URL = None     # Can be set to a specific RTSP stream if needed

class ImprovedPersonDetector:
    def __init__(self):
        """Initialize the detector with authorized images"""
        self.authorized_faces = []
        self.authorized_image_paths = []
        
        # Ensure directory exists with proper permissions
        if not os.path.exists(AUTHORIZED_DIR):
            os.makedirs(AUTHORIZED_DIR, exist_ok=True)
        
        # Load existing image paths
        if os.path.exists(JSON_FILE):
            try:
                with open(JSON_FILE, 'r') as f:
                    self.authorized_image_paths = json.load(f)
                    # Normalize all paths to use proper OS-specific separators
                    self.authorized_image_paths = [os.path.normpath(p) for p in self.authorized_image_paths]
            except json.JSONDecodeError:
                st.sidebar.error("❌ Error reading JSON file. Creating a new one.")
                with open(JSON_FILE, 'w') as f:
                    json.dump([], f)
                self.authorized_image_paths = []
        
        # Initialize face detection and recognition models
        try:
            self.mtcnn = MTCNN(keep_all=True, device='cpu', margin=20, min_face_size=80)
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        except Exception as e:
            st.error(f"❌ Error initializing face detection models: {str(e)}")
        
    def load_images(self):
        """Load authorized images"""
        self.authorized_faces = []
        valid_paths = []
        
        for idx, image_path in enumerate(self.authorized_image_paths):
            # Normalize path for the current OS
            image_path = os.path.normpath(image_path)
            
            # Check if file exists
            if not os.path.exists(image_path):
                st.sidebar.warning(f"❌ Image not found: {image_path}")
                continue
                
            try:
                person_image = Image.open(image_path).convert('RGB')
                faces = self.mtcnn(person_image)
                
                if faces is not None and len(faces) > 0:
                    face_tensor = faces[0]
                    face_embedding = self.resnet(face_tensor.unsqueeze(0)).detach().numpy()
                    
                    self.authorized_faces.append({
                        'id': idx + 1,
                        'image': person_image,
                        'embedding': face_embedding,
                        'name': f"Person_{idx + 1}"
                    })
                    
                    st.sidebar.success(f"✅ Loaded authorized person: Person_{idx + 1}")
                    valid_paths.append(image_path)
                else:
                    st.sidebar.warning(f"⚠️ No face found in authorized image {idx + 1}")
            except Exception as e:
                st.sidebar.error(f"❌ Error processing image {image_path}: {str(e)}")
        
        # Update paths to include only valid ones
        self.authorized_image_paths = valid_paths
        # Update JSON with only valid paths
        with open(JSON_FILE, 'w') as f:
            json.dump(valid_paths, f)
            
        st.sidebar.info(f"Loaded {len(self.authorized_faces)} authorized persons")
    
    def remove_images(self):
        """Remove all authorized images and delete all files listed in JSON"""
        if os.path.exists(JSON_FILE):
            try:
                with open(JSON_FILE, 'r') as f:
                    image_paths = json.load(f)
                
                for file_path in image_paths:
                    # Normalize path for current OS
                    file_path = os.path.normpath(file_path)
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            st.sidebar.success(f"🗑️ Deleted file: {os.path.basename(file_path)}")
                        except Exception as e:
                            st.sidebar.error(f"❌ Failed to delete {os.path.basename(file_path)}: {str(e)}")
                    else:
                        st.sidebar.warning(f"File not found: {os.path.basename(file_path)}")
                
                with open(JSON_FILE, 'w') as f:
                    json.dump([], f)
            except Exception as e:
                st.sidebar.error(f"❌ Error cleaning up authorized images: {str(e)}")
        
        self.authorized_faces = []
        self.authorized_image_paths = []
        st.sidebar.success("🗑️ All authorized images removed from memory, directory, and JSON. Detection continues.")
    
    def match_face(self, face_embedding, threshold=0.75):
        """Match a detected face with authorized faces using stricter criteria"""
        if not self.authorized_faces:
            return None, 0
        
        best_match = None
        best_cosine_score = 0
        best_euclidean_dist = float('inf')
        
        for auth_face in self.authorized_faces:
            cosine_similarity = np.dot(face_embedding, auth_face['embedding'].T) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(auth_face['embedding'])
            )
            cosine_score = (cosine_similarity + 1) / 2
            
            euclidean_dist = np.linalg.norm(face_embedding - auth_face['embedding'])
            
            if cosine_score > best_cosine_score and euclidean_dist < 1.1:
                best_cosine_score = cosine_score
                best_euclidean_dist = euclidean_dist
                best_match = auth_face
        
        if best_cosine_score >= threshold:
            confidence = (best_cosine_score * 100).item() if isinstance(best_cosine_score, np.ndarray) else best_cosine_score * 100
            return best_match, confidence
        return None, 0
        
    def process_frame(self, frame, threshold=0.75):
        """Process frame to detect authorized persons"""
        try:
            if frame is None:
                return np.zeros((480, 640, 3), dtype=np.uint8), []
                
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            faces = self.mtcnn(pil_image)
            
            detected_persons = []
            
            if faces is not None:
                boxes = self.mtcnn.detect(pil_image)[0]
                if boxes is not None:
                    for i, face_tensor in enumerate(faces):
                        if i >= len(boxes):
                            continue
                        
                        face_embedding = self.resnet(face_tensor.unsqueeze(0)).detach().numpy()
                        match, confidence = self.match_face(face_embedding, threshold)
                        
                        if match:
                            box = boxes[i]
                            left, top, right, bottom = map(int, box)
                            person_name = match['name']
                            detected_persons.append({"name": person_name, "confidence": confidence})
                            
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            info_text = f"{person_name} ({confidence:.1f}%)"
                            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, info_text, (left + 6, bottom - 6),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            # Don't interrupt the camera feed on error, just log it
            st.error(f"Error in frame processing: {str(e)}")
            return np.zeros((480, 640, 3), dtype=np.uint8), []
        
        return frame, detected_persons

def save_uploaded_file(uploaded_file):
    """Save uploaded file to the authorized_images directory and update JSON"""
    # Ensure the directory exists
    if not os.path.exists(AUTHORIZED_DIR):
        os.makedirs(AUTHORIZED_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Remove any problematic characters from filename
    safe_filename = ''.join(c for c in uploaded_file.name if c.isalnum() or c in '._-')
    filename = f"{timestamp}_{safe_filename}"
    file_path = os.path.join(AUTHORIZED_DIR, filename)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, 'r') as f:
                image_paths = json.load(f)
        except json.JSONDecodeError:
            st.warning("❌ Error reading JSON file. Creating a new one.")
            image_paths = []
    else:
        image_paths = []
    
    # Use OS-specific normalized path
    file_path = os.path.normpath(file_path)
    image_paths.append(file_path)
    
    with open(JSON_FILE, 'w') as f:
        json.dump(image_paths, f)
    
    return file_path

def open_camera_source(source_id):
    """
    Try to open the camera with the given source identifier
    Returns the VideoCapture object or None if failed
    """
    # First try to interpret as a number
    try:
        if isinstance(source_id, str) and source_id.isdigit():
            source_id = int(source_id)
    except ValueError:
        pass  # Keep as string if not a valid integer
    
    # Try to open the camera
    cap = cv2.VideoCapture(source_id)
    
    # Check if opened successfully
    if cap.isOpened():
        return cap
    else:
        return None

def find_available_camera():
    """
    Find an available camera by trying different sources
    Returns the first working camera or None if none found
    """
    # Check common camera indices
    for source_id in [0, 1, 2, 3, 4]:
        cap = open_camera_source(source_id)
        if cap is not None:
            return cap
    
    # Try common device paths on Linux
    for device in ['/dev/video0', '/dev/video1', '/dev/video2']:
        if os.path.exists(device):
            cap = open_camera_source(device)
            if cap is not None:
                return cap
    
    # If SERVER_RTSP_URL is defined, try that
    if SERVER_RTSP_URL:
        cap = open_camera_source(SERVER_RTSP_URL)
        if cap is not None:
            return cap
    
    # No camera found
    return None

def run_detection(detector, threshold, webcam_placeholder, log_placeholder, camera_source):
    """Run the detection loop independently"""
    # Try to open the specified camera source
    cap = open_camera_source(camera_source)
    
    # If that fails, try to find any available camera
    if cap is None or not cap.isOpened():
        webcam_placeholder.warning("⚠️ Couldn't open specified camera, trying to find an available one...")
        cap = find_available_camera()
    
    # If still no camera, show error
    if cap is None or not cap.isOpened():
        webcam_placeholder.error("❌ Could not open any webcam")
        st.session_state.detection_active = False
        return
    
    try:
        frame_count = 0
        connection_retry_count = 0
        blank_frame_count = 0
        
        while st.session_state.detection_active:
            ret, frame = cap.read()
            
            # Handle connection issues or blank frames
            if not ret or frame is None or frame.size == 0:
                blank_frame_count += 1
                connection_retry_count += 1
                
                # After several failed attempts, show a message
                if blank_frame_count > 10:
                    webcam_placeholder.warning("⚠️ Camera connection issue. Attempting to reconnect...")
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, "Camera disconnected...", (150, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    webcam_placeholder.image(blank_frame, channels="BGR", use_column_width=True)
                
                # Try to reconnect if we've had several consecutive failures
                if connection_retry_count > 30:  # About 1.5 seconds
                    webcam_placeholder.info("🔄 Reconnecting to camera...")
                    cap.release()
                    time.sleep(1)
                    cap = open_camera_source(camera_source)
                    if cap is None or not cap.isOpened():
                        cap = find_available_camera()
                    
                    connection_retry_count = 0
                    # If still can't connect, abort
                    if cap is None or not cap.isOpened():
                        webcam_placeholder.error("❌ Failed to reconnect to any camera. Stopping detection.")
                        st.session_state.detection_active = False
                        break
                
                time.sleep(0.05)
                continue
            
            # Reset counters on successful frame read
            blank_frame_count = 0
            connection_retry_count = 0
            
            # Process frames at a reduced rate to lower CPU usage
            if frame_count % 3 == 0:  # Process every 3rd frame
                frame, detected_persons = detector.process_frame(frame, threshold)
                
                status = "✅ DETECTED" if detected_persons else "Scanning..."
                status_color = (0, 255, 0) if detected_persons else (255, 255, 255)
                cv2.putText(frame, status, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
                if detected_persons:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    for person in detected_persons:
                        st.session_state.detection_log.insert(0, {
                            "Timestamp": current_time,
                            "Person": person["name"],
                            "Confidence": f"{person['confidence']:.1f}%"
                        })
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                webcam_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                
                if st.session_state.detection_log:
                    # Limit log to prevent memory issues
                    if len(st.session_state.detection_log) > 100:
                        st.session_state.detection_log = st.session_state.detection_log[:100]
                    log_placeholder.dataframe(st.session_state.detection_log, height=400)
                else:
                    log_placeholder.info("No authorized persons detected yet")
            
            frame_count += 1
            time.sleep(0.05)  # Short sleep to prevent high CPU usage
    
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
    finally:
        if cap is not None:
            cap.release()

def main():
    st.set_page_config(page_title="Authorized Person Detection", layout="wide")
    
    st.title("Authorized Person Detection System")
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = ImprovedPersonDetector()
        st.session_state.detection_active = False
        st.session_state.detection_log = []
    
    col1, col2 = st.columns([2, 1])
    webcam_placeholder = col1.empty()
    log_placeholder = col2.empty()
    
    # Camera configuration in sidebar
    st.sidebar.header("Configuration")
    
    # Camera source selection
    st.sidebar.subheader("Camera Settings")
    camera_options = {
        "Default Webcam (0)": 0,
        "Secondary Camera (1)": 1,
        "Video Device (/dev/video0)": "/dev/video0"
    }
    
    # Add RTSP option if defined
    if SERVER_RTSP_URL:
        camera_options["Server Camera (RTSP)"] = SERVER_RTSP_URL
    
    # Add custom option
    camera_source = st.sidebar.selectbox(
        "Select Camera Source", 
        list(camera_options.keys())
    )
    
    # Custom camera URL/index
    custom_source = st.sidebar.text_input(
        "Or enter custom camera source (device path, URL, or index)",
        ""
    )
    
    # Determine the actual camera source to use
    final_camera_source = custom_source if custom_source else camera_options[camera_source]
    
    st.sidebar.subheader("Upload Authorized Persons")
    authorized_uploads = st.sidebar.file_uploader(
        "Upload images of authorized persons", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="auth_uploads"
    )
    
    if st.sidebar.button("Load Authorized Images"):
        # Save and load newly uploaded images
        for uploaded_file in authorized_uploads:
            file_path = save_uploaded_file(uploaded_file)
            if file_path not in st.session_state.detector.authorized_image_paths:
                st.session_state.detector.authorized_image_paths.append(file_path)
        # Load all images (including previously saved ones)
        st.session_state.detector.load_images()
    
    if st.sidebar.button("Remove Authorized Images"):
        st.session_state.detector.remove_images()
    
    # Display thumbnails of authorized persons if any exist
    if st.session_state.detector.authorized_image_paths:
        st.sidebar.subheader("Authorized Persons")
        # Calculate how many columns we need
        num_images = len(st.session_state.detector.authorized_image_paths)
        cols_per_row = min(3, num_images)  # Max 3 columns per row
        
        if cols_per_row > 0:  # Check if we have any images
            cols = st.sidebar.columns(cols_per_row)
            for idx, img_path in enumerate(st.session_state.detector.authorized_image_paths):
                norm_path = os.path.normpath(img_path)
                # Only try to display if file exists
                if os.path.exists(norm_path):
                    try:
                        with cols[idx % cols_per_row]:
                            img = Image.open(norm_path)
                            st.image(img, caption=f"Person {idx + 1}", width=100)
                    except Exception as e:
                        st.sidebar.error(f"Cannot display image {idx+1}: {str(e)}")
    
    with col1:
        st.header("Live Camera Feed")
        threshold = st.slider("Detection Confidence Threshold", 0.6, 0.9, 0.75, 0.05)
        
        # Test camera button
        if st.button("Test Camera Connection"):
            test_cap = open_camera_source(final_camera_source)
            if test_cap is not None and test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret and frame is not None:
                    st.success(f"✅ Successfully connected to camera source: {final_camera_source}")
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(rgb_frame, caption="Camera Test Image", use_column_width=True)
                else:
                    st.error(f"❌ Camera opened but failed to read frame from: {final_camera_source}")
                test_cap.release()
            else:
                st.error(f"❌ Failed to connect to camera source: {final_camera_source}")
                # Try fallback options
                st.info("🔍 Trying to find available cameras...")
                fallback_cap = find_available_camera()
                if fallback_cap is not None:
                    ret, frame = fallback_cap.read()
                    if ret and frame is not None:
                        st.success("✅ Found an available camera! Use this instead.")
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(rgb_frame, caption="Available Camera", use_column_width=True)
                    fallback_cap.release()
                else:
                    st.error("❌ No cameras found on this system.")
        
        # Start/Stop detection button
        if st.button("Start/Stop Detection"):
            st.session_state.detection_active = not st.session_state.detection_active
            if st.session_state.detection_active:
                if len(st.session_state.detector.authorized_faces) == 0:
                    st.warning("⚠️ No authorized persons loaded. Detection will not recognize anyone.")
                st.session_state.detection_log = []
                run_detection(st.session_state.detector, threshold, webcam_placeholder, log_placeholder, final_camera_source)
        
        status = "ACTIVE" if st.session_state.detection_active else "INACTIVE"
        if st.session_state.detection_active:
            st.success(f"Detection is {status}")
        else:
            st.error(f"Detection is {status}")
        
    with col2:
        st.header("Detection Log")
        with st.container():
            if st.session_state.detection_log:
                log_placeholder.dataframe(st.session_state.detection_log, height=400)
            else:
                log_placeholder.info("No authorized persons detected yet")
    
    # Server info section
    st.markdown("---")
    
    # System info expander for debugging
    with st.expander("System Information"):
        # Show server environment information if available
        st.write("**Camera Access Information:**")
        
        # Check if we're running on Linux
        is_linux = os.name == 'posix'
        st.write(f"- Operating System: {'Linux' if is_linux else 'Windows/Other'}")
        
        # On Linux, check available video devices
        if is_linux:
            video_devices = [d for d in os.listdir('/dev') if d.startswith('video')]
            st.write(f"- Available video devices: {', '.join(video_devices) if video_devices else 'None found'}")
            
            # Try to get webcam permissions
            st.write("- To grant webcam permissions on Linux server:")
            st.code("sudo chmod 777 /dev/video*")
        
        # Display currently selected camera source
        st.write(f"- Current camera source: {final_camera_source}")
    
    # Add informational footer
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. Select a camera source or enter a custom one (device path, URL, or index)
    2. Test the camera connection with the "Test Camera Connection" button
    3. Upload one or more images of authorized persons using the sidebar
    4. Click "Load Authorized Images" to process the images
    5. Click "Start/Stop Detection" to begin detection
    6. Detected authorized persons will appear in the log
    """)
    
if __name__ == "__main__":
    main()
