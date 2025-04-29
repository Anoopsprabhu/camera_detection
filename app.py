import streamlit as st
import cv2
import numpy as np
import os
import json
import base64
import time
from datetime import datetime
from PIL import Image, ImageOps
import io
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# Define the directory and JSON file
AUTHORIZED_DIR = "authorized_images"
JSON_FILE = "authorized_images.json"

# RTC Configuration with free public STUN servers to facilitate connections
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", 
                 "stun:stun1.l.google.com:19302"]}
    ]}
)

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
            st.sidebar.success("✅ Face detection models loaded successfully")
        except Exception as e:
            st.sidebar.error(f"❌ Error initializing face detection models: {str(e)}")
        
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
            st.warning(f"Error in frame processing: {str(e)}")
        
        return frame, detected_persons

class WebcamFaceDetectionProcessor(VideoProcessorBase):
    def __init__(self, detector, threshold=0.75):
        self.detector = detector
        self.threshold = threshold
        self.detection_log = []
        self.last_detection_time = time.time() - 10  # Initialize with an offset
        self.detection_interval = 3  # Log detections every 3 seconds
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process the frame
        img, detected_persons = self.detector.process_frame(img, self.threshold)
        
        # Add status text
        status = "✅ DETECTED" if detected_persons else "Scanning..."
        status_color = (0, 255, 0) if detected_persons else (255, 255, 255)
        cv2.putText(img, status, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Log detections at intervals to avoid flooding
        current_time = time.time()
        if detected_persons and (current_time - self.last_detection_time >= self.detection_interval):
            self.last_detection_time = current_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for person in detected_persons:
                if "detection_log" in st.session_state:  # Check if the session state exists
                    st.session_state.detection_log.insert(0, {
                        "Timestamp": timestamp,
                        "Person": person["name"],
                        "Confidence": f"{person['confidence']:.1f}%"
                    })
                    
                    # Limit log to prevent memory issues
                    if len(st.session_state.detection_log) > 100:
                        st.session_state.detection_log = st.session_state.detection_log[:100]
                
                # Also update this processor's log
                self.detection_log.insert(0, {
                    "Timestamp": timestamp,
                    "Person": person["name"],
                    "Confidence": f"{person['confidence']:.1f}%"
                })
                
                # Limit processor log as well
                if len(self.detection_log) > 100:
                    self.detection_log = self.detection_log[:100]
        
        return img

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

def process_webcam_snapshot(img_bytes):
    """Process webcam snapshot for face detection"""
    try:
        # Convert the base64 image to numpy array
        decoded = base64.b64decode(img_bytes.split(",")[1])
        img = Image.open(io.BytesIO(decoded))
        img_array = np.array(img)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_snapshot.jpg"
        file_path = os.path.join(AUTHORIZED_DIR, filename)
        
        # Save the snapshot
        img.save(file_path)
        
        # Add to authorized paths
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
        
        return file_path, True
    except Exception as e:
        st.error(f"Error processing webcam snapshot: {str(e)}")
        return None, False

def main():
    st.set_page_config(page_title="Authorized Person Detection", layout="wide")
    
    st.title("Authorized Person Detection System")
    
    # Initialize session state at the very start
    if 'detector' not in st.session_state:
        try:
            st.session_state.detector = ImprovedPersonDetector()
            st.success("✅ Detector initialized successfully")
        except Exception as e:
            st.error(f"❌ Failed to initialize detector: {str(e)}")
            return  # Exit if detector initialization fails
    
    # Initialize detection log if it doesn't exist
    if 'detection_log' not in st.session_state:
        st.session_state.detection_log = []
    
    # Initialize webcam state
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False

    col1, col2 = st.columns([2, 1])
    
    st.sidebar.header("Configuration")
    
    # Add option for uploading files or taking webcam snapshot
    st.sidebar.subheader("Upload Authorized Persons")
    
    upload_tab, webcam_tab = st.sidebar.tabs(["Upload Image", "Take Snapshot"])
    
    with upload_tab:
        authorized_uploads = st.file_uploader(
            "Upload images of authorized persons", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="auth_uploads"
        )
        
        if st.button("Process Uploaded Images"):
            # Save and load newly uploaded images
            for uploaded_file in authorized_uploads:
                file_path = save_uploaded_file(uploaded_file)
                if file_path not in st.session_state.detector.authorized_image_paths:
                    st.session_state.detector.authorized_image_paths.append(file_path)
            # Load all images (including previously saved ones)
            st.session_state.detector.load_images()
    
    with webcam_tab:
        st.write("Take a snapshot for authorization")
        snapshot_placeholder = st.empty()
        
        # Create an HTML component with JavaScript to access the webcam for snapshots
        snapshot_html = """
        <div style="text-align:center;">
            <video id="video" width="320" height="240" autoplay style="display:block; margin:0 auto;"></video>
            <button id="snap" style="margin-top:10px; padding:8px 16px;">Take Snapshot</button>
            <canvas id="canvas" width="320" height="240" style="display:none;"></canvas>
        </div>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const snap = document.getElementById('snap');
            const constraints = { video: true };
            
            // Access webcam
            async function init() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia(constraints);
                    video.srcObject = stream;
                } catch(e) {
                    console.error('Error accessing camera:', e);
                    document.body.innerHTML += '<p style="color:red;">Error accessing camera. Please check permissions.</p>';
                }
            }
            
            // Take snapshot and send to Streamlit
            snap.addEventListener('click', function() {
                const context = canvas.getContext('2d');
                context.drawImage(video, 0, 0, 320, 240);
                const dataURL = canvas.toDataURL('image/jpeg');
                window.parent.postMessage({
                    type: 'streamlit:webcamSnapshot',
                    data: dataURL
                }, '*');
            });
            
            init();
        </script>
        """
        
        # Render the HTML/JavaScript component
        from streamlit.components.v1 import html
        html(snapshot_html, height=300)
        
        # JavaScript callback handler
        def webcam_snapshot_receiver():
            html_code = """
            <script>
            window.addEventListener('message', function(event) {
                if (event.data.type === 'streamlit:webcamSnapshot') {
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: event.data.data
                    }, '*');
                }
            });
            </script>
            """
            return html(html_code, height=0)
        
        snapshot_value = webcam_snapshot_receiver()
        
        if snapshot_value:
            # Process the snapshot
            file_path, success = process_webcam_snapshot(snapshot_value)
            if success:
                st.success(f"Snapshot saved as {os.path.basename(file_path)}")
                if file_path not in st.session_state.detector.authorized_image_paths:
                    st.session_state.detector.authorized_image_paths.append(file_path)
                    st.session_state.detector.load_images()
    
    if st.sidebar.button("Remove All Authorized Images"):
        st.session_state.detector.remove_images()
    
    # Display thumbnails of authorized persons if any exist
    if st.session_state.detector.authorized_image_paths:
        st.sidebar.subheader("Authorized Persons")
        num_images = len(st.session_state.detector.authorized_image_paths)
        cols_per_row = min(3, num_images)  # Max 3 columns per row
        
        if cols_per_row > 0:
            cols = st.sidebar.columns(cols_per_row)
            for idx, img_path in enumerate(st.session_state.detector.authorized_image_paths):
                norm_path = os.path.normpath(img_path)
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
        
        # Add toggle button for webcam
        if st.button("Start Detection" if not st.session_state.webcam_active else "Stop Detection"):
            st.session_state.webcam_active = not st.session_state.webcam_active
            st.experimental_rerun()
        
        # Create WebRTC component only when webcam is active
        if st.session_state.webcam_active:
            try:
                webrtc_ctx = webrtc_streamer(
                    key="face-detection",
                    video_processor_factory=lambda: WebcamFaceDetectionProcessor(
                        st.session_state.detector, threshold
                    ),
                    rtc_configuration=RTC_CONFIGURATION,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                    mode=WebRtcMode.SENDRECV,
                )
                
                if webrtc_ctx.video_processor:
                    if len(st.session_state.detector.authorized_faces) == 0:
                        st.warning("⚠️ No authorized persons loaded. Detection will not recognize anyone.")
                    st.success("✅ Webcam is active")
                else:
                    st.error("❌ Webcam is not active. Please check camera permissions.")
            except Exception as e:
                st.error(f"❌ Error initializing webcam: {str(e)}")
                st.info("Please ensure your browser has camera access and try refreshing the page.")
        else:
            st.info("Webcam is currently off. Click 'Start Detection' to begin.")
    
    with col2:
        st.header("Detection Log")
        log_placeholder = st.empty()
        
        # Update the detection log from the session state
        if st.session_state.detection_log:
            log_placeholder.dataframe(st.session_state.detection_log, height=400, use_container_width=True)
        else:
            log_placeholder.info("No authorized persons detected yet")
    
    # Add informational footer
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. Upload images of authorized persons using the sidebar or take a snapshot
    2. Start webcam detection using the "START" button next to the video feed
    3. Adjust the confidence threshold if needed
    4. Detected authorized persons will appear in the log
    
    ### Requirements:
    - You must allow camera access in your browser
    - For best results, ensure good lighting and face the camera directly
    """)
    
    # Add refresh button for the detection log
    if st.button("Refresh Detection Log"):
        st.session_state.detection_log = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()
