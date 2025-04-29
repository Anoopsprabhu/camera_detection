import streamlit as st
import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av

# Configuration
AUTHORIZED_DIR = "authorized_images"
JSON_FILE = "authorized_images.json"

# Improved RTC Configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class FaceDetector:
    def __init__(self):
        self.authorized_faces = []
        self.authorized_image_paths = []
        
        # Setup directory
        os.makedirs(AUTHORIZED_DIR, exist_ok=True)
        
        # Load existing images
        if os.path.exists(JSON_FILE):
            try:
                with open(JSON_FILE, 'r') as f:
                    self.authorized_image_paths = json.load(f)
            except json.JSONDecodeError:
                with open(JSON_FILE, 'w') as f:
                    json.dump([], f)
        
        # Initialize models with error handling
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.mtcnn = MTCNN(
                keep_all=True,
                device=self.device,
                margin=20,
                min_face_size=60
            )
            self.resnet = InceptionResnetV1(
                pretrained='vggface2'
            ).eval().to(self.device)
            st.sidebar.success("✅ Models loaded successfully")
        except Exception as e:
            st.sidebar.error(f"❌ Model loading failed: {str(e)}")
            raise

    def load_images(self):
        """Load and process authorized faces"""
        self.authorized_faces = []
        
        for img_path in self.authorized_image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                faces = self.mtcnn(img)
                
                if faces is not None:
                    embedding = self.resnet(faces[0].unsqueeze(0)).detach().cpu().numpy()
                    self.authorized_faces.append({
                        'path': img_path,
                        'embedding': embedding,
                        'name': os.path.basename(img_path).split('.')[0]
                    })
            except Exception as e:
                st.warning(f"Failed to process {img_path}: {str(e)}")
        
        st.sidebar.info(f"Loaded {len(self.authorized_faces)} authorized faces")

    def process_frame(self, frame, threshold=0.7):
        """Process each video frame for face detection"""
        detected = []
        try:
            # Convert frame to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Detect faces
            faces = self.mtcnn(pil_img)
            boxes, _ = self.mtcnn.detect(pil_img)
            
            if faces is not None and boxes is not None:
                for i, (face, box) in enumerate(zip(faces, boxes)):
                    # Get embedding for detected face
                    embedding = self.resnet(face.unsqueeze(0).to(self.device)).detach().cpu().numpy()
                    
                    # Find best match
                    best_match = None
                    best_score = 0
                    
                    for auth_face in self.authorized_faces:
                        # Calculate cosine similarity
                        sim = np.dot(embedding, auth_face['embedding'].T)[0][0]
                        sim = (sim + 1) / 2  # Convert to [0,1] range
                        
                        if sim > best_score and sim > threshold:
                            best_score = sim
                            best_match = auth_face
                    
                    # Draw results
                    if best_match:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{best_match['name']} ({best_score*100:.1f}%)"
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        detected.append({
                            'name': best_match['name'],
                            'confidence': best_score*100
                        })
                        
        except Exception as e:
            st.warning(f"Frame processing error: {str(e)}")
        
        return frame, detected

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Get detector from session state
        self.detector = st.session_state.detector
        self.threshold = st.session_state.detection_threshold
        self.last_log_time = 0
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img, detections = self.detector.process_frame(img, self.threshold)
        
        # Add status indicator
        status = "DETECTED" if detections else "Scanning..."
        color = (0, 255, 0) if detections else (255, 255, 255)
        cv2.putText(processed_img, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Log detections (limit frequency)
        current_time = time.time()
        if detections and current_time - self.last_log_time > 3:  # Log every 3 seconds
            self.last_log_time = current_time
            if 'detection_log' in st.session_state:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for d in detections:
                    st.session_state.detection_log.insert(0, {
                        'Timestamp': timestamp,
                        'Person': d['name'],
                        'Confidence': f"{d['confidence']:.1f}%"
                    })
                    # Keep log manageable
                    if len(st.session_state.detection_log) > 50:
                        st.session_state.detection_log.pop()
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

def main():
    st.set_page_config(page_title="Face Detection", layout="wide")
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = FaceDetector()
        st.session_state.detector.load_images()
    
    if 'detection_log' not in st.session_state:
        st.session_state.detection_log = []
    
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
        
    if 'detection_threshold' not in st.session_state:
        st.session_state.detection_threshold = 0.75
    
    st.title("Authorized Person Detection")
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload reference images", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if st.sidebar.button("Process Uploads") and uploaded_files:
        for file in uploaded_files:
            save_path = os.path.join(AUTHORIZED_DIR, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            if save_path not in st.session_state.detector.authorized_image_paths:
                st.session_state.detector.authorized_image_paths.append(save_path)
        
        # Save paths to JSON
        with open(JSON_FILE, 'w') as f:
            json.dump(st.session_state.detector.authorized_image_paths, f)
        
        # Reload faces
        st.session_state.detector.load_images()
        st.sidebar.success(f"Processed {len(uploaded_files)} images")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Camera Feed")
        
        # Threshold control
        st.session_state.detection_threshold = st.slider(
            "Recognition Threshold",
            min_value=0.5,
            max_value=0.95,
            value=st.session_state.detection_threshold,
            step=0.05
        )
        
        # Webcam control
        if st.button("Start Camera" if not st.session_state.webcam_active else "Stop Camera"):
            st.session_state.webcam_active = not st.session_state.webcam_active
            st.experimental_rerun()
        
        # WebRTC streamer
        if st.session_state.webcam_active:
            webrtc_ctx = webrtc_streamer(
                key="face-detection",
                video_processor_factory=VideoProcessor,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={
                    "video": {
                        "width": {"min": 640, "ideal": 1280},
                        "height": {"min": 480, "ideal": 720},
                        "frameRate": {"ideal": 24, "min": 15}
                    },
                    "audio": False
                },
                async_processing=True,
                mode=WebRtcMode.SENDRECV
            )
            
            if not webrtc_ctx or not webrtc_ctx.state.playing:
                st.error("""
                **Camera Access Issue**  
                Please ensure:
                - Camera permissions are enabled
                - No other app is using the camera
                - Try refreshing the page
                - Recommended browser: Chrome
                """)
    
    with col2:
        st.header("Detection Log")
        
        if st.session_state.detection_log:
            st.dataframe(st.session_state.detection_log, height=500)
        else:
            st.info("No detections yet")
        
        if st.button("Clear Log"):
            st.session_state.detection_log = []
            st.experimental_rerun()

if __name__ == "__main__":
    main()
