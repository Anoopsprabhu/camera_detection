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

# Define the directory and JSON file
AUTHORIZED_DIR = "authorized_images"
JSON_FILE = "authorized_images.json"

# Improved RTC Configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ],
    "iceTransportPolicy": "relay"
})

class ImprovedPersonDetector:
    def __init__(self):
        """Initialize the detector with authorized images"""
        self.authorized_faces = []
        self.authorized_image_paths = []
        
        # Ensure directory exists
        os.makedirs(AUTHORIZED_DIR, exist_ok=True)
        
        # Load existing image paths
        if os.path.exists(JSON_FILE):
            try:
                with open(JSON_FILE, 'r') as f:
                    self.authorized_image_paths = json.load(f)
            except json.JSONDecodeError:
                st.sidebar.error("❌ Error reading JSON file. Creating a new one.")
                with open(JSON_FILE, 'w') as f:
                    json.dump([], f)
        
        # Initialize models
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(keep_all=True, device=self.device, margin=20)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
    def load_images(self):
        """Load authorized images"""
        self.authorized_faces = []
        valid_paths = []
        
        for idx, image_path in enumerate(self.authorized_image_paths):
            if not os.path.exists(image_path):
                st.sidebar.warning(f"❌ Image not found: {image_path}")
                continue
                
            try:
                img = Image.open(image_path).convert('RGB')
                faces = self.mtcnn(img)
                
                if faces is not None:
                    face_embedding = self.resnet(faces[0].unsqueeze(0).to(self.device)).detach().cpu().numpy()
                    self.authorized_faces.append({
                        'id': idx + 1,
                        'image': img,
                        'embedding': face_embedding,
                        'name': f"Person_{idx + 1}"
                    })
                    valid_paths.append(image_path)
            except Exception as e:
                st.sidebar.error(f"❌ Error processing image: {str(e)}")
        
        self.authorized_image_paths = valid_paths
        with open(JSON_FILE, 'w') as f:
            json.dump(valid_paths, f)
            
    def process_frame(self, frame, threshold=0.75):
        """Process frame to detect authorized persons"""
        detected_persons = []
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Detect faces
            faces = self.mtcnn(pil_img)
            boxes, _ = self.mtcnn.detect(pil_img)
            
            if faces is not None and boxes is not None:
                for i, (face, box) in enumerate(zip(faces, boxes)):
                    # Get embedding
                    embedding = self.resnet(face.unsqueeze(0).to(self.device)).detach().cpu().numpy()
                    
                    # Find best match
                    best_match, confidence = None, 0
                    for auth_face in self.authorized_faces:
                        sim = np.dot(embedding, auth_face['embedding'].T)
                        sim = (sim + 1) / 2  # Convert to [0,1] range
                        if sim > confidence and sim > threshold:
                            confidence = sim
                            best_match = auth_face
                    
                    if best_match:
                        left, top, right, bottom = map(int, box)
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        label = f"{best_match['name']} ({confidence[0][0]*100:.1f}%)"
                        cv2.putText(frame, label, (left, top-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        detected_persons.append({
                            "name": best_match['name'],
                            "confidence": confidence[0][0]*100
                        })
                        
        except Exception as e:
            st.warning(f"Processing error: {str(e)}")
            
        return frame, detected_persons

class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self, detector, threshold):
        self.detector = detector
        self.threshold = threshold
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img, detections = self.detector.process_frame(img, self.threshold)
        
        # Add status text
        status = "✅ DETECTED" if detections else "Scanning..."
        color = (0, 255, 0) if detections else (255, 255, 255)
        cv2.putText(processed_img, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Log detections
        if detections and 'detection_log' in st.session_state:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for person in detections:
                st.session_state.detection_log.insert(0, {
                    "Timestamp": timestamp,
                    "Person": person["name"],
                    "Confidence": f"{person['confidence']:.1f}%"
                })
                if len(st.session_state.detection_log) > 100:
                    st.session_state.detection_log.pop()
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

def main():
    st.set_page_config(page_title="Face Detection", layout="wide")
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = ImprovedPersonDetector()
    if 'detection_log' not in st.session_state:
        st.session_state.detection_log = []
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    
    st.title("Authorized Person Detection System")
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Upload images
    uploaded_files = st.sidebar.file_uploader(
        "Upload authorized person images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if st.sidebar.button("Process Uploaded Images") and uploaded_files:
        for file in uploaded_files:
            path = os.path.join(AUTHORIZED_DIR, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            if path not in st.session_state.detector.authorized_image_paths:
                st.session_state.detector.authorized_image_paths.append(path)
        st.session_state.detector.load_images()
        st.sidebar.success(f"Processed {len(uploaded_files)} images")
    
    if st.sidebar.button("Remove All Authorized Images"):
        st.session_state.detector.authorized_faces = []
        st.session_state.detector.authorized_image_paths = []
        with open(JSON_FILE, 'w') as f:
            json.dump([], f)
        st.sidebar.success("Cleared all authorized images")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Live Detection")
        threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.75, 0.05)
        
        # Webcam control
        if st.button("Start Detection" if not st.session_state.webcam_active else "Stop Detection"):
            st.session_state.webcam_active = not st.session_state.webcam_active
            st.experimental_rerun()
        
        if st.session_state.webcam_active:
            webrtc_ctx = webrtc_streamer(
                key="face-detection",
                video_processor_factory=lambda: FaceDetectionProcessor(
                    st.session_state.detector,
                    threshold
                ),
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={
                    "video": {
                        "width": {"min": 640, "ideal": 1280},
                        "height": {"min": 480, "ideal": 720},
                        "frameRate": {"ideal": 30, "min": 15}
                    },
                    "audio": False
                },
                async_processing=True,
                mode=WebRtcMode.SENDRECV
            )
            
            if not webrtc_ctx or not webrtc_ctx.state.playing:
                st.error("""
                **Camera not accessible**  
                Please ensure:
                - Camera permissions are enabled
                - No other app is using the camera
                - Try refreshing the page
                - Chrome works best for this application
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
