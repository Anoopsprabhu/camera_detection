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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import uuid

# Configuration
AUTHORIZED_DIR = "authorized_images"
JSON_FILE = "authorized_images.json"
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class ImprovedFaceDetector:
    def __init__(self):
        """Initialize the detector with authorized images"""
        self.authorized_faces = []
        self.authorized_image_paths = []
        
        os.makedirs(AUTHORIZED_DIR, exist_ok=True)
        
        if os.path.exists(JSON_FILE):
            try:
                with open(JSON_FILE, 'r') as f:
                    self.authorized_image_paths = json.load(f)
                # Validate paths
                self.authorized_image_paths = [path for path in self.authorized_image_paths if os.path.exists(path)]
                with open(JSON_FILE, 'w') as f:
                    json.dump(self.authorized_image_paths, f)
            except json.JSONDecodeError:
                with open(JSON_FILE, 'w') as f:
                    json.dump([], f)
        
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.mtcnn = MTCNN(
                keep_all=True,
                device=self.device,
                margin=20,
                min_face_size=80
            )
            self.resnet = InceptionResnetV1(
                pretrained='vggface2'
            ).eval().to(self.device)
            st.sidebar.success("✅ Models loaded successfully")
        except Exception as e:
            st.sidebar.error(f"❌ Model loading failed: {str(e)}")
            raise

    def load_images(self):
        """Load authorized images"""
        self.authorized_faces = []
        
        for idx, image_path in enumerate(self.authorized_image_paths):
            if not os.path.exists(image_path):
                st.sidebar.warning(f"Image file not found: {image_path}")
                continue
            try:
                person_image = Image.open(image_path).convert('RGB')
                faces = self.mtcnn(person_image)
                
                if faces is not None and len(faces) > 0:
                    face_tensor = faces[0]
                    face_embedding = self.resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()
                    
                    self.authorized_faces.append({
                        'id': idx + 1,
                        'image': person_image,
                        'embedding': face_embedding,
                        'name': f"Person_{idx + 1}"
                    })
                    st.sidebar.success(f"✅ Loaded authorized person: Person_{idx + 1}")
                else:
                    st.sidebar.error(f"❌ No face found in authorized image {idx + 1}")
            except Exception as e:
                st.sidebar.error(f"❌ Error processing {image_path}: {str(e)}")
        
        st.sidebar.info(f"Loaded {len(self.authorized_faces)} authorized persons")

    def remove_images(self):
        """Remove all authorized images and delete all files listed in JSON"""
        if os.path.exists(JSON_FILE):
            try:
                with open(JSON_FILE, 'r') as f:
                    image_paths = json.load(f)
                
                for file_path in image_paths:
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
                st.sidebar.error(f"❌ Error removing images: {str(e)}")
        
        self.authorized_faces = []
        self.authorized_image_paths = []
        st.sidebar.success("🗑️ All authorized images removed")

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
                        
                        face_embedding = self.resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()
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
            
            return frame, detected_persons
        except Exception as e:
            st.warning(f"Frame processing error: {str(e)}")
            return frame, []

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = st.session_state.detector
        self.threshold = st.session_state.detection_threshold
        self.last_log_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img, detections = self.detector.process_frame(img, self.threshold)
        
        status = "DETECTED" if detections else "Scanning..."
        color = (0, 255, 0) if detections else (255, 255, 255)
        cv2.putText(processed_img, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if detections and time.time() - self.last_log_time > 3:
            self.last_log_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for person in detections:
                st.session_state.detection_log.insert(0, {
                    "Timestamp": timestamp,
                    "Person": person["name"],
                    "Confidence": f"{person['confidence']:.1f}%"
                })
                if len(st.session_state.detection_log) > 50:
                    st.session_state.detection_log.pop()
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

def save_uploaded_file(uploaded_file):
    """Save uploaded file with unique timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    file_path = os.path.join(AUTHORIZED_DIR, filename)
    
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        if not os.path.exists(file_path):
            st.error(f"Failed to save file: {file_path}")
            return None
        st.success(f"Saved file: {file_path}")
        return file_path
    except Exception as e:
        st.error(f"Error saving file {filename}: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Authorized Person Detection", layout="wide")
    
    if 'detector' not in st.session_state:
        st.session_state.detector = ImprovedFaceDetector()
        st.session_state.webcam_active = False
        st.session_state.detection_log = []
        st.session_state.detection_threshold = 0.75
    
    st.title("Authorized Person Detection System")
    st.write(f"Current working directory: {os.getcwd()}")  # Debug
    st.write(f"Authorized directory: {os.path.abspath(AUTHORIZED_DIR)}")  # Debug
    
    col1, col2 = st.columns([2, 1])
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        st.subheader("Upload Authorized Persons")
        
        uploaded_files = st.file_uploader(
            "Upload images of authorized persons",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="auth_uploads"
        )
        
        if st.button("Load Authorized Images") and uploaded_files:
            st.session_state.detector.authorized_image_paths = []
            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    st.session_state.detector.authorized_image_paths.append(file_path)
            
            with open(JSON_FILE, 'w') as f:
                json.dump(st.session_state.detector.authorized_image_paths, f)
            
            st.session_state.detector.load_images()
        
        if st.button("Remove Authorized Images"):
            st.session_state.detector.remove_images()
        
        if st.session_state.detector.authorized_image_paths:
            st.subheader("Authorized Persons")
            cols = st.columns(min(3, len(st.session_state.detector.authorized_image_paths)))
            for idx, img_path in enumerate(st.session_state.detector.authorized_image_paths):
                st.write(f"Attempting to open: {img_path}")  # Debug
                if not os.path.exists(img_path):
                    st.warning(f"Image file not found: {img_path}")
                    continue
                try:
                    with cols[idx % 3]:
                        img = Image.open(img_path)
                        st.image(img, caption=f"Person {idx + 1}", width=100)
                except Exception as e:
                    st.error(f"Failed to load image {img_path}: {str(e)}")
                    continue

    with col1:
        st.header("Live Camera Feed")
        st.session_state.detection_threshold = st.slider(
            "Recognition Threshold",
            min_value=0.5,
            max_value=0.95,
            value=st.session_state.detection_threshold,
            step=0.05
        )
        
        if st.button("Start/Stop Camera"):
            st.session_state.webcam_active = not st.session_state.webcam_active
        
        if st.session_state.webcam_active:
            webrtc_ctx = webrtc_streamer(
                key=f"face-detection-{uuid.uuid4()}",
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
            
            status = "ACTIVE" if webrtc_ctx and webrtc_ctx.state.playing else "INACTIVE"
            if status == "ACTIVE":
                st.success(f"Detection is {status}")
            else:
                st.error(f"Detection is {status}\nPlease ensure camera permissions are enabled")
    
    with col2:
        st.header("Detection Log")
        with st.container():
            if st.session_state.detection_log:
                st.dataframe(st.session_state.detection_log, height=400)
            else:
                st.info("No authorized persons detected yet")
            
            if st.button("Clear Log"):
                st.session_state.detection_log = []
                st.experimental_rerun()

if __name__ == "__main__":
    main()
