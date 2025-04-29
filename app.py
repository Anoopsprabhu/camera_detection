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
        
        if not os.path.exists(AUTHORIZED_DIR):
            os.makedirs(AUTHORIZED_DIR)
        
        if os.path.exists(JSON_FILE):
            with open(JSON_FILE, 'r') as f:
                self.authorized_image_paths = json.load(f)
                # Filter out non-existent files
                self.authorized_image_paths = [path for path in self.authorized_image_paths if os.path.exists(path)]
                # Save filtered paths back to JSON
                with open(JSON_FILE, 'w') as f_write:
                    json.dump(self.authorized_image_paths, f_write)
        
        self.mtcnn = MTCNN(keep_all=True, device='cpu', margin=20, min_face_size=80)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        
    def load_images(self):
        """Load authorized images"""
        self.authorized_faces = []
        
        for idx, image_path in enumerate(self.authorized_image_paths):
            try:
                if not os.path.exists(image_path):
                    st.sidebar.error(f"❌ File not found: {os.path.basename(image_path)}")
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
                        'name': f"Person_{idx + 1}"
                    })
                    
                    st.sidebar.success(f"✅ Loaded authorized person: Person_{idx + 1}")
                else:
                    st.sidebar.error(f"❌ No face found in authorized image {idx + 1}")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading image {idx + 1}: {str(e)}")
        
        st.sidebar.info(f"Loaded {len(self.authorized_faces)} authorized persons")
    
    def remove_images(self):
        """Remove all authorized images and delete all files listed in JSON"""
        if os.path.exists(JSON_FILE):
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
            # Convert frame to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Get faces with MTCNN
            try:
                faces = self.mtcnn(pil_image)
                boxes = self.mtcnn.detect(pil_image)[0] if faces is not None else None
            except Exception as e:
                # If MTCNN fails, try with a resized image
                new_size = (640, 480)
                pil_image = pil_image.resize(new_size)
                faces = self.mtcnn(pil_image)
                boxes = self.mtcnn.detect(pil_image)[0] if faces is not None else None
            
            detected_persons = []
            
            if faces is not None and boxes is not None:
                # Process each detected face
                for i, face_tensor in enumerate(faces):
                    if i >= len(boxes):
                        continue
                    
                    # Get embedding
                    try:
                        face_embedding = self.resnet(face_tensor.unsqueeze(0)).detach().numpy()
                        match, confidence = self.match_face(face_embedding, threshold)
                    except Exception as e:
                        # Skip this face if embedding fails
                        continue
                    
                    if match:
                        # Get coordinates
                        box = boxes[i]
                        left, top, right, bottom = map(int, box)
                        
                        # Ensure coordinates are within image bounds
                        height, width = frame.shape[:2]
                        left = max(0, min(left, width-1))
                        top = max(0, min(top, height-1))
                        right = max(0, min(right, width-1))
                        bottom = max(0, min(bottom, height-1))
                        
                        person_name = match['name']
                        detected_persons.append({"name": person_name, "confidence": confidence})
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        
                        # Draw label background
                        text_height = 25
                        text_y_pos = min(bottom, height - 5)
                        text_bottom = min(bottom + text_height, height)
                        cv2.rectangle(frame, (left, text_y_pos - text_height), (right, text_y_pos), 
                                     (0, 255, 0), cv2.FILLED)
                        
                        # Draw label text
                        info_text = f"{person_name} ({confidence:.1f}%)"
                        cv2.putText(frame, info_text, (left + 6, text_y_pos - 6),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        except Exception as e:
            # Add error text to the frame
            cv2.putText(frame, f"Error: {str(e)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, detected_persons

def save_uploaded_file(uploaded_file):
    """Save uploaded file to the authorized_images directory and update JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    file_path = os.path.join(AUTHORIZED_DIR, filename)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            image_paths = json.load(f)
    else:
        image_paths = []
    
    image_paths.append(file_path)
    with open(JSON_FILE, 'w') as f:
        json.dump(image_paths, f)
    
    return file_path

def run_detection(detector, threshold, webcam_placeholder, log_placeholder, camera_source=0):
    """Run the detection loop independently"""
    # Try to open the webcam with specified source
    cap = cv2.VideoCapture(camera_source)
    
    # If failed, try alternative sources
    if not cap.isOpened():
        webcam_placeholder.warning(f"❌ Could not open webcam with source {camera_source}. Trying alternative sources...")
        
        # Try common camera indices
        for src in [0, 1, 2, -1]:
            if src == camera_source:
                continue  # Skip the one we already tried
                
            webcam_placeholder.info(f"Trying camera source: {src}")
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                webcam_placeholder.success(f"✅ Successfully connected to camera source: {src}")
                st.session_state.camera_source = src  # Save working source for future use
                break
                
        # If still not open, try platform-specific options
        if not cap.isOpened():
            if os.name == 'posix':  # Linux/Mac
                for dev_path in ['/dev/video0', '/dev/video1', '/dev/video2']:
                    webcam_placeholder.info(f"Trying device path: {dev_path}")
                    cap = cv2.VideoCapture(dev_path)
                    if cap.isOpened():
                        webcam_placeholder.success(f"✅ Successfully connected to: {dev_path}")
                        break
    
    # Final check if any camera source worked
    if not cap.isOpened():
        webcam_placeholder.error("❌ Could not open any webcam. Please check your camera connection and permissions.")
        st.session_state.detection_active = False
        
        # Provide test mode with a sample image
        if webcam_placeholder.button("⚠️ Use Test Mode (with sample image)"):
            # Create a sample frame
            sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(sample_frame, "TEST MODE - NO CAMERA", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            webcam_placeholder.image(sample_frame, channels="BGR", use_column_width=True)
            webcam_placeholder.warning("⚠️ Test mode active - no actual detection will occur")
        
        return
    
    try:
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Check if camera provides frames
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            webcam_placeholder.error("❌ Camera opened but not providing frames. Please check camera.")
            st.session_state.detection_active = False
            cap.release()
            return
            
        webcam_placeholder.success("✅ Camera successfully initialized!")
        
        while st.session_state.detection_active:
            ret, frame = cap.read()
            if not ret:
                webcam_placeholder.warning("⚠️ Failed to get frame from camera")
                time.sleep(1)  # Wait a bit before trying again
                continue
            
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
                log_placeholder.dataframe(st.session_state.detection_log, height=400)
            else:
                log_placeholder.info("No authorized persons detected yet")
            
            time.sleep(0.05)
    
    except Exception as e:
        webcam_placeholder.error(f"❌ Error during detection: {str(e)}")
        st.session_state.detection_active = False
    
    finally:
        cap.release()

def main():
    st.set_page_config(page_title="Authorized Person Detection", layout="wide")
    
    st.title("Authorized Person Detection System")
    st.subheader("Authorized Person")
    
    if 'detector' not in st.session_state:
        st.session_state.detector = ImprovedPersonDetector()
        st.session_state.detection_active = False
        st.session_state.detection_log = []
        st.session_state.camera_source = 0  # Default camera source
    
    col1, col2 = st.columns([2, 1])
    webcam_placeholder = col1.empty()
    log_placeholder = col2.empty()
    
    st.sidebar.header("Configuration")
    st.sidebar.subheader("Upload Authorized Persons (Optional)")
    authorized_uploads = st.sidebar.file_uploader(
        "Upload images of authorized persons", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="auth_uploads"
    )
    
    if st.sidebar.button("Load Authorized Images"):
        st.session_state.detector.authorized_image_paths = []
        for uploaded_file in authorized_uploads:
            file_path = save_uploaded_file(uploaded_file)
            st.session_state.detector.authorized_image_paths.append(file_path)
        st.session_state.detector.load_images()
    
    if st.sidebar.button("Remove Authorized Images"):
        st.session_state.detector.remove_images()
    
    # Add error handling for displaying authorized person images
    if st.session_state.detector.authorized_image_paths:
        st.sidebar.subheader("Authorized Persons")
        valid_paths = []
        
        for path in st.session_state.detector.authorized_image_paths:
            if os.path.exists(path):
                valid_paths.append(path)
        
        if valid_paths:
            cols = st.sidebar.columns(min(3, len(valid_paths)))
            for idx, img_path in enumerate(valid_paths):
                try:
                    with cols[idx % 3]:
                        img = Image.open(img_path)
                        st.image(img, caption=f"Person {idx + 1}", width=100)
                except Exception as e:
                    st.sidebar.error(f"Error displaying image: {os.path.basename(img_path)}")
        else:
            st.sidebar.warning("No valid image files found in authorized images list")
    
    with col1:
        st.header("Live Camera Feed")
        threshold = 0.75
        
        # Camera source selection
        camera_options = ["Default (0)", "Camera 1", "Camera 2", "Camera 3", "USB Camera", "Integrated Webcam"]
        camera_values = [0, 1, 2, 3, "USB", "Integrated"]
        
        cam_col1, cam_col2 = st.columns([2, 1])
        with cam_col1:
            selected_camera = st.selectbox(
                "Select Camera Source", 
                options=camera_options,
                index=0,
                help="If your webcam isn't working, try a different source"
            )
        
        # Map selection to camera source value
        camera_idx = camera_options.index(selected_camera)
        camera_source = camera_values[camera_idx]
        
        # Store the selected camera source
        if isinstance(camera_source, str):
            if camera_source == "USB":
                st.session_state.camera_source = 1  # Common for USB cameras
            elif camera_source == "Integrated":
                st.session_state.camera_source = 0  # Common for integrated webcams
        else:
            st.session_state.camera_source = camera_source
            
        with cam_col2:
            if st.button("Test Camera"):
                # Quick camera test without starting detection
                test_placeholder = st.empty()
                try:
                    test_cap = cv2.VideoCapture(st.session_state.camera_source)
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        if ret:
                            test_placeholder.image(
                                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                caption="Camera Test Successful", 
                                use_column_width=True
                            )
                            st.success(f"✅ Camera {st.session_state.camera_source} is working!")
                        else:
                            test_placeholder.error("Camera opened but could not read frame")
                    else:
                        test_placeholder.error(f"Could not open camera source: {st.session_state.camera_source}")
                except Exception as e:
                    test_placeholder.error(f"Camera test error: {str(e)}")
                finally:
                    if 'test_cap' in locals() and test_cap is not None:
                        test_cap.release()
        
        # Start/Stop button
        if st.button("Start/Stop Detection"):
            st.session_state.detection_active = not st.session_state.detection_active
            if st.session_state.detection_active:
                st.session_state.detection_log = []
                run_detection(
                    st.session_state.detector, 
                    threshold, 
                    webcam_placeholder, 
                    log_placeholder,
                    st.session_state.camera_source
                )
        
        # Alternative Detection Method
        if st.checkbox("Enable Alternative Detection (no webcam)"):
            st.info("You can upload an image for detection instead of using webcam")
            test_image = st.file_uploader("Upload image for detection", type=["jpg", "jpeg", "png"])
            
            if test_image is not None:
                image = Image.open(test_image)
                image_np = np.array(image)
                
                # Convert RGB to BGR (OpenCV format)
                if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                processed_image, detected_persons = st.session_state.detector.process_frame(image_np, threshold)
                
                # Show results
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Detection Results", use_column_width=True)
                
                if detected_persons:
                    st.success(f"Detected {len(detected_persons)} authorized persons")
                    for person in detected_persons:
                        st.info(f"✓ {person['name']} (Confidence: {person['confidence']:.1f}%)")
                        
                        # Add to log
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.detection_log.insert(0, {
                            "Timestamp": current_time,
                            "Person": person["name"],
                            "Confidence": f"{person['confidence']:.1f}%"
                        })
                else:
                    st.warning("No authorized persons detected in the image")
        
        status = "ACTIVE" if st.session_state.detection_active else "INACTIVE"
        if st.session_state.detection_active:
            st.success(f"Detection is {status}")
        else:
            st.error(f"Detection is {status}")
        
    with col2:
        # "Detection Log" heading is placed above the content
        st.header("Detection Log")
        # Table or info message is placed below the heading within a container
        with st.container():
            
            if st.session_state.detection_log:
                log_placeholder.dataframe(st.session_state.detection_log, height=400)
            else:
                log_placeholder.info("No authorized persons detected yet")
    
if __name__ == "__main__":
    main()
