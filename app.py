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

AUTHORIZED_DIR = "authorized_images"
JSON_FILE = "authorized_images.json"

class ImprovedPersonDetector:
    def __init__(self):
        self.authorized_faces = []
        self.authorized_image_paths = []

        if not os.path.exists(AUTHORIZED_DIR):
            os.makedirs(AUTHORIZED_DIR)

        if os.path.exists(JSON_FILE):
            with open(JSON_FILE, 'r') as f:
                self.authorized_image_paths = json.load(f)

        self.mtcnn = MTCNN(keep_all=True, device='cpu', margin=20, min_face_size=80)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def load_images(self):
        self.authorized_faces = []
        valid_paths = []

        for idx, image_path in enumerate(self.authorized_image_paths):
            if not os.path.exists(image_path):
                st.sidebar.warning(f"⚠️ Image not found: {os.path.basename(image_path)}")
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
                    valid_paths.append(image_path)
                    st.sidebar.success(f"✅ Loaded authorized person: Person_{idx + 1}")
                else:
                    st.sidebar.error(f"❌ No face found in image {os.path.basename(image_path)}")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {image_path}: {e}")

        self.authorized_image_paths = valid_paths
        with open(JSON_FILE, 'w') as f:
            json.dump(valid_paths, f)
        st.sidebar.info(f"✔️ Loaded {len(self.authorized_faces)} authorized persons")

    def remove_images(self):
        if os.path.exists(JSON_FILE):
            with open(JSON_FILE, 'r') as f:
                image_paths = json.load(f)

            for file_path in image_paths:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        st.sidebar.success(f"🗑️ Deleted: {os.path.basename(file_path)}")
                    except Exception as e:
                        st.sidebar.error(f"❌ Failed to delete {file_path}: {e}")
                else:
                    st.sidebar.warning(f"File not found: {os.path.basename(file_path)}")

            with open(JSON_FILE, 'w') as f:
                json.dump([], f)

        self.authorized_faces = []
        self.authorized_image_paths = []
        st.sidebar.success("🧹 Authorized images cleared")

    def match_face(self, face_embedding, threshold=0.75):
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

        return frame, detected_persons

def save_uploaded_file(uploaded_file):
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

def run_detection(detector, threshold, webcam_placeholder, log_placeholder):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        webcam_placeholder.error("❌ Could not open webcam")
        st.session_state.detection_active = False
        return

    try:
        while st.session_state.detection_active:
            ret, frame = cap.read()
            if not ret:
                break

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

    if st.session_state.detector.authorized_image_paths:
        st.sidebar.subheader("Authorized Persons")
        cols = st.sidebar.columns(min(3, len(st.session_state.detector.authorized_image_paths)))
        for idx, img_path in enumerate(st.session_state.detector.authorized_image_paths):
            with cols[idx % 3]:
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        st.image(img, caption=f"Person {idx + 1}", width=100)
                    except Exception as e:
                        st.warning(f"❌ Cannot load image {idx + 1}: {e}")
                else:
                    st.warning(f"⚠️ Missing file: {os.path.basename(img_path)}")

    with col1:
        st.header("Live Camera Feed")
        threshold = 0.75

        if st.button("Start/Stop Detection"):
            st.session_state.detection_active = not st.session_state.detection_active
            if st.session_state.detection_active:
                st.session_state.detection_log = []
                run_detection(st.session_state.detector, threshold, webcam_placeholder, log_placeholder)

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

if __name__ == "__main__":
    main()
