import streamlit as st
import os
import json
from datetime import datetime
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

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
                    st.sidebar.success(f"✅ Loaded: Person_{idx + 1}")
                else:
                    st.sidebar.error(f"❌ No face found in {os.path.basename(image_path)}")
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
                    st.sidebar.warning(f"⚠️ Missing: {os.path.basename(file_path)}")

            with open(JSON_FILE, 'w') as f:
                json.dump([], f)

        self.authorized_faces = []
        self.authorized_image_paths = []
        st.sidebar.success("🧹 All authorized images removed")

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

# --- Video Transformer using streamlit-webrtc ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self, detector, threshold=0.75):
        self.detector = detector
        self.threshold = threshold

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        faces = self.detector.mtcnn(pil_img)
        boxes, _ = self.detector.mtcnn.detect(pil_img)

        if faces is not None and boxes is not None:
            for i, face_tensor in enumerate(faces):
                face_embedding = self.detector.resnet(face_tensor.unsqueeze(0)).detach().numpy()
                match, confidence = self.detector.match_face(face_embedding, self.threshold)
                if match:
                    box = boxes[i]
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{match['name']} ({confidence:.1f}%)"
                    cv2.putText(img, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return img

def save_uploaded_file(uploaded_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    file_path = os.path.join(AUTHORIZED_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    image_paths = []
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            image_paths = json.load(f)
    image_paths.append(file_path)
    with open(JSON_FILE, 'w') as f:
        json.dump(image_paths, f)

    return file_path

def main():
    st.set_page_config(page_title="Authorized Person Detection", layout="wide")
    st.title("Authorized Person Detection System")

    if 'detector' not in st.session_state:
        st.session_state.detector = ImprovedPersonDetector()
        st.session_state.detection_log = []

    col1, col2 = st.columns([2, 1])

    st.sidebar.header("Configuration")
    authorized_uploads = st.sidebar.file_uploader(
        "Upload authorized person images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
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
                        st.warning(f"❌ Can't load image {idx + 1}: {e}")
                else:
                    st.warning(f"⚠️ Missing file: {os.path.basename(img_path)}")

    with col1:
        st.header("Live Camera Feed")
        st.info("Turn on webcam and allow browser access")
        webrtc_streamer(
            key="person-detect",
            video_transformer_factory=lambda: VideoProcessor(st.session_state.detector)
        )

    with col2:
        st.header("Detection Log")
        if st.session_state.detection_log:
            st.dataframe(st.session_state.detection_log, height=400)
        else:
            st.info("No authorized persons detected yet")

if __name__ == "__main__":
    main()
