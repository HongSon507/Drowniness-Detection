import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import time
import math
import numpy as np
import threading
import torch
import torch.nn as nn
from torchvision import models, transforms
import mediapipe as mp

# --- 1. CẤU HÌNH ---
MODEL_PATH = "results/best_model.pth"
IMG_SIZE = (64, 64)
CLASSES = ["AWAKE", "SLEEPY"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DROWSY_TIME_THRESHOLD = 0.3  # 

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


current_status = "KHONG THAY MAT"
prob_sleepy = 0.0
roi_box = None
thread_is_running = False
lock = threading.Lock()

def build_model():
    m = models.mobilenet_v2(weights=None) 
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(1280, 256), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(256, 2),
    )
    try:
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except Exception as e:
        pass
    m.eval()
    return m.to(DEVICE)

model = build_model()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(MEAN, STD),
])

def process_roi(roi_img):
    if roi_img.size == 0: return None
    img_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_input = img_resized[..., np.newaxis] 
    tensor = transform(img_input).unsqueeze(0).to(DEVICE)
    return tensor


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def prediction_task(frame):
    global current_status, prob_sleepy, roi_box, thread_is_running
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        
        temp_status = "KHONG THAY MAT"
        temp_prob = 0.0
        temp_box = None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                
                # Cắt 2 mắt độc lập
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                lx_min = min([int(face_landmarks.landmark[i].x * w) for i in left_eye_indices])
                lx_max = max([int(face_landmarks.landmark[i].x * w) for i in left_eye_indices])
                ly_min = min([int(face_landmarks.landmark[i].y * h) for i in left_eye_indices])
                ly_max = max([int(face_landmarks.landmark[i].y * h) for i in left_eye_indices])
                
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                rx_min = min([int(face_landmarks.landmark[i].x * w) for i in right_eye_indices])
                rx_max = max([int(face_landmarks.landmark[i].x * w) for i in right_eye_indices])
                ry_min = min([int(face_landmarks.landmark[i].y * h) for i in right_eye_indices])
                ry_max = max([int(face_landmarks.landmark[i].y * h) for i in right_eye_indices])

                pad = 15
                lx_min, lx_max = max(0, lx_min - pad), min(w, lx_max + pad)
                ly_min, ly_max = max(0, ly_min - pad), min(h, ly_max + pad)
                rx_min, rx_max = max(0, rx_min - pad), min(w, rx_max + pad)
                ry_min, ry_max = max(0, ry_min - pad), min(h, ry_max + pad)

                left_eye_img = frame[ly_min:ly_max, lx_min:lx_max]
                right_eye_img = frame[ry_min:ry_max, rx_min:rx_max]

                temp_box = [(lx_min, ly_min, lx_max, ly_max), (rx_min, ry_min, rx_max, ry_max)]

                l_tensor = process_roi(left_eye_img)
                r_tensor = process_roi(right_eye_img)

                if l_tensor is not None and r_tensor is not None:
                    with torch.no_grad():
                        out_l = model(l_tensor)
                        out_r = model(r_tensor)
                        
                        prob_l_sleepy = torch.softmax(out_l, dim=1)[0][1].item()
                        prob_r_sleepy = torch.softmax(out_r, dim=1)[0][1].item()
                        
                       
                        if prob_l_sleepy > 0.3 or prob_r_sleepy > 0.3:
                            temp_status = "SLEEPY"
                            temp_prob = max(prob_l_sleepy, prob_r_sleepy)
                        else:
                            temp_status = "AWAKE"
                            temp_prob = min(prob_l_sleepy, prob_r_sleepy)

        with lock:
            current_status = temp_status
            prob_sleepy = temp_prob
            roi_box = temp_box

    except Exception as e:
        print(f"Thread error: {e}")
    finally:
        thread_is_running = False

def main():
    global thread_is_running
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    closed_eyes_start_time = None
    is_drowsy_alarm = False
    counter = 0

    print("RUNNING -press q to exit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 3 frame / 1 detection to reduce cpu load
        if counter % 3 == 0 and not thread_is_running:
            thread_is_running = True
            threading.Thread(target=prediction_task, args=(frame.copy(),), daemon=True).start()

        with lock:
            display_status = current_status
            display_prob = prob_sleepy
            display_box = roi_box

        if display_status == "SLEEPY" or display_status == "KHONG THAY MAT":
            if closed_eyes_start_time is None:
                closed_eyes_start_time = time.time()
            else:
                elapsed_time = time.time() - closed_eyes_start_time
                if elapsed_time > DROWSY_TIME_THRESHOLD:
                    is_drowsy_alarm = True
        else:
            closed_eyes_start_time = None
            is_drowsy_alarm = False

# warning box
        if display_box is not None:
            color = (0, 0, 255) if display_status == "SLEEPY" else (0, 255, 0)
            for box in display_box:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
                cv2.putText(frame, f"{display_status} {display_prob*100:.0f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if is_drowsy_alarm:
            text = "SLEEPY"
            bg_color = (0, 0, 255)
            text_color = (255, 255, 255)
        elif display_status == "KHONG THAY MAT":
            text = "NO MASK"
            bg_color = (0, 165, 255)
            text_color = (255, 255, 255)
        else:
            text = "AWAKE"
            bg_color = (0, 255, 0)
            text_color = (0, 0, 0)

        padding = 20
        font_scale = 0.7
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(frame, (10, 10), (10 + tw + 2*padding, 10 + th + 2*padding), bg_color, -1)
        cv2.putText(frame, text, (10 + padding, 10 + th + padding), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

        cv2.imshow("Drowsiness Detection", frame)
        
        counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
