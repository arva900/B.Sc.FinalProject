from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np
import time
import pickle
from sklearn import svm
import threading
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

landmark_size = 21 * 2  # x + y
model_path = 'svm_model.pkl'

def handle_hand_input(x, y, closed, screen_width, screen_height, pyautogui):
    screen_x = int(x * screen_width)
    screen_y = int(y * screen_height)
    pyautogui.moveTo(screen_x, screen_y, duration=0.1)
    if closed:
        pyautogui.mouseDown()
    else:
        pyautogui.mouseUp()

def wait_for_hand(duration=5, padding_ratio=0.2):
    cap = cv2.VideoCapture(1)
    start_time = None
    roi_coords = None

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            if start_time is None:
                start_time = time.time()

            for hand in results.multi_hand_landmarks:
                x_list = [lm.x for lm in hand.landmark]
                y_list = [lm.y for lm in hand.landmark]
                min_x, max_x = int(min(x_list) * w), int(max(x_list) * w)
                min_y, max_y = int(min(y_list) * h), int(max(y_list) * h)

                dx = int((max_x - min_x) * padding_ratio)
                dy = int((max_y - min_y) * padding_ratio)

                roi_coords = (
                    max(0, min_x - dx),
                    max(0, min_y - dy),
                    min(w, max_x + dx),
                    min(h, max_y + dy)
                )

                cv2.rectangle(frame, (roi_coords[0], roi_coords[1]), (roi_coords[2], roi_coords[3]), (0, 255, 255), 2)

        if start_time is not None:
            cv2.putText(frame, "Place your hand inside the box...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elapsed = time.time() - start_time
            remaining = int(duration - elapsed)
            cv2.putText(frame, f"{remaining}s", (w - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 100, 255), 3)

            if elapsed >= duration:
                break
        else:
            cv2.putText(frame, "Waiting for hand...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return roi_coords

def record_gesture(label, roi, duration=5):
    cap = cv2.VideoCapture(1)
    data = []
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                features = []
                for lm in hand.landmark:
                    features.extend([lm.x, lm.y])
                if len(features) == landmark_size:
                    data.append((features, label))

        if roi:
            x1, y1, x2, y2 = roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        cv2.putText(frame, f"Recording gesture: {label}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        remaining = int(duration - (time.time() - start_time))
        cv2.putText(frame, f"{remaining}s", (w - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 100), 3)

        cv2.imshow("Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return data

def train_svm(data):
    X = [item[0] for item in data]
    y = [item[1] for item in data]
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(X, y)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model

def realtime_classification(model, screen_width, screen_height, pyautogui, handle_hand_input):
    cap = cv2.VideoCapture(1)
    save_dir = "saved_images"
    os.makedirs(save_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                features = []
                for lm in hand.landmark:
                    features.extend([lm.x, lm.y])
                if len(features) == landmark_size:
                    wrist = hand.landmark[0]
                    wrist_x = wrist.x
                    wrist_y = wrist.y
                    prediction = model.predict([features])[0]
                    prob = model.predict_proba([features])[0]
                    confidence = max(prob)
                    if(prob[1] > 0.7):
                        prediction = 1
                    else:
                        prediction = 0
                    cv2.putText(frame, f"Gesture: {prediction} ({confidence:.2f})", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    handle_hand_input(wrist_x, wrist_y, int(prediction), screen_width, screen_height, pyautogui)

                    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    # filename = f"{save_dir}/gesture_{prediction}_{timestamp}.jpg"
                    # cv2.imwrite(filename, frame)

        cv2.imshow("Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
