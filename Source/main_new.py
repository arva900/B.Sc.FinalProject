import cv2
import numpy as np
import time
from scipy.ndimage import binary_fill_holes
from color_model import get_channel_stats, get_color_mask
from hand_tracker import get_hand_bbox
import pyautogui

# פונקציית מרחק גאוסי
def gaussian_distance(vec, stats):
    dist = 0
    for i in range(3):
        mean, std = stats[i]
        if std < 1e-6:
            std = 1e-6
        dist += ((vec[i] - mean) ** 2) / (std ** 2)
    return dist

def handle_hand_input(x, y, closed, screen_width, screen_height, pyautogui):
    screen_x = int(x * screen_width)
    screen_y = int(y * screen_height)
    pyautogui.moveTo(screen_x, screen_y, duration=0.1)
    if closed:
        pyautogui.mouseDown()
    else:
        pyautogui.mouseUp()

def main(sampling_stats_hand, sampling_stats_ball):
    # מצלמה
    cap = cv2.VideoCapture(1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    screen_width, screen_height = pyautogui.size()
    signal = 0
    center = (0, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_disp = frame.copy()
        frame_disp = cv2.flip(frame_disp, 1)

        key = cv2.waitKey(1) & 0xFF

        bbox = get_hand_bbox(frame)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            frame_width = frame.shape[1]
            x_min_flipped = frame_width - x_max
            x_max_flipped = frame_width - x_min
            center = (int((x_min_flipped + x_max_flipped) / 2), int((y_min + y_max) / 2))

            if x_min < x_max and y_max > y_min:
                cv2.rectangle(frame_disp, (x_min_flipped, y_min), (x_max_flipped, y_max), (255, 255, 0), 2)

                sample_x_min = int((2*x_min + x_max)/3)
                sample_x_max = int((2*x_max + x_min)/3)
                sample_y_min = int((2*y_min + y_max)/3)
                sample_y_max = int((2*y_max + y_min)/3)
                small_roi = frame[sample_y_min:sample_y_max, sample_x_min:sample_x_max]
                small_x_min_flipped = frame_width - sample_x_max
                small_x_max_flipped = frame_width - sample_x_min


                if small_roi.size > 0:
                    hsv_small = cv2.cvtColor(small_roi, cv2.COLOR_BGR2HSV)
                    ycrcb_small = cv2.cvtColor(small_roi, cv2.COLOR_BGR2YCrCb)

                    s_mean = np.mean(hsv_small[:, :, 1])
                    cr_mean = np.mean(ycrcb_small[:, :, 1])
                    cb_mean = np.mean(ycrcb_small[:, :, 2])
                    feature_vec = [s_mean, cr_mean, cb_mean]

                    dist_to_hand = gaussian_distance(feature_vec, sampling_stats_hand)
                    dist_to_ball = gaussian_distance(feature_vec, sampling_stats_ball)

                    signal = 1 if (dist_to_hand < dist_to_ball) else 0

                    label = "Hand" if (signal == 1) else "Ball"
                    color = (0, 255, 0) if label == "Hand" else (0, 0, 255)

                    cv2.rectangle(frame_disp, (small_x_min_flipped, sample_y_min), (small_x_max_flipped, sample_y_max),
                                  color, 2)
                    cv2.putText(frame_disp, f"{label} ({dist_to_hand:.1f}/{dist_to_ball:.1f})", (small_x_min_flipped, sample_y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    cv2.putText(frame_disp, f"Signal: {signal}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if signal else (0, 0, 255), 3)
                    handle_hand_input(center[0]/w, center[1]/h, signal, screen_width, screen_height, pyautogui)


        else:
            cv2.putText(frame_disp, "No hand detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # signal = 0

        cv2.imshow("Tracking", frame_disp)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

