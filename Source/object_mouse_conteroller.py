import cv2
import time
import numpy as np
from main_new import main as main_object_track

SAMPLE_HAND, SAMPLE_BALL, COMPARE = 0, 1, 2
sample_box_size = 100
sample_duration = 7  # זמן דגימה בשניות

def get_channel_stats(roi, channel_index):
    channel = roi[:, :, channel_index]
    mean = np.mean(channel)
    std = np.std(channel)
    return (mean, std)

def gaussian_model_distance(stats1, stats2):
    dist = 0
    for (m1, s1), (m2, s2) in zip(stats1, stats2):
        dist += abs(m1 - m2) / (s1 + s2 + 1e-6)
    return dist

def sample_objects_from_video():
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sampling_stats_hand = None
    sampling_stats_ball = None
    start_time = None
    state = SAMPLE_HAND

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_disp = cv2.flip(frame.copy(), 1)

        center_x, center_y = w // 2, h // 2
        half_size = sample_box_size // 2
        x1, y1 = center_x - half_size, center_y - half_size
        x2, y2 = center_x + half_size, center_y + half_size

        if state in [SAMPLE_HAND, SAMPLE_BALL]:
            label = "hand" if state == SAMPLE_HAND else "same hand or object"

            if start_time is None:
                start_time = time.time()

            elapsed = time.time() - start_time
            remaining = int(sample_duration - elapsed + 1)

            # ציור מסגרת וטקסט
            cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame_disp, f"Sampling {label} in: {remaining}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Sampling", frame_disp)
            cv2.waitKey(1)

            if elapsed >= sample_duration:
                roi = frame[y1:y2, x1:x2]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                ycrcb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

                stats_s = get_channel_stats(hsv_roi, 1)     # S
                stats_cr = get_channel_stats(ycrcb_roi, 1)  # Cr
                stats_cb = get_channel_stats(ycrcb_roi, 2)  # Cb
                stats = [stats_s, stats_cr, stats_cb]

                if state == SAMPLE_HAND:
                    sampling_stats_hand = stats
                    state = SAMPLE_BALL
                    start_time = None
                elif state == SAMPLE_BALL:
                    sampling_stats_ball = stats
                    state = COMPARE
                    break

    cap.release()
    cv2.destroyAllWindows()

    dist = gaussian_model_distance(sampling_stats_hand, sampling_stats_ball)
    print("dist is " + str(dist))
    is_same = dist < 2

    return sampling_stats_hand, sampling_stats_ball, is_same

# הפעלת הקוד בפועל

