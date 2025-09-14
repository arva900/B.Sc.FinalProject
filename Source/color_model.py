import cv2
import numpy as np

def get_channel_stats(image_roi, channel_index):
    """
    מחשב תוחלת ושונות של ערוץ יחיד (כמו H, S, או Cr)
    """
    channel = image_roi[:, :, channel_index]
    mu = np.mean(channel)
    sigma = np.std(channel)
    return mu, sigma

def get_color_mask(image_roi, stats_list, k=2.5):
    """
    יוצרת מסכה בינארית: פיקסלים שנמצאים בטווח של μ ± kσ בכל אחד מהערוצים
    משתמש בערוצים: S (HSV), Cr, Cb (YCrCb) – כמו בפונקציית הדגימה
    """
    hsv = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image_roi, cv2.COLOR_BGR2YCrCb)

    S = hsv[:, :, 1]
    Cr = ycrcb[:, :, 1]
    Cb = ycrcb[:, :, 2]

    s_mu, s_sigma = stats_list[0]
    cr_mu, cr_sigma = stats_list[1]
    cb_mu, cb_sigma = stats_list[2]

    s_mask = np.abs(S.astype(np.float32) - s_mu) < k * s_sigma
    cr_mask = np.abs(Cr.astype(np.float32) - cr_mu) < k * cr_sigma
    cb_mask = np.abs(Cb.astype(np.float32) - cb_mu) < k * cb_sigma

    # אפשר לבחור: חייבים לפחות 2 מתוך 3 (או כולם)
    combined_mask = (s_mask.astype(int) + cr_mask.astype(int) + cb_mask.astype(int)) >= 2

    return (combined_mask.astype(np.uint8)) * 255
