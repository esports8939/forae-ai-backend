# -----------------------------------------------------
# FORAE AI – SKIN ANALYSIS ENGINE
# -----------------------------------------------------

import cv2
import numpy as np


# -----------------------------------------------------
# FACE ALIGNMENT
# -----------------------------------------------------
def align_face(image):
    """
    Simple face alignment using HaarCascade.
    Works well for centered faces.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return image

    # (Optional) Rotation can be added later
    return image


# -----------------------------------------------------
# FACE REGION COORDINATES
# -----------------------------------------------------
def get_face_regions(image):
    h, w = image.shape[:2]

    return {
        "forehead": (int(w * 0.25), int(h * 0.05), int(w * 0.5), int(h * 0.20)),
        "left_cheek": (int(w * 0.10), int(h * 0.30), int(w * 0.25), int(h * 0.25)),
        "right_cheek": (int(w * 0.65), int(h * 0.30), int(w * 0.25), int(h * 0.25)),
        "nose": (int(w * 0.40), int(h * 0.35), int(w * 0.20), int(h * 0.20)),
        "chin": (int(w * 0.35), int(h * 0.60), int(w * 0.30), int(h * 0.25)),
    }


def crop_region(img, box):
    x, y, w, h = box
    return img[y:y + h, x:x + w].copy()


# -----------------------------------------------------
# ACNE DETECTION
# -----------------------------------------------------
def detect_acne(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 80, 60])
    upper1 = np.array([10, 255, 255])

    lower2 = np.array([170, 80, 60])
    upper2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # Texture-based filtering
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    texture = cv2.absdiff(gray, blur)
    _, tex_mask = cv2.threshold(texture, 12, 255, cv2.THRESH_BINARY)

    final = cv2.bitwise_and(mask, tex_mask)

    # Contour filtering (remove noise)
    contours, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(final)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 800:
            cv2.drawContours(cleaned, [cnt], -1, 255, -1)

    severity = np.sum(cleaned > 0)

    return {"score": int(severity), "mask": cleaned}


# -----------------------------------------------------
# PORE DETECTION
# -----------------------------------------------------
def detect_pores(region):
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    lap = cv2.Laplacian(blur, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)

    _, mask = cv2.threshold(lap, 22, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, _ = cv2.connectedComponents(mask)

    return {"score": int(num_labels - 1), "mask": mask}


# -----------------------------------------------------
# WRINKLE DETECTION
# -----------------------------------------------------
def detect_wrinkles(region):
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    edges = cv2.Canny(enhanced, 20, 80)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thin = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    score = np.sum(thin > 0)

    return {"score": int(score), "mask": thin}


# -----------------------------------------------------
# REDNESS DETECTION
# -----------------------------------------------------
def detect_redness(region):
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    _, A, _ = cv2.split(lab)

    A_norm = cv2.normalize(A, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(A_norm, 135, 255, cv2.THRESH_BINARY)

    score = np.sum(mask > 0)

    return {"score": int(score), "mask": mask}


# -----------------------------------------------------
# OILINESS DETECTION
# -----------------------------------------------------
def detect_oiliness(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = ((v > 200) & (s < 80)).astype(np.uint8) * 255

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    score = np.sum(mask > 0)

    return {"score": int(score), "mask": mask}


# -----------------------------------------------------
# SKIN TONE ANALYSIS
# -----------------------------------------------------
def detect_skin_tone(region):
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    avg_L = np.mean(L)
    avg_A = np.mean(A)
    avg_B = np.mean(B)

    # Tone depth
    if avg_L > 180:
        depth = "Very Light"
    elif avg_L > 150:
        depth = "Light"
    elif avg_L > 120:
        depth = "Medium"
    elif avg_L > 90:
        depth = "Tan / Medium-Deep"
    else:
        depth = "Deep"

    # Undertone
    if avg_B > 140:
        undertone = "Warm"
    elif avg_A > 140:
        undertone = "Neutral-Warm"
    elif avg_B < 120:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    return {"depth": depth, "undertone": undertone}


# -----------------------------------------------------
# FULL PIPELINE
# -----------------------------------------------------
def analyze_face(image):

    aligned = align_face(image)
    regions = get_face_regions(aligned)

    results = {}

    for name, box in regions.items():
        crop = crop_region(aligned, box)

        results[name] = {
            "acne": detect_acne(crop),
            "pores": detect_pores(crop),
            "wrinkles": detect_wrinkles(crop),
            "redness": detect_redness(crop),
            "oiliness": detect_oiliness(crop),
            "skin_tone": detect_skin_tone(crop),
            "crop": crop
        }

    return aligned, regions, results
