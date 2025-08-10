import cv2
import numpy as np

def preprocess_eye_image(eye_img):
    # Resize and normalize image for CNN input
    img = cv2.resize(eye_img, (64, 64))
    img = img / 255.0
    return img.astype(np.float32)
