import time
import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

def calculate_EAR(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calibrate_eye_aspect_ratio(duration=10):
    cap = cv2.VideoCapture(0)
    ears = []
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb)

        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            for face_landmarks in results.multi_face_landmarks:
                LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]

                points = np.array([(face_landmarks.landmark[i].x * w,
                                    face_landmarks.landmark[i].y * h) for i in LEFT_EYE_IDX])

                ear = calculate_EAR(points)
                ears.append(ear)

        cv2.imshow('Calibration', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if ears:
        open_ear_avg = np.percentile(ears, 90)
        closed_ear_avg = np.percentile(ears, 10)
        print(f"Open Eye EAR ~ {open_ear_avg:.3f}")
        print(f"Closed Eye EAR ~ {closed_ear_avg:.3f}")
        threshold = (open_ear_avg + closed_ear_avg) / 2
        print(f"Suggested EAR Threshold: {threshold:.3f}")
        return threshold
    else:
        print("No EAR data collected.")
        return None

if __name__ == "__main__":
    calibrate_eye_aspect_ratio()
