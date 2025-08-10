import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
import os
from utils import preprocess_eye_image
from sos_messenger import send_sos_message
import pyttsx3
import math

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

class BlinkDetector:
    def __init__(self, model_path=None, ear_threshold=0.21):
        self.ear_threshold = ear_threshold
        self.engine = pyttsx3.init()
        self.blink_times = []
        self.COOLDOWN = 0.25
        self.WINDOW = 2.0
        self.last_action_time = time.time()
        self.model = None

        if model_path and os.path.isfile(model_path):
            print(f"Loading model from {model_path} ...")
            self.model = tf.keras.models.load_model(model_path)
        else:
            print("No valid model path given or file missing. Falling back to EAR threshold method.")

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def calculate_EAR(self, eye_landmarks):
        A = math.dist(eye_landmarks[1], eye_landmarks[5])
        B = math.dist(eye_landmarks[2], eye_landmarks[4])
        C = math.dist(eye_landmarks[0], eye_landmarks[3])
        return (A + B) / (2.0 * C)

    def extract_eye_region(self, frame, landmarks, eye_indices):
        h, w, _ = frame.shape
        points = [(int(landmarks.landmark[i].x * w),
                   int(landmarks.landmark[i].y * h)) for i in eye_indices]
        x_min = max(min(p[0] for p in points), 0)
        y_min = max(min(p[1] for p in points), 0)
        x_max = min(max(p[0] for p in points), w)
        y_max = min(max(p[1] for p in points), h)
        eye_img = frame[y_min:y_max, x_min:x_max]
        if eye_img.size == 0:
            return None
        eye_img = cv2.resize(eye_img, (64, 64))
        return eye_img

    def predict_gesture(self, eye_img):
        img = preprocess_eye_image(eye_img)
        preds = self.model.predict(np.expand_dims(img, axis=0))
        class_id = np.argmax(preds)
        return class_id

    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
                    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

                    left_eye = self.extract_eye_region(frame, face_landmarks, LEFT_EYE_IDX)
                    right_eye = self.extract_eye_region(frame, face_landmarks, RIGHT_EYE_IDX)

                    blink_detected = False

                    if self.model is not None:
                        # Use CNN model for prediction if available
                        if left_eye is not None and right_eye is not None:
                            left_pred = self.predict_gesture(left_eye)
                            right_pred = self.predict_gesture(right_eye)
                            # Assuming class '1' means blink, adjust as per your classes
                            if left_pred == 1 and right_pred == 1 and (time.time() - self.last_action_time) > self.COOLDOWN:
                                blink_detected = True
                    else:
                        # Fallback to EAR threshold method
                        if left_eye is not None and right_eye is not None:
                            # Calculate EAR for left and right eyes
                            left_eye_points = [(int(face_landmarks.landmark[i].x * frame.shape[1]),
                                                int(face_landmarks.landmark[i].y * frame.shape[0])) for i in LEFT_EYE_IDX]
                            right_eye_points = [(int(face_landmarks.landmark[i].x * frame.shape[1]),
                                                 int(face_landmarks.landmark[i].y * frame.shape[0])) for i in RIGHT_EYE_IDX]
                            left_ear = self.calculate_EAR(left_eye_points)
                            right_ear = self.calculate_EAR(right_eye_points)
                            avg_ear = (left_ear + right_ear) / 2
                            if avg_ear < self.ear_threshold and (time.time() - self.last_action_time) > self.COOLDOWN:
                                blink_detected = True

                    if blink_detected:
                        self.blink_times.append(time.time())
                        self.last_action_time = time.time()

            # After window duration, interpret blink counts
            if self.blink_times and (time.time() - self.blink_times[-1] > self.WINDOW):
                blink_count = len(self.blink_times)

                if blink_count == 1:
                    print("Detected: YES")
                    self.speak("Yes")
                elif blink_count == 2:
                    print("Detected: NO")
                    self.speak("No")
                elif blink_count == 3:
                    print("Detected: I need help")
                    self.speak("I need help")

                    phone_number = "+91"  # Your number here
                    message = "SOS! I need immediate help. Please check on me."
                    send_sos_message(phone_number, message)
                elif blink_count == 4:
                    print("Detected: CALL")
                    self.speak("Calling")
                    # Extend with call API or logic here
                else:
                    print(f"Detected {blink_count} blinks - no mapped action")

                self.blink_times.clear()

            cv2.imshow("Eye Gesture Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
