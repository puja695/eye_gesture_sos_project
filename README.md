# ÔøΩÔøΩÔ∏è Eye Gesture SOS Project

An AI-powered system that detects eye gestures and sends automated SOS alerts instantly.

## Ì∫Ä Features

- Robust blink detection powered by CNN and MediaPipe  
- Multi-blink gesture commands: Yes, No, SOS, Call  
- Real-time webcam processing for smooth user experience  
- Automated WhatsApp message sending without manual input  

## ‚öôÔ∏è Setup Instructions

1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

2. Run the application:  
   ```bash
   python src/main.py
   ```  

## Ì≥ã Requirements

- Python 3.7 or higher  
- TensorFlow, OpenCV, MediaPipe, PyAutoGUI, PyWhatKit, pyttsx3  

## Ì¥î Important Notes

- Ensure WhatsApp Web is logged in and active in your default browser.  
- Adjust `pyautogui` click coordinates in `src/sos_messenger.py` to fit your screen resolution for reliable message sending.  

---

Made for safety using hands-free emergency communication  

