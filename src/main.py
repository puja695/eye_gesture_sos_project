from blink_detector import BlinkDetector

if __name__ == "__main__":
    model_path = None  # Disable CNN model loading, use EAR threshold method
    detector = BlinkDetector(model_path=model_path, ear_threshold=0.21)
    detector.run()