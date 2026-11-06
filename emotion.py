import cv2
import os
from deepface import DeepFace

# Set up Cascade Classifier safely
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)

# --- NEW VARIABLES FOR OPTIMIZATION ---
frame_count = 0
skip_frames = 5  # Only run AI analysis every 5th frame
emotions_dict = {} # Persistent dictionary to keep the HUD visible
# --------------------------------------

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # ONLY RUN AI ANALYSIS ON SPECIFIC FRAMES
        if frame_count % skip_frames == 0:
            try:
                face_roi = frame[y:y+h, x:x+w]
                results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotions_dict = results[0]['emotion']
            except Exception:
                pass

        # DRAW HUD (Always visible, but updates only every 5th frame)
        if emotions_dict:
            display_emotions = ['happy', 'sad', 'angry', 'surprise', 'neutral']
            for i, emo_name in enumerate(display_emotions):
                score = emotions_dict.get(emo_name, 0)
                bar_x, bar_y = x + w + 10, y + (i * 30)
                bar_width = int(score * 1.5)
                
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 150, bar_y + 20), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (255, 200, 0), -1)
                cv2.putText(frame, f"{emo_name}: {int(score)}%", (bar_x, bar_y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow('Optimized HUD - VIT Bhopal', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()`