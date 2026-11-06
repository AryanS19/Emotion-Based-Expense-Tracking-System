import cv2
from deepface import DeepFace
import csv
from datetime import datetime

def log_expense(amount, category):
    # 1. Initialize camera for a quick snapshot
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release() # Release immediately to save resources

    dominant_emotion = "Unknown"
    
    if ret:
        try:
            # 2. Analyze the user's face at the moment of spending
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = results[0]['dominant_emotion']
        except Exception:
            dominant_emotion = "Neutral"

    # 3. Save the correlated data to a CSV file
    file_exists = False
    try:
        with open('expense_mood_data.csv', 'r') as f: file_exists = True
    except FileNotFoundError: pass

    with open('expense_mood_data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        # Add headers if it's a new file
        if not file_exists:
            writer.writerow(['Date', 'Amount', 'Category', 'Emotion'])
        
        # Log the entry
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), amount, category, dominant_emotion])
    
    print(f"Logged: {amount} for {category} while feeling {dominant_emotion}")

# Example Usage:
amount_input = input("Enter amount spent: ")
cat_input = input("Enter category (Food/Travel/Games): ")
log_expense(amount_input, cat_input)