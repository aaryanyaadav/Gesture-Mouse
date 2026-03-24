import cv2
import pyautogui
import math
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

# Initialize camera
cap = cv2.VideoCapture(0)

# Screen size
screen_w, screen_h = pyautogui.size()

# Setup MediaPipe HandLandmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )   

    # Detect hands
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Index finger tip (8)
        x1 = int(hand[8].x * w)
        y1 = int(hand[8].y * h)

        # Thumb tip (4)
        x2 = int(hand[4].x * w)
        y2 = int(hand[4].y * h)

        # Draw points
        cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
        cv2.circle(frame, (x2, y2), 10, (0, 255, 0), -1)

        # Middle finger tip (12)
        x3 = int(hand[12].x * w)
        y3 = int(hand[12].y * h)

        cv2.circle(frame, (x3, y3), 10, (0, 0, 255), -1)

        # Move mouse
        screen_x = int(x1 * screen_w / w)
        screen_y = int(y1 * screen_h / h)
        pyautogui.moveTo(screen_x, screen_y,duration=0.1)

        # Click detection
        distance = math.hypot(x2 - x1, y2 - y1)

        if distance < 40:
            pyautogui.click()
            cv2.putText(frame, "CLICK", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if abs(y1 - y3) < 40:  
            if y3 < y1:  
                pyautogui.scroll(50)
                cv2.putText(frame, "SCROLL UP", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            elif y3 > y1:  
                pyautogui.scroll(-50)
                cv2.putText(frame, "SCROLL DOWN", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()