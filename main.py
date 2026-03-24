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

# Smoothening variables
prev_x, prev_y = 0, 0
smoothening = 5

# Frame reduction (stability)
frame_reduction = 100

# Disable failsafe
pyautogui.FAILSAFE = False

# Setup MediaPipe
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

        # Landmarks
        x1 = int(hand[8].x * w)   # Index
        y1 = int(hand[8].y * h)

        x2 = int(hand[4].x * w)   # Thumb
        y2 = int(hand[4].y * h)

        x3 = int(hand[12].x * w)  # Middle
        y3 = int(hand[12].y * h)

        # Draw points
        cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
        cv2.circle(frame, (x2, y2), 10, (0, 255, 0), -1)
        cv2.circle(frame, (x3, y3), 10, (0, 0, 255), -1)

        # Boundary control
        x1 = max(frame_reduction, min(w - frame_reduction, x1))
        y1 = max(frame_reduction, min(h - frame_reduction, y1))

        # Map to screen
        screen_x = int((x1 - frame_reduction) * screen_w / (w - 2 * frame_reduction))
        screen_y = int((y1 - frame_reduction) * screen_h / (h - 2 * frame_reduction))

        # Smooth movement
        curr_x = prev_x + (screen_x - prev_x) / smoothening
        curr_y = prev_y + (screen_y - prev_y) / smoothening

        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        # Distance (for click)
        distance = math.hypot(x2 - x1, y2 - y1)

        # Scroll gesture (2 fingers)
        if abs(y1 - y3) < 40:
            if y3 < y1:
                pyautogui.scroll(40)
                cv2.putText(frame, "SCROLL UP", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            elif y3 > y1:
                pyautogui.scroll(-40)
                cv2.putText(frame, "SCROLL DOWN", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Click (only when NOT scrolling)
        elif distance < 40:
            pyautogui.click()
            cv2.putText(frame, "CLICK", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()