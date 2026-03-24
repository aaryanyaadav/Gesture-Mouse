# Virtual Mouse using Hand Tracking

A real-time **gesture-controlled virtual mouse system** built using Computer Vision and AI. This project allows users to control their computer cursor using hand gestures captured via a webcam.

---

##  Features

*  **Cursor Movement** using index finger
*  **Click Action** using thumb + index finger
*  **Scroll Up/Down** using two-finger gesture
*  **Smooth Cursor Control** with motion filtering


---

## Technologies Used

* **Python**
* **OpenCV** – video capture & image processing
* **MediaPipe (Tasks API)** – hand landmark detection
* **PyAutoGUI** – mouse control automation
* **NumPy & Math** – calculations


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/virtual-mouse.git
cd virtual-mouse
```

### 2. Install dependencies

```bash
pip install opencv-python mediapipe pyautogui numpy
```

### 3. Download required model

Download the MediaPipe hand model:

```
hand_landmarker.task
```

Place it inside the project folder.

---

## Usage

Run the script:

```bash
python main.py
```

Press `ESC` to exit.

---

##  Controls

|    Gesture        | Action      |
| ----------------- | ----------- |
|    Index finger   | Move cursor |
|    Thumb + Index  | Click       |
|    Index + Middle | Scroll      |

---

##  Project Structure

```
virtual-mouse/
│
├── main.py
├── hand_landmarker.task
├── requirements.txt
└── README.md
```

##  Use Cases

* Touchless human-computer interaction
* Accessibility for physically challenged users
* Smart home / IoT control systems
* Interactive AI applications

