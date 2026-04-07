import cv2
import mediapipe as mp
import torch
import numpy as np
from collections import deque
import pyautogui
import time

from meta_ddnet_model import DDNet  # Your model

# =========================
# CONFIGURATION
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_FRAMES = 32
CONFIDENCE_THRESHOLD = 0.85
COOLDOWN_TIME = 1.0  # Seconds to wait between triggering actions

# Map your model's integer outputs to specific OS actions
# (Update these based on what gestures you fine-tuned for!)
GESTURE_ACTIONS = {
    0: lambda: pyautogui.press('space'),      # e.g., Play/Pause
    1: lambda: pyautogui.hotkey('ctrl', 't'), # e.g., Open New Tab
    2: lambda: pyautogui.scroll(-500),        # e.g., Scroll Down
    3: lambda: pyautogui.press('esc')         # e.g., Exit full screen
}

# =========================
# INITIALIZATION
# =========================
# Load MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,          # Assuming one-handed control for now
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Load Model
model = DDNet().to(DEVICE)
model.load_state_dict(torch.load("weights/meta_model_seed_7.pth"))
model.eval()

# Rolling buffer for the last 32 frames
frame_buffer = deque(maxlen=TARGET_FRAMES)
last_action_time = 0

print("Starting Webcam Interface... Press 'q' to quit.")
cap = cv2.VideoCapture(0)

# =========================
# THE REAL-TIME LOOP
# =========================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Extract skeletons
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract the 21 joints (x, y, z)
        joints = []
        for lm in hand_landmarks.landmark:
            joints.append([lm.x, lm.y, lm.z])
            
        frame_buffer.append(np.array(joints))
        
        # Draw skeleton on screen for visual feedback
        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    else:
        # If no hand is seen, we can optionally clear the buffer or append zeros
        # For OS control, clearing it prevents accidental triggers from stale data
        frame_buffer.clear()

    # =========================
    # INFERENCE & CONTROL
    # =========================
    # Only predict if we have exactly 32 frames
    if len(frame_buffer) == TARGET_FRAMES and (time.time() - last_action_time > COOLDOWN_TIME):
        
        # 1. Convert buffer to numpy array: shape (32, 21, 3)
        sequence = np.array(frame_buffer)
        
        # 2. Normalize exactly like your training preprocessing!
        # Anchor the whole sequence to the first frame's wrist
        first_frame_wrist = sequence[0:1, 0:1, :]
        sequence = sequence - first_frame_wrist
        
        # 3. Format for PyTorch (Batch=1, Frames=32, Channels=63)
        sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # 4. Predict
        with torch.no_grad():
            outputs = model(sequence)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            conf_val = confidence.item()
            pred_class = predicted.item()

        # 5. Trigger OS Action
        if conf_val > CONFIDENCE_THRESHOLD:
            print(f"Action Triggered! Gesture {pred_class} (Confidence: {conf_val:.2f})")
            
            if pred_class in GESTURE_ACTIONS:
                GESTURE_ACTIONS[pred_class]() # Execute the mapped function
                
            last_action_time = time.time() # Reset cooldown
            frame_buffer.clear()           # Clear buffer so we don't rapid-fire

    cv2.imshow("Alternative Input Controller", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()