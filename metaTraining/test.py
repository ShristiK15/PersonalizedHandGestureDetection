import cv2
import mediapipe as mp
import time
import numpy as np
import torch
import torch.nn.functional as F
import os
from scipy import interpolate
import pyautogui
import os
import webbrowser

# =========================
# CONFIG
# =========================
TARGET_FRAMES = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIBRARY_FILE = "gesture_library.pt"

COSINE_THRESHOLD = 0.75
CONFIDENCE_THRESHOLD = 0.10
MARGIN_THRESHOLD = 0.01
TEMPERATURE = 1.2

# =========================
# ACTION CONTROL
# =========================
ACTION_COOLDOWN = 2.0
last_action_time = 0

# =========================
# MOUSE SMOOTHING & STATE
# =========================
prev_x, prev_y = 0, 0
SMOOTHING = 5
is_pinched = False
is_scrolling = False
scroll_start_y = 0
last_click_time = 0
DOUBLE_CLICK_DELAY = 0.4
last_right_click_time = 0 # NEW: Tracker for right click
CLICK_DELAY = 0.5         # NEW: Cooldown between clicks
# =========================
# MOTION GATING STATE
# =========================
motion_state = "WAITING"
prev_wrist = None
VEL_THRESHOLD = 0.015  # Hand velocity threshold for gating
ready_start_time = 0

# =========================
# UI STATE
# =========================
last_label = "None"

# =========================
# LOAD MODEL
# =========================
from meta_ddnet_model import DDNet

model = DDNet().to(DEVICE)
model.load_state_dict(torch.load("meta_weights/meta_model_seed_7.pth", map_location=DEVICE))
model.eval()

# =========================
# LOAD LIBRARY
# =========================
if os.path.exists(LIBRARY_FILE):
    gesture_library = torch.load(LIBRARY_FILE)
    
    print("\n" + "="*40)
    print("📦 GESTURE LIBRARY LOADED")
    print("="*40)
    
    if gesture_library:
        # Loop through and print each gesture and its action
        for name, data in gesture_library.items():
            action = data.get("action", "No action assigned")
            # ljust(15) keeps the arrows aligned neatly in the terminal
            print(f" 🔹 {name.ljust(15)} ->  {action}")
    else:
        print(" Library file exists but is empty.")
        
    print("="*40 + "\n")
    
else:
    gesture_library = {}
    print("\n📦 No existing gesture library found. Starting fresh.\n")

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# =========================
# GLOBAL STATE
# =========================
gesture_data = []
recording = False
countdown = False
mode = "IDLE"  

start_time = 0
WIN_NAME = "Gesture System"

# =========================
# UI FUNCTIONS
# =========================
def draw_cursor(frame, x, y):
    cv2.circle(frame, (x, y), 15, (0, 255, 0), 2)
    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

def draw_ui(frame, mode, motion_state, last_label):
    cv2.rectangle(frame, (10, 10), (360, 180), (30, 30, 30), -1)

    cv2.putText(frame, f"Mode: {mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display motion gating state if in PREDICT mode
    status_text = motion_state if mode == "PREDICT" else ("Recording" if recording else "Idle")
    cv2.putText(frame, f"Status: {status_text}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Gesture: {last_label}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, "R:Record", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "P:Predict", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "M:Mouse", (150, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Q:Quit", (150, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# =========================
# MOUSE CONTROL
# =========================
def control_mouse(hand_landmarks, frame):
    global prev_x, prev_y

    h, w, _ = frame.shape
    screen_w, screen_h = pyautogui.size()

    x = hand_landmarks.landmark[8].x
    y = hand_landmarks.landmark[8].y

    frame_x = int(x * w)
    frame_y = int(y * h)

    screen_x = int(x * screen_w)
    screen_y = int(y * screen_h)

    # Exponential Moving Average for smoothing
    curr_x = prev_x + (screen_x - prev_x) / SMOOTHING
    curr_y = prev_y + (screen_y - prev_y) / SMOOTHING

    pyautogui.moveTo(curr_x, curr_y)

    prev_x, prev_y = curr_x, curr_y

    return frame_x, frame_y, screen_y

def finger_distance(hand, p1, p2):
    x1, y1 = hand.landmark[p1].x, hand.landmark[p1].y
    x2, y2 = hand.landmark[p2].x, hand.landmark[p2].y
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def get_finger_states(hand_landmarks):
    tips = [8, 12, 16, 20]  
    states = []
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            states.append(1) 
        else:
            states.append(0) 
    return states 

# =========================
# PREPROCESS + MODEL
# =========================
def preprocess_gesture(sequence, target_frames=32):
    data = np.array(sequence, dtype=np.float32)

    if len(data) < 5:
        return None

    # 1. VELOCITY FILTERING: Trims dead space at start/end
    velocity = np.linalg.norm(data[1:] - data[:-1], axis=(1, 2))
    valid = velocity > 0.002

    if np.sum(valid) < 5:
        return None

    start = np.argmax(valid)
    end = len(valid) - np.argmax(valid[::-1])
    data = data[start:end+1]

    T = len(data)
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_frames)

    data = data.reshape(T, -1)

    try:
        f = interpolate.interp1d(x_old, data, axis=0)
        data = f(x_new)
    except:
        return None

    data = data.reshape(target_frames, 21, 3)

    # 2. SPATIAL NORMALIZATION: Makes model immune to camera distance/position
    wrist = data[:, 0:1, :]
    data = data - wrist  # Center to wrist

    max_val = np.max(np.linalg.norm(data, axis=2))
    if max_val > 0:
        data = data / max_val  # Scale between 0 and 1

    return data.astype(np.float32)

def get_embedding(sequence):
    tensor = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return F.normalize(model(tensor), dim=1)

def match_gesture(query_emb):
    if len(gesture_library) == 0:
        return "None", 0

    best_label = "Unknown"
    best_score = -1

    for name, data in gesture_library.items():
        score = torch.mm(query_emb, data["prototype"].t()).item()
        if score > best_score:
            best_score = score
            best_label = name

    # ENFORCE THRESHOLD: Ignore low-confidence matches
    if best_score < COSINE_THRESHOLD:
        return "None", best_score

    return best_label, best_score

# =========================
# ACTION EXECUTION
# =========================
def execute_action(label):
    global last_action_time

    if label not in gesture_library or label == "None":
        return

    action = gesture_library[label].get("action", None)
    if action is None:
        return

    if time.time() - last_action_time < ACTION_COOLDOWN:
        return

    print(f"Executing action: {action}")

    try:
        if isinstance(action, str):
            if '+' in action:
                keys = action.split('+')
                pyautogui.hotkey(*keys)
                last_action_time = time.time()
                return

            parts = action.split()
            cmd = parts[0].lower()

            # -------------------------
            #  MOUSE & KEYBOARD
            # -------------------------
            if cmd == "hotkey":
                pyautogui.hotkey(*parts[1:])
            elif cmd == "mouse_click":
                pyautogui.click()
            elif cmd == "scroll":
                pyautogui.scroll(int(parts[1]))
            elif cmd == "screenshot":
                pyautogui.screenshot(f"s_{int(time.time())}.png")
            
            # -------------------------
            #  MEDIA CONTROLS
            # -------------------------
            elif cmd in ["volumeup", "volumedown", "volumemute", "playpause", "nexttrack", "prevtrack"]:
                pyautogui.press(cmd)
            
            # -------------------------
            #  NEW: APP & SYSTEM COMMANDS
            # -------------------------
            elif cmd == "open":
                # Example: "open https://github.com" or "open C:\Users\Document.pdf"
                target = " ".join(parts[1:])
                if target.startswith("http"):
                    webbrowser.open(target)
                else:
                    # Natively opens files/folders with their default application
                    os.startfile(target) 
            
            elif cmd == "run":
                # Example: "run calc" or "run notepad"
                app = " ".join(parts[1:])
                os.system(f"start {app}")
            
            elif cmd == "desktop":
                # Minimizes all windows to show the desktop
                pyautogui.hotkey('win', 'd')
                
            elif cmd == "lock":
                # Locks the computer screen 
                os.system("rundll32.exe user32.dll,LockWorkStation")

            else:
                print(f"Unknown string command: {cmd}")

        elif isinstance(action, dict):
            # ... (Keep your existing dict logic here if you still use it) ...
            pass
                
    except Exception as e:
        print(" Error:", e)

    last_action_time = time.time()

def run_prediction(data_sequence):
    global last_label
    processed = preprocess_gesture(data_sequence)
    if processed is not None:
        emb = get_embedding(processed)
        label, score = match_gesture(emb)
        if label != "None":
            last_label = label
            execute_action(label)

# =========================
# MAIN LOOP
# =========================
cap = cv2.VideoCapture(0)

print("R Record | P Predict | M Mouse | Q Quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        mode = "RECORD"; countdown = True; gesture_data = []; start_time = time.time()
    elif key == ord('p'):
        mode = "PREDICT"; motion_state = "WAITING"; gesture_data = []
    elif key == ord('m'):
        mode = "MOUSE"
    elif key == ord('q'):
        break

    cursor_x, cursor_y = None, None

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        pts = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
        
        # Calculate Wrist Velocity for Motion Gating
        curr_wrist = np.array([hand.landmark[0].x, hand.landmark[0].y])
        if prev_wrist is not None:
            velocity = np.linalg.norm(curr_wrist - prev_wrist)
        else:
            velocity = 0
        prev_wrist = curr_wrist

        # =========================
        # INTENTIONAL MOTION GATING (PREDICT MODE)
        # =========================
        if mode == "PREDICT":
            if velocity > VEL_THRESHOLD:
                if motion_state == "RECORDING" and len(gesture_data) > 10:
                    run_prediction(gesture_data)
                motion_state = "WAITING"
                gesture_data = []
            
            elif velocity <= VEL_THRESHOLD:
                if motion_state == "WAITING":
                    motion_state = "READY"
                    ready_start_time = time.time()
                elif motion_state == "READY" and (time.time() - ready_start_time) > 0.2:
                    motion_state = "RECORDING"

            if motion_state == "RECORDING":
                gesture_data.append(pts)

        # =========================
        # RECORD MODE (Legacy Countdown)
        # =========================
        elif mode == "RECORD":
            if countdown:
                if time.time() - start_time < 3:
                    cv2.putText(frame, "Get Ready", (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                else:
                    countdown = False; recording = True; start_time = time.time()
            elif recording:
                if time.time() - start_time < 3:
                    gesture_data.append(pts)
                else:
                    recording = False
                    processed = preprocess_gesture(gesture_data)
                    if processed is not None:
                        emb = get_embedding(processed)
                        name = input("Gesture name: ")
                        
                        # =========================
                        #  PROTOTYPE UPDATER
                        # =========================
                        if name in gesture_library:
                            print(f"\n '{name}' already exists! Skipping action input.")
                            
                            # Keep the existing action
                            action = gesture_library[name]["action"]
                            
                            # Append the new sample
                            gesture_library[name]["samples"].append(emb)
                            
                            # Recalculate the prototype by averaging all samples
                            all_samples = torch.cat(gesture_library[name]["samples"], dim=0)
                            new_prototype = torch.mean(all_samples, dim=0, keepdim=True)
                            
                            # Re-normalize to maintain cosine similarity scale
                            gesture_library[name]["prototype"] = F.normalize(new_prototype, dim=1)
                            
                            print(f"Updated prototype for '{name}' (Total samples: {len(gesture_library[name]['samples'])})")
                            
                        else:
                            # New gesture: ask for the action trigger
                            action = input("Action (e.g., desktop, run calc, volumeup): ")
                            gesture_library[name] = {
                                "samples": [emb],
                                "prototype": emb,
                                "action": action
                            }
                            print(f"Created new gesture '{name}'.")
                            
                        torch.save(gesture_library, LIBRARY_FILE)
                        print("-" * 40 + "\n")

        # =========================
        # MOUSE MODE
        # =========================
        elif mode == "MOUSE":
            cursor_x, cursor_y, screen_y = control_mouse(hand, frame)
            fingers = get_finger_states(hand)
            index, middle, ring, pinky = fingers
            thumb_index_dist = finger_distance(hand, 4, 8)

            #  PINCH CLICK & DRAG (Debounced)
            if thumb_index_dist < 0.04:
                if not is_pinched:
                    pyautogui.mouseDown()
                    is_pinched = True
            else:
                if is_pinched:
                    pyautogui.mouseUp()
                    is_pinched = False

            #  DOUBLE CLICK (Index + Middle Tap)
            if index == 1 and middle == 1 and ring == 0:
                if not is_scrolling: 
                    is_scrolling = True
                    scroll_start_y = screen_y
                else:
                    # Dynamic Scroll: Scale scroll amount by hand vertical movement
                    delta_y = scroll_start_y - screen_y
                    if abs(delta_y) > 20: 
                        pyautogui.scroll(int(delta_y * 1.5))
                        scroll_start_y = screen_y 
            else:
                is_scrolling = False

            #  RIGHT CLICK (Pinky Up, Index/Middle Down)
            if pinky == 1 and index == 0 and middle == 0:
                if time.time() - last_right_click_time > CLICK_DELAY:
                    pyautogui.rightClick()
                    last_right_click_time = time.time()


        # DRAW HAND
        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    else:
        prev_wrist = None 
        if is_pinched: # Failsafe release
            pyautogui.mouseUp()
            is_pinched = False

    # DRAW CURSOR & UI
    if cursor_x is not None:
        draw_cursor(frame, cursor_x, cursor_y)
    draw_ui(frame, mode, motion_state, last_label)

    cv2.imshow(WIN_NAME, frame)

cap.release()
cv2.destroyAllWindows()