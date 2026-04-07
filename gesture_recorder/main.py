import cv2

from mediapipe_setup import hands, mp_draw, mp_hands
from extract_landmarks import extract_coordinates
from sequence_buffer import update_sequence, sequence
from preprocess import preprocess_sequence
from dataset_utils import save_sample, update_labels

gesture_name = input("Enter gesture name: ")

sample_id = 0
recording = False

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(img)

    coords = extract_coordinates(results)

    if coords is not None:

        mp_draw.draw_landmarks(
            frame,
            results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS
        )

        if recording:

            seq = update_sequence(coords)

            if len(seq) == 32:

                seq = preprocess_sequence(seq)

                path = save_sample(seq, gesture_name, sample_id)

                update_labels(path, gesture_name)

                print("Saved:", path)

                sample_id += 1

                sequence.clear()

                recording = False

    cv2.putText(frame,
                "Press R to record",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2)

    cv2.imshow("Gesture Recorder", frame)

    key = cv2.waitKey(1)

    if key == ord('r'):
        recording = True

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()