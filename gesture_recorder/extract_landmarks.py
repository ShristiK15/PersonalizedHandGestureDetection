import numpy as np

def extract_coordinates(results):

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]

    coords = []

    for lm in hand.landmark:
        coords.append([lm.x, lm.y, lm.z])

    return np.array(coords)