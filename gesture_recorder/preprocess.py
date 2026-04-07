import numpy as np

def preprocess_sequence(seq):

    seq = np.array(seq)

    # wrist normalization
    first_frame_wrist = seq[0:1, 0:1, :]
    seq = seq - first_frame_wrist

    # scale normalization
    scale = np.max(np.abs(seq))

    if scale > 0:
        seq = seq / scale

    return seq.astype("float32")