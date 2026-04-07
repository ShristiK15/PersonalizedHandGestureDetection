def preprocess_skeleton(sequence, target_frames=32):
    import numpy as np
    from scipy import interpolate

    data = sequence.copy()

    # Convert zero frames to NaN
    zero_mask = np.sum(np.abs(data), axis=(1, 2)) == 0
    data[zero_mask] = np.nan

    # Flatten
    original_shape = data.shape
    data = data.reshape(original_shape[0], -1)

    # --- Trim empty frames ---
    valid_frames = ~np.isnan(data).any(axis=1)

    if not valid_frames.any():
        return np.zeros((target_frames, 21, 3), dtype=np.float32)

    first_valid = np.argmax(valid_frames)
    last_valid = len(valid_frames) - np.argmax(valid_frames[::-1])
    data = data[first_valid:last_valid]

    # --- Interpolate missing frames ---
    x = np.arange(len(data))
    is_valid = ~np.isnan(data).any(axis=1)

    if not is_valid.all():
        if np.sum(is_valid) < 2:
            data[:] = data[is_valid][0]
        else:
            f = interpolate.interp1d(
                x[is_valid],
                data[is_valid],
                axis=0,
                kind='linear',
                fill_value="extrapolate"
            )
            data = f(x)

    data = np.nan_to_num(data)

    # --- Resize to 32 frames safely ---
    current_len = len(data)

    if current_len < 2:
        data = np.repeat(data, target_frames, axis=0)

    elif current_len != target_frames:
        x_old = np.linspace(0, 1, current_len)
        x_new = np.linspace(0, 1, target_frames)
        f = interpolate.interp1d(x_old, data, axis=0, kind='linear')
        data = f(x_new)

    # Reshape back
    data = data.reshape(target_frames, 21, 3)

    # Wrist normalization
    wrist = data[:, 0:1, :]
    data = data - wrist

    return data.astype(np.float32)