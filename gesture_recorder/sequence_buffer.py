sequence = []

def update_sequence(coords):

    global sequence

    sequence.append(coords)

    if len(sequence) > 32:
        sequence.pop(0)

    return sequence