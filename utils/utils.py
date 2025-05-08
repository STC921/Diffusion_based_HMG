def get_motion_shape(dataset="h36m"):
    if dataset == "h36m":
        data = (128, 125, 17, 3)
        num_frames = 240
        keep_ratio = 0.20
    elif dataset == "humaneva":
        data = (128, 75, 15, 3)
        num_frames = 75
        keep_ratio = 0.15
    else:
        data = None
        num_frames = None
        keep_ratio = None

    return data, num_frames, keep_ratio