import numpy as np
import math
from itertools import chain


def div(img: np.ndarray,
        window_shape: tuple = (256, 256, 3)):
    """Checks if the image is divisible by the window shape."""
    y = window_shape[0]
    x = window_shape[1]
    c = window_shape[2]

    if img.shape[0] % y != 0:
        y_out = math.ceil(img.shape[0] / y) * y
    else:
        y_out = img.shape[0]
    if img.shape[1] % x != 0:
        x_out = math.ceil(img.shape[1] / x) * x
    else:
        x_out = img.shape[1]

    return y_out, x_out, c


def padding(array: np.ndarray, desired_y: int, desired_x: int, constant: int = 0):
    """
    Pads an rgb(a) array to a desired size with a constant.
    """

    y = array.shape[0]
    x = array.shape[1]

    a = (desired_x - y) // 2
    aa = desired_x - a - y

    b = (desired_y - x) // 2
    bb = desired_y - b - x

    padded_array = np.pad(array, pad_width=((a, aa), (b, bb), (0, 0)), mode='constant', constant_values=constant)

    # TODO: Change y and x.
    return padded_array


def remove_boxes_score(patch_scores, score):
    """Returns the masks of provided score values that are >= the specified score."""
    indices = []

    for scores in patch_scores:
        indices.append(scores >= score)
    return indices


def remove_boxes_edge(patch_boxes, pixels):
    """Returns the indices of provided boxes that are within the specified area."""
    indices = []

    for boxes in patch_boxes:
        # TODO: remove hardcoded size
        x_max = 256 - pixels
        x_min = pixels
        y_max = 256 - pixels
        y_min = pixels

        keep_boxes = []

        for idx, bb in enumerate(boxes):
            x1 = bb[0]
            y1 = bb[1]
            x2 = bb[2]
            y2 = bb[3]

            if x1 >= x_min and y1 >= y_min and x2 <= x_max and y2 <= y_max:
                keep_boxes.append(idx)
        indices.append(np.array(keep_boxes))
    return indices


def translate_boxes_to_full_image(patch_indices, patch_boxes):
    boxes_new = []

    for patch_idx, patch_box in zip(patch_indices, patch_boxes):
        y_factor = patch_idx[0]
        x_factor = patch_idx[1]

        patch_box_new = []
        for bb in patch_box:
            bb_new = bb * 1  # create a new np.array
            bb_new[0] += x_factor * 128  # x1
            bb_new[1] += y_factor * 128  # y1
            bb_new[2] += x_factor * 128  # x2
            bb_new[3] += y_factor * 128  # y2
            patch_box_new.append(bb_new)

        if patch_box_new:
            boxes_new.append(np.stack(patch_box_new))
        else:
            boxes_new.append(np.empty(shape=(0, 4), dtype=np.float32))

    return boxes_new


def flatten_list(list_of_lists):
    return np.array(list(chain(*list_of_lists)))
