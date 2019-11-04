import functools

import numpy as np


@functools.lru_cache(maxsize=1)
def get_colormap(max_class_id=256):
    palette = np.array(
        [
            [255, 255, 255], [255, 0, 0], [0, 255, 0],
            [0, 0, 255], [255, 0, 255], [0, 255, 255],
            [255, 255, 0], [128, 0, 0], [0, 128, 0],
            [0, 0, 128], [128, 128, 0], [128, 0, 128],
            [0, 128, 128], [128, 128, 128], [255, 128, 0],
            [128, 255, 0], [255, 0, 128], [128, 0, 255],
            [0, 255, 128], [0, 128, 255]
        ], dtype=np.uint8
    )

    indices = np.random.choice(range(palette.shape[0]), size=max_class_id, replace=False)

    return palette[indices, :]


def blend_rgb_with_mask(rgb, mask):
    if rgb.shape != mask.shape:
        raise RuntimeError(f'Mismatching shapes; {rgb.shape}, {mask.shape}')

    alpha = 0.4
    beta = 1.0 - alpha

    blended = alpha * rgb + beta * mask

    return blended
