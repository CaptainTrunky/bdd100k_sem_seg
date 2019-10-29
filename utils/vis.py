import functools

import numpy as np


@functools.lru_cache(maxsize=1)
def get_colormap(max_class_id=256):
    colormap = np.zeros((max_class_id, 3), dtype=np.uint8)

    colormap[0, :] = [255, 0, 0]
    colormap[6, :] = [128, 128, 0]
    colormap[11, :] = [0, 255, 0]
    colormap[13, :] = [0, 0, 255]
    colormap[14, :] = [0, 128, 128]
    colormap[255, :] = [255, 255, 255] 

    return colormap
