from typing import List
import numpy as np


def traverse_3d_array(
    block_map: np.ndarray,
    sliding_window_width: int = 16,
    sliding_window_height: int = 16,
    sliding_window_depth: int = 16,
) -> List[np.ndarray]:
    if len(block_map.shape) != 3:
        raise ValueError("block_map must be a 3d array")
    block_list = []
    for x in range(-sliding_window_width + 1, block_map.shape[0]):
        for y in range(-sliding_window_height + 1, block_map.shape[1]):
            for z in range(-sliding_window_depth + 1, block_map.shape[2]):
                new_block = np.zeros(
                    (sliding_window_width, sliding_window_height, sliding_window_depth),
                    dtype=np.int32,
                )
                for i in range(sliding_window_width):
                    for j in range(sliding_window_height):
                        for k in range(sliding_window_depth):
                            if (
                                x + i >= 0
                                and x + i < block_map.shape[0]
                                and y + j >= 0
                                and y + j < block_map.shape[1]
                                and z + k >= 0
                                and z + k < block_map.shape[2]
                            ):
                                new_block[i, j, k] = block_map[x + i, y + j, z + k]
                            else:
                                new_block[i, j, k] = -1
                block_list.append(new_block)
    return block_list
