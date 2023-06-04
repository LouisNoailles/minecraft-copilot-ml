import numpy as np
import pytest

from minecraft_copilot_ml.data_processing.traverse_3d_block import (
    traverse_3d_array,
)


def test_traverse_3d_array() -> None:
    block_map = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [10, 11, 12],
            ],
        ]
    )
    sliding_window_width = 2
    sliding_window_height = 2
    sliding_window_depth = 2
    new_blocks = traverse_3d_array(
        block_map=block_map,
        sliding_window_width=sliding_window_width,
        sliding_window_height=sliding_window_height,
        sliding_window_depth=sliding_window_depth,
    )
    assert len(new_blocks) == 36
    assert new_blocks[0].shape == (
        sliding_window_width,
        sliding_window_height,
        sliding_window_depth,
    )


def test_traverse_3d_bad_shape() -> None:
    block_map = np.empty((0, 0))
    with pytest.raises(ValueError):
        traverse_3d_array(
            block_map=block_map,
            sliding_window_width=2,
            sliding_window_height=2,
            sliding_window_depth=2,
        )
