"""Helpers for the point-segmentation undo ("back to last *SWITCH*") logic.

Kept dependency-free so the behaviour can be unit-tested without OpenCV,
pandas or a display (see tests/test_segment_cut_undo.py).
"""


def rewind_seek_target(switch_frame: int) -> int:
    """Return the 0-based capture position to seek so the NEXT ``cap.read()``
    returns exactly the ``*SWITCH*`` frame.

    In ``cv2_point_segment_cut`` the recorded ``'Frame'`` equals
    ``frame_number``, which is incremented *after* every ``cap.read()``. The
    frame the user was watching when the switch was recorded therefore sits at
    the 0-based capture position ``switch_frame - 1``. Seeking straight to
    ``switch_frame`` lands one frame past the switch (the off-by-one that
    shifts every action recorded after a rewind).
    """
    return max(0, int(switch_frame) - 1)
