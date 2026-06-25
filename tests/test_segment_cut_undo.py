"""Regression tests for the point-segmentation undo ("back to last *SWITCH*").

Reproduces the off-by-one that Alexis reported during the 2026-06-24 review:
pressing '4' to rewind to the last side switch shifted the whole sequence by
one frame index. Runs with plain ``python`` - no OpenCV / pandas / display
needed, so it works in CI and on a Mac without nvenc.

    python tests/test_segment_cut_undo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from segment_undo import rewind_seek_target  # noqa: E402

# cv2.CAP_PROP_POS_FRAMES constant value, hard-coded so the test stays
# dependency-free.
CAP_PROP_POS_FRAMES = 1


class FakeCapture:
    """Minimal stand-in mimicking cv2.VideoCapture frame-position semantics.

    ``read()`` returns the 0-based index of the frame it just decoded and
    advances the position; ``set(CAP_PROP_POS_FRAMES, n)`` makes the next
    ``read()`` return frame ``n``. This is exactly the contract
    ``cv2_point_segment_cut`` relies on.
    """

    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.pos = 0

    def read(self):
        if self.pos >= self.n_frames:
            return False, None
        idx = self.pos
        self.pos += 1
        return True, idx

    def set(self, prop, value):
        if prop == CAP_PROP_POS_FRAMES:
            self.pos = int(value)

    def get(self, prop):
        if prop == CAP_PROP_POS_FRAMES:
            return self.pos
        return 0


def displayed_index_after_rewind(seek_target):
    """True index of the frame shown right after rewinding to ``seek_target``."""
    cap = FakeCapture(n_frames=300)
    cap.set(CAP_PROP_POS_FRAMES, seek_target)
    ok, idx = cap.read()
    assert ok
    return idx


# A switch recorded while watching true frame 48. frame_number is incremented
# AFTER the read, so the value stored in temp_list is 49.
SWITCH_TRUE_INDEX = 48
RECORDED_SWITCH_FRAME = SWITCH_TRUE_INDEX + 1


def test_old_behaviour_was_off_by_one():
    # The previous code seeked straight to the recorded frame and re-watched
    # the frame AFTER the switch -> the reported one-index shift.
    assert displayed_index_after_rewind(RECORDED_SWITCH_FRAME) == SWITCH_TRUE_INDEX + 1


def test_fix_lands_exactly_on_the_switch_frame():
    target = rewind_seek_target(RECORDED_SWITCH_FRAME)
    assert displayed_index_after_rewind(target) == SWITCH_TRUE_INDEX


def test_frame_number_resyncs_to_recording_convention():
    # After the fixed rewind re-reads the switch frame, frame_number
    # (= cap.get(POS)) must equal the switch's recorded Frame, so every action
    # recorded afterwards keeps the same index as before the rewind.
    cap = FakeCapture(n_frames=300)
    cap.set(CAP_PROP_POS_FRAMES, rewind_seek_target(RECORDED_SWITCH_FRAME))
    cap.read()
    assert cap.get(CAP_PROP_POS_FRAMES) == RECORDED_SWITCH_FRAME


def test_rewind_target_never_negative():
    # Defensive: a switch stored at frame 0 must not seek to -1.
    assert rewind_seek_target(0) == 0


if __name__ == "__main__":
    failures = 0
    for name in sorted(n for n in dir() if n.startswith("test_")):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    if failures:
        sys.exit(f"\n{failures} test(s) failed")
    print("\nAll undo regression tests passed.")
