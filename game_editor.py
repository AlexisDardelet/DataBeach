"""Module for editing beach volleyball game videos."""
import sys
import cv2


class GameEditor:
    """
    Class to edit a video of a beach volleyball game,
    from the raw video to segmented clips of each point.
    It also returns dictionaries with information about the score.
    """

    def __init__(
        self,
        video_dir: str = None,
        video_path: str = None,
        output_dir: str = None,
    ) -> None:
        """
        Initializes the GameEditor class.

        Args:
            video_dir (str): Path to the directory containing
                the raw game video(s).
            video_path (str): Path to a specific video file.
            output_dir (str, optional): Path to the output directory.
        """
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.montage_actions = {}
        self.video_path = video_path

    def montage_operation(
        self, play_speed: float = 1.0
    ) -> dict:
        """
        Records the montage actions for video pre-processing.

        Args:
            play_speed (float): Video playback speed. Defaults to 1.0.

        Returns:
            dict: Dictionary with keys 'start_frame',
                'last_frame', 'rotation_state'.
        """
        # Ensure a video path has been provided before proceeding
        if self.video_path is None:
            raise ValueError(
                "video_path must be set before calling "
                "montage_operation()."
            )

        montage_actions = {}
        starting_game_frame = 0

        # Define the help overlay text shown on each frame
        help_lines = [
            "Keys:",
            "q : quit",
            "space : pause/resume",
            "0 : start of match",
            "+ : speed up",
            "- : speed down",
            "r : rotate right",
            "l : rotate left",
        ]

        # Monkey-patch cv2.imshow to overlay help text on every displayed frame
        _orig_imshow = cv2.imshow

        def _imshow_with_help(winname, frame):
            if frame is not None:
                x, y = 30, 120
                for i, line in enumerate(help_lines):
                    cv2.putText(
                        frame, line,
                        (x, y + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2,
                        cv2.LINE_AA,
                    )
                _orig_imshow(winname, frame)

        cv2.imshow = _imshow_with_help

        # Open the video and retrieve total frame count for last-frame default
        cap = cv2.VideoCapture(self.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last_game_frame = (
            frame_count - 1 if frame_count > 0 else None
        )
        print(f"Last frame index: {last_game_frame}")

        # Helper to adjust waitKey delay based on current playback speed
        def _wait_key_fast(ms):
            adj = max(1, int(ms / play_speed))
            return cv2.waitKey(adj)

        # Validate that the video was opened successfully
        if not cap.isOpened():
            print("Error: unable to open the video.")
            sys.exit()

        # Initialize playback state variables
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = 0
        paused = False
        rotation_state = 0
        ret = False

        try:
            # Main playback loop: read, transform, display, and handle input
            while cap.isOpened():
                # Read the next frame only when not paused
                if not paused:
                    ret, frame = cap.read()

                    if not ret:
                        print("End of video or read error.")
                        break

                    # Apply rotation based on the current rotation state
                    if rotation_state == 90:
                        frame = cv2.rotate(
                            frame, cv2.ROTATE_90_CLOCKWISE
                        )
                    elif rotation_state == 180:
                        frame = cv2.rotate(
                            frame, cv2.ROTATE_180
                        )
                    elif rotation_state == 270:
                        frame = cv2.rotate(
                            frame,
                            cv2.ROTATE_90_COUNTERCLOCKWISE,
                        )

                    # Overlay playback speed indicator on the frame
                    cv2.putText(
                        frame,
                        f"Playback speed: x{play_speed:.1f}",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA,
                    )

                    frame_number += 1

                # Show a pause indicator when the video is paused
                if paused and ret:
                    cv2.putText(
                        frame, "|| PAUSE ||",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA,
                    )

                # Display the current frame (with help overlay via patch)
                if ret:
                    cv2.imshow(
                        f'{self.video_path}', frame
                    )

                # Handle keyboard input for playback control
                key = _wait_key_fast(30) & 0xFF
                if key == ord('q'):
                    # Quit the playback loop
                    break
                if key == ord(' '):
                    # Toggle pause/resume
                    paused = not paused
                elif key == ord('0'):
                    # Mark the current frame as the start of the match
                    starting_game_frame = frame_number
                    start_time = starting_game_frame / fps
                    print(
                        f"Match start marked at frame "
                        f"{starting_game_frame}, "
                        f"i.e. {start_time:.2f} seconds"
                    )
                    break
                elif key == ord('+'):
                    # Increase playback speed
                    play_speed += 0.5
                elif key == ord('-'):
                    # Decrease playback speed (minimum 0.5x)
                    play_speed = max(0.5, play_speed - 0.5)
                elif key == ord('r'):
                    # Rotate the video 90° clockwise
                    rotation_state = (
                        rotation_state + 90
                    ) % 360
                elif key == ord('l'):
                    # Rotate the video 90° counter-clockwise
                    rotation_state = (
                        rotation_state - 90
                    ) % 360

        finally:
            # Release resources and restore the original cv2.imshow
            cap.release()
            cv2.destroyAllWindows()
            cv2.imshow = _orig_imshow

        # Store and return the collected montage metadata
        montage_actions = {
            'start_frame': starting_game_frame,
            'last_frame': last_game_frame,
            'rotation_state': rotation_state,
        }
        self.montage_actions = montage_actions
